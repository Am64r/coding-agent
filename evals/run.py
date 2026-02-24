import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agent_loop"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from agent import OpenAIClient
from .harness import EvalHarness
from .tasks import ALL_TASKS, TASK_MAP
from .task import EvalTask, TaskResult, COST_PER_1K
from .command_runner import HostCommandRunner, DockerCommandRunner, build_docker_image
from .verifier import set_command_runner
import tool_library
from tool_gen.generator import generate_tool
from tool_gen.pipeline import _validate_tool_code


AUGMENTED_SYSTEM_PROMPT = """\
You are a coding agent. You have tools to read files, write files, and run shell commands.
You also have specialized code-generation tools. Check your full tool list — if a specialized \
tool matches the task, USE IT instead of writing code from scratch. The specialized tools \
generate correct, well-tested code.

All files you create and shell commands you run operate inside the workspace: agent/created_files/
Use relative paths (e.g. "solution.py") and they will be placed there automatically.

Work step by step:
1. Read any existing files to understand the codebase
2. Check if any of your specialized tools can generate the code you need
3. If so, call the tool, then write the result to a file
4. If not, write the code yourself
When the task is complete, give a clear summary of what you did without calling any more tools.
{tool_examples}\
"""

DEFAULT_DOCKER_IMAGE = "coding-agent-evals:latest"
DEFAULT_DOCKERFILE = Path(__file__).parent / "Dockerfile.benchmark"
DOCKER_SMOKE_TASK_IDS = ["hello_world", "rest_api_client", "fix_race_condition"]


def _build_tool_examples_section(usage_examples):
    if not usage_examples:
        return ""
    lines = ["\n## Specialized Tool Usage Examples\n"]
    for name, example in usage_examples.items():
        lines.append(f"### {name}")
        lines.append(example.strip())
        lines.append("")
    return "\n".join(lines)


def _parse_config(spec):
    if spec.endswith("+tools"):
        return spec[:-6], True
    return spec, False


def _generation_cost(model, input_tokens, output_tokens):
    rates = COST_PER_1K.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens / 1000) * rates["input"] + (output_tokens / 1000) * rates["output"]


def _extract_agent_observable_signals(task_result, max_chars=3000):
    lines = []
    for tc in task_result.trajectory:
        if tc.name != "run_shell":
            continue
        output = tc.result or ""
        if any(token in output for token in ("Traceback", "AssertionError", "FAILED", "Error:", "Exit code:")):
            cmd = tc.args.get("command", "") if isinstance(tc.args, dict) else ""
            lines.append(f"$ {cmd}\n{output[:800]}")
    if not lines:
        return "No explicit self-test failure logs were observed in run_shell outputs."
    content = "\n\n".join(lines[-4:])
    return content[:max_chars]


def _generation_feedback(task_result, allow_verifier_feedback):
    if allow_verifier_feedback:
        return task_result.verify_message
    runtime_error = f"\nAgent runtime error: {task_result.error}" if task_result.error else ""
    signals = _extract_agent_observable_signals(task_result)
    return (
        "Hidden verifier result: FAIL.\n"
        "Do not assume access to hidden tests. Infer likely failure modes from the agent's own actions.\n"
        f"{runtime_error}\n\n"
        "Agent-observable signals:\n"
        f"{signals}"
    )


def _serialize_trajectory(trajectory):
    return [
        {
            "name": tc.name,
            "args": tc.args,
            "result": tc.result,
            "duration_ms": round(tc.duration_ms, 3),
        }
        for tc in trajectory
    ]


def _append_jsonl(log_path, payload):
    if not log_path:
        return
    record = {"ts": datetime.now().isoformat(), **payload}
    with Path(log_path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _load_current_tools():
    lib_schemas, lib_handlers = tool_library.load_tools()
    if not lib_schemas:
        return None, None, set()
    usage_examples = tool_library.load_tool_usage_examples()
    tool_examples = _build_tool_examples_section(usage_examples)
    prompt = AUGMENTED_SYSTEM_PROMPT.format(tool_examples=tool_examples)
    return (lib_schemas, lib_handlers), prompt, set(lib_handlers.keys())


def _make_harness(model, verbose, extra_tools=None, system_prompt=None, command_runner=None):
    return EvalHarness(
        client=OpenAIClient(model=model),
        verbose=verbose,
        model_name=model,
        extra_tools=extra_tools,
        system_prompt=system_prompt,
        command_runner=command_runner,
    )


def _run_self_improving(
    tasks, cheap_model, sota_model, verbose, command_runner,
    max_gen_attempts=3, allow_verifier_feedback=False, log_path=None,
    config_name=None, run_index=None
):
    """Run all tasks through the self-improving pipeline.

    Per task: run cheap model with accumulated library tools. On failure,
    SOTA generates a new tool and cheap model retries.  Tools persist
    across tasks within a single run so earlier tools can help later tasks.

    Returns (results: list[TaskResult], meta: dict).
    """
    tool_library.clear_all()

    results = []
    meta = {
        "tools_generated": {},
        "tool_sources": {},
        "cross_task_reuse": [],
        "total_gen_cost": 0.0,
    }

    for i, task in enumerate(tasks, 1):
        if not verbose:
            print(f"[{i}/{len(tasks)}] {task.id}...", end=" ", flush=True)

        extra_tools, sys_prompt, lib_names = _load_current_tools()
        harness = _make_harness(
            cheap_model, verbose, extra_tools, sys_prompt, command_runner=command_runner
        )
        result = harness.run_task(task)
        result.tools_used = [tc.name for tc in result.trajectory if tc.name in lib_names]
        sent_feedback = None if result.passed else _generation_feedback(result, allow_verifier_feedback)
        _append_jsonl(log_path, {
            "event": "task_initial_result",
            "config": config_name,
            "run_index": run_index,
            "task_index": i,
            "task_id": task.id,
            "model": cheap_model,
            "passed": result.passed,
            "verify_message": result.verify_message,
            "feedback_sent_to_generator": sent_feedback,
            "trajectory": _serialize_trajectory(result.trajectory),
            "tools_available": sorted(lib_names),
            "tools_used": result.tools_used,
            "cost": result.estimated_cost,
        })

        for tn in result.tools_used:
            src = meta["tool_sources"].get(tn)
            if src and src != task.id:
                meta["cross_task_reuse"].append({
                    "task_id": task.id,
                    "tool_name": tn,
                    "source_task": src,
                    "passed": result.passed,
                })

        if result.passed:
            results.append(result)
            continue

        retry_info = None
        task_gen_cost = 0.0
        success = False

        for attempt in range(1, max_gen_attempts + 1):
            if not verbose:
                print(f"\n  [gen {attempt}/{max_gen_attempts}]", end=" ", flush=True)
            else:
                print(f"\n  [tool_gen] attempt {attempt}/{max_gen_attempts} for {task.id}")

            try:
                current_feedback = _generation_feedback(result, allow_verifier_feedback)
                tool_code, gen_in, gen_out = generate_tool(
                    task_prompt=task.prompt,
                    trajectory=result.trajectory,
                    verify_message=current_feedback,
                    model=sota_model,
                    retry_info=retry_info,
                    existing_tools=tool_library.load_tool_summaries(),
                )
            except Exception as e:
                _append_jsonl(log_path, {
                    "event": "generation_error",
                    "config": config_name,
                    "run_index": run_index,
                    "task_id": task.id,
                    "attempt": attempt,
                    "error": str(e),
                })
                if verbose:
                    print(f"  [tool_gen] generation error: {e}")
                continue

            task_gen_cost += _generation_cost(sota_model, gen_in, gen_out)

            valid, name_or_err = _validate_tool_code(tool_code, verbose=verbose)
            _append_jsonl(log_path, {
                "event": "generation_attempt",
                "config": config_name,
                "run_index": run_index,
                "task_id": task.id,
                "attempt": attempt,
                "model": sota_model,
                "tokens_in": gen_in,
                "tokens_out": gen_out,
                "feedback_sent_to_generator": current_feedback,
                "tool_code": tool_code,
                "validation_passed": valid,
                "validation_result": name_or_err,
            })
            if not valid:
                if verbose:
                    print(f"  [tool_gen] invalid: {name_or_err}")
                continue

            tool_name = name_or_err
            tool_library.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
            tool_path = tool_library.GENERATED_DIR / f"{tool_name}.py"
            tool_path.write_text(tool_code)
            tool_library.register_tool(
                name=tool_name, file_path=tool_path, task_id=task.id,
                generator_model=sota_model, verified=True, verified_with=cheap_model,
            )

            et, sp, ln = _load_current_tools()
            harness2 = _make_harness(
                cheap_model, verbose, et, sp, command_runner=command_runner
            )

            if not verbose:
                print("retry...", end=" ", flush=True)
            else:
                print(f"  [tool_gen] re-running {task.id} with {tool_name}...")

            retry_result = harness2.run_task(task)
            retry_result.tools_used = [tc.name for tc in retry_result.trajectory if tc.name in ln]
            retry_result.extra_cost = task_gen_cost
            _append_jsonl(log_path, {
                "event": "generation_retry_result",
                "config": config_name,
                "run_index": run_index,
                "task_id": task.id,
                "attempt": attempt,
                "tool_name": tool_name,
                "passed": retry_result.passed,
                "verify_message": retry_result.verify_message,
                "feedback_sent_to_generator_next_attempt": _generation_feedback(
                    retry_result, allow_verifier_feedback
                ),
                "trajectory": _serialize_trajectory(retry_result.trajectory),
                "tools_used": retry_result.tools_used,
                "cost": retry_result.estimated_cost,
            })

            if retry_result.passed:
                meta["tools_generated"][task.id] = tool_name
                meta["tool_sources"][tool_name] = task.id
                results.append(retry_result)
                success = True
                break
            else:
                retry_info = {
                    "tool_name": tool_name,
                    "verify_message": _generation_feedback(retry_result, allow_verifier_feedback),
                }
                tool_library.remove_tool(tool_name)

        if not success:
            result.extra_cost = task_gen_cost
            results.append(result)

        meta["total_gen_cost"] += task_gen_cost

    passed = sum(1 for r in results if r.passed)
    total_cost = sum(r.estimated_cost for r in results)
    print(f"\n{'='*60}")
    print(f"Self-improving: {passed}/{len(results)} passed ({100*passed/len(results):.0f}%)")
    print(f"Cost: ${total_cost:.4f} (includes ${meta['total_gen_cost']:.4f} tool generation)")
    print(f"Tools in library: {len(meta['tools_generated'])}")
    if meta["cross_task_reuse"]:
        print(f"Cross-task reuse events: {len(meta['cross_task_reuse'])}")
        for ev in meta["cross_task_reuse"]:
            s = "PASS" if ev["passed"] else "FAIL"
            print(f"  {ev['tool_name']} ({ev['source_task']}) -> {ev['task_id']}: {s}")
    print(f"{'='*60}")

    return results, meta


def _print_pipeline_insights(all_meta):
    for config_name, meta_list in all_meta.items():
        print(f"\n{'='*60}")
        print(f"Pipeline Insights: {config_name}")
        print(f"{'='*60}")

        all_reuse = []

        for run_idx, meta in enumerate(meta_list):
            if len(meta_list) > 1:
                print(f"\n  Run {run_idx + 1}:")

            if meta["tools_generated"]:
                print(f"  Tools generated:")
                for task_id, tool_name in meta["tools_generated"].items():
                    print(f"    {task_id} -> {tool_name}")

            if meta["cross_task_reuse"]:
                print(f"  Cross-task tool reuse:")
                for ev in meta["cross_task_reuse"]:
                    s = "PASS" if ev["passed"] else "FAIL"
                    print(f"    {ev['tool_name']} (from {ev['source_task']}) -> {ev['task_id']}: {s}")
                all_reuse.extend(meta["cross_task_reuse"])

            print(f"  Generation cost: ${meta['total_gen_cost']:.4f}")

        if len(meta_list) > 1 and all_reuse:
            reuse_pass = sum(1 for ev in all_reuse if ev["passed"])
            print(f"\n  Cross-task reuse total: {len(all_reuse)} events, {reuse_pass} helped pass")


def _serialize_results(all_runs, all_meta, num_runs):
    configs = {}

    for config_name, run_list in all_runs.items():
        model, with_tools = _parse_config(config_name)
        runs_data = []
        total_passes = 0
        total_cost = 0.0

        for run_results in run_list:
            run_tasks = []
            for r in run_results:
                task_data = {
                    "task_id": r.task_id,
                    "passed": r.passed,
                    "cost": round(r.estimated_cost, 6),
                    "tokens_in": r.input_tokens,
                    "tokens_out": r.output_tokens,
                    "duration_ms": round(r.total_duration_ms, 1),
                    "tool_calls": r.num_tool_calls,
                }
                if r.tools_used:
                    task_data["tools_used"] = r.tools_used
                if r.extra_cost > 0:
                    task_data["generation_cost"] = round(r.extra_cost, 6)
                run_tasks.append(task_data)
                if r.passed:
                    total_passes += 1
                total_cost += r.estimated_cost
            runs_data.append(run_tasks)

        total_task_runs = len(run_list) * len(run_list[0]) if run_list else 0
        config_data = {
            "model": model,
            "with_tools": with_tools,
            "runs": runs_data,
            "summary": {
                "pass_rate": round(total_passes / total_task_runs, 4) if total_task_runs else 0,
                "avg_cost_per_run": round(total_cost / len(run_list), 6) if run_list else 0,
                "total_passes": total_passes,
                "total_task_runs": total_task_runs,
            },
        }

        if config_name in all_meta:
            config_data["pipeline"] = [
                {
                    "tools_generated": m["tools_generated"],
                    "cross_task_reuse": m["cross_task_reuse"],
                    "total_gen_cost": round(m["total_gen_cost"], 6),
                }
                for m in all_meta[config_name]
            ]

        configs[config_name] = config_data

    return {
        "timestamp": datetime.now().isoformat(),
        "num_runs": num_runs,
        "num_tasks": len(ALL_TASKS),
        "configs": configs,
    }


def main():
    parser = argparse.ArgumentParser(description="Run agent evals")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--task", metavar="ID", help="Run a single task by ID")
    group.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument("--model", default="gpt-4o", help="Model to use (default: gpt-4o)")
    parser.add_argument("--compare", nargs="+", metavar="CONFIG",
                        help="Compare configs (e.g. --compare gpt-4o-mini gpt-4o-mini+tools gpt-4o)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs per config in --compare mode (default: 1)")
    parser.add_argument("--output", metavar="PATH",
                        help="Save benchmark results to JSON file")
    parser.add_argument("--sota-model", default="gpt-4o",
                        help="SOTA model for tool generation in +tools configs (default: gpt-4o)")
    parser.add_argument("--allow-verifier-feedback", action="store_true",
                        help="Allow tool generation to see raw hidden verifier output (default: off)")
    parser.add_argument("--benchmark-log", metavar="PATH",
                        help="Write live benchmark/tool-gen events as JSONL")
    parser.add_argument("--with-tools", action="store_true",
                        help="Include generated tools from the tool library")
    parser.add_argument("--runner", choices=["host", "docker"], default="host",
                        help="Command runner backend for shell and verifier commands (default: host)")
    parser.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE,
                        help=f"Docker image to use with --runner docker (default: {DEFAULT_DOCKER_IMAGE})")
    parser.add_argument("--dockerfile", default=str(DEFAULT_DOCKERFILE),
                        help=f"Dockerfile used by --build-image (default: {DEFAULT_DOCKERFILE})")
    parser.add_argument("--build-image", action="store_true",
                        help="Build docker image before running benchmarks")
    parser.add_argument("--docker-smoke", action="store_true",
                        help="Run a quick docker smoke benchmark on representative tasks")
    parser.add_argument("--quiet", action="store_true", help="Suppress agent output")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    if args.build_image:
        build_result = build_docker_image(
            image=args.docker_image,
            dockerfile=Path(args.dockerfile),
            context=project_root,
        )
        if build_result.stdout:
            print(build_result.stdout)
        if build_result.returncode != 0:
            if build_result.stderr:
                print(build_result.stderr)
            elif build_result.error:
                print(build_result.error)
            sys.exit(1)

    command_runner = HostCommandRunner()
    if args.runner == "docker" or args.docker_smoke:
        command_runner = DockerCommandRunner(args.docker_image)
    set_command_runner(command_runner)

    if args.compare:
        all_runs = {}
        all_meta = {}

        for config_spec in args.compare:
            model, with_tools = _parse_config(config_spec)
            config_runs = []

            for run_idx in range(args.runs):
                print(f"\n{'#'*60}")
                print(f"  {config_spec}  (run {run_idx + 1}/{args.runs})")
                print(f"{'#'*60}")

                if with_tools:
                    results, meta = _run_self_improving(
                        ALL_TASKS, model, args.sota_model, verbose=not args.quiet,
                        command_runner=command_runner,
                        allow_verifier_feedback=args.allow_verifier_feedback,
                        log_path=args.benchmark_log,
                        config_name=config_spec,
                        run_index=run_idx + 1,
                    )
                    config_runs.append(results)
                    all_meta.setdefault(config_spec, []).append(meta)
                else:
                    harness = _make_harness(
                        model, verbose=not args.quiet, command_runner=command_runner
                    )
                    results = []
                    for task_i, task in enumerate(ALL_TASKS, 1):
                        if args.quiet:
                            print(f"[{task_i}/{len(ALL_TASKS)}] {task.id}...", end=" ", flush=True)
                        r = harness.run_task(task)
                        results.append(r)
                        _append_jsonl(args.benchmark_log, {
                            "event": "task_result",
                            "config": config_spec,
                            "run_index": run_idx + 1,
                            "task_index": task_i,
                            "task_id": r.task_id,
                            "model": model,
                            "passed": r.passed,
                            "verify_message": r.verify_message,
                            "trajectory": _serialize_trajectory(r.trajectory),
                            "tools_used": r.tools_used,
                            "cost": r.estimated_cost,
                        })
                    EvalHarness._print_summary(results)
                    config_runs.append(results)

            all_runs[config_spec] = config_runs

        if args.runs == 1:
            single_runs = {k: v[0] for k, v in all_runs.items()}
            EvalHarness.compare(single_runs)
        else:
            EvalHarness.compare_multi_run(all_runs)

        if all_meta:
            _print_pipeline_insights(all_meta)

        if args.output:
            data = _serialize_results(all_runs, all_meta, args.runs)
            Path(args.output).write_text(json.dumps(data, indent=2) + "\n")
            print(f"\nResults saved to {args.output}")

        return

    if args.docker_smoke:
        smoke_tasks = [TASK_MAP[task_id] for task_id in DOCKER_SMOKE_TASK_IDS]
        harness = _make_harness(
            args.model, verbose=not args.quiet, command_runner=command_runner
        )
        harness.run_all(smoke_tasks)
        return

    extra_tools = None
    augmented_prompt = None
    if args.with_tools:
        extra_tools, augmented_prompt, tool_names = _load_current_tools()
        if extra_tools:
            schemas = extra_tools[0]
            print(f"[tool library] Loaded {len(schemas)} tools: "
                  f"{', '.join(s['function']['name'] for s in schemas)}")
        else:
            print("[tool library] No verified tools found.")

    if not args.task and not args.all:
        parser.print_help()
        print(f"\nAvailable tasks: {', '.join(TASK_MAP.keys())}")
        sys.exit(0)

    harness = _make_harness(
        args.model, verbose=not args.quiet, extra_tools=extra_tools,
        system_prompt=augmented_prompt, command_runner=command_runner
    )

    if args.task:
        if args.task not in TASK_MAP:
            print(f"Unknown task: {args.task!r}. Available: {', '.join(TASK_MAP.keys())}")
            sys.exit(1)
        harness.run_task(TASK_MAP[args.task])
    else:
        harness.run_all(ALL_TASKS)


if __name__ == "__main__":
    main()
