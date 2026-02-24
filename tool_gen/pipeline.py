import sys
import time
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agent_loop"))

from agent import OpenAIClient, Agent
from evals.harness import EvalHarness, _build_toolbox
from evals.task import EvalTask, TaskResult, ToolCallRecord

from .generator import generate_tool
import tool_library


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


def _build_tool_examples_section(usage_examples):
    if not usage_examples:
        return ""
    lines = ["\n## Specialized Tool Usage Examples\n"]
    for name, example in usage_examples.items():
        lines.append(f"### {name}")
        lines.append(example.strip())
        lines.append("")
    return "\n".join(lines)


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


def _validate_tool_code(code, verbose=True):
    namespace = {}
    try:
        exec(code, namespace)
    except Exception as e:
        return False, f"Code execution error: {e}"

    if "SCHEMA" not in namespace:
        return False, "Missing SCHEMA definition"

    schema = namespace["SCHEMA"]
    func_name = schema.get("function", {}).get("name")
    if not func_name:
        return False, "SCHEMA missing function.name"

    if func_name not in namespace:
        return False, f"Function '{func_name}' not defined"

    if not callable(namespace[func_name]):
        return False, f"'{func_name}' is not callable"

    if "USAGE_EXAMPLE" not in namespace and verbose:
        print(f"  Warning: tool '{func_name}' missing USAGE_EXAMPLE (recommended for agent discoverability)")

    return True, func_name


def _run_with_library_tools(task, model, verbose):
    client = OpenAIClient(model=model)
    lib_schemas, lib_handlers = tool_library.load_tools()
    usage_examples = tool_library.load_tool_usage_examples()

    workspace = Path(tempfile.mkdtemp(prefix=f"eval_{task.id}_"))
    trajectory = []

    try:
        task.setup(workspace)
        base_schemas, base_dispatch = _build_toolbox(workspace)
        all_schemas = base_schemas + lib_schemas

        def merged_dispatch(name, args):
            if name in lib_handlers:
                try:
                    return str(lib_handlers[name](**args))
                except Exception as e:
                    return f"Error: {e}"
            return base_dispatch(name, args)

        def recording_dispatch(name, args):
            t0 = time.monotonic()
            result = merged_dispatch(name, args)
            trajectory.append(ToolCallRecord(
                name=name, args=args, result=result,
                duration_ms=(time.monotonic() - t0) * 1000,
            ))
            return result

        if lib_schemas:
            tool_examples = _build_tool_examples_section(usage_examples)
            system_prompt = AUGMENTED_SYSTEM_PROMPT.format(tool_examples=tool_examples)
        else:
            system_prompt = None

        agent = Agent(
            client=client,
            tools=all_schemas,
            dispatch_fn=recording_dispatch,
            verbose=verbose,
            system_prompt=system_prompt,
        )

        t0 = time.monotonic()
        try:
            agent_result = agent.run(task.prompt)
            final_response = agent_result.content
            input_tokens = agent_result.input_tokens
            output_tokens = agent_result.output_tokens
            error = None
        except Exception as e:
            final_response = ""
            input_tokens = output_tokens = 0
            error = str(e)

        total_ms = (time.monotonic() - t0) * 1000
        verify_result = task.verify(workspace)

        result = TaskResult(
            task_id=task.id,
            passed=verify_result.passed,
            verify_message=verify_result.message,
            trajectory=trajectory,
            final_response=final_response,
            total_duration_ms=total_ms,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=error,
        )

        status = "PASS" if verify_result.passed else "FAIL"
        if verbose:
            print(f"\n{status} {task.id} -- {len(trajectory)} tool calls, {total_ms/1000:.1f}s")
            print(f"     {verify_result.message[:200]}")
            print(f"     tokens: {input_tokens:,} in / {output_tokens:,} out -- ${result.estimated_cost:.4f}")

        return result
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def run_pipeline(
    task: EvalTask,
    cheap_model: str = "gpt-4o-mini",
    sota_model: str = "gpt-4o",
    max_attempts: int = 3,
    verbose: bool = True,
    allow_verifier_feedback: bool = False,
):
    if verbose:
        print(f"\n{'='*60}")
        print(f"Pipeline: {task.id}")
        print(f"Cheap: {cheap_model} | SOTA: {sota_model}")
        print(f"{'='*60}")

    cheap_client = OpenAIClient(model=cheap_model)
    harness = EvalHarness(client=cheap_client, verbose=verbose, model_name=cheap_model)
    initial_result = harness.run_task(task)

    if initial_result.passed:
        if verbose:
            print(f"\n>>> {task.id} already passes with {cheap_model} -- no tool needed")
        return {
            "task_id": task.id,
            "status": "already_passing",
            "initial_result": initial_result,
            "tool_generated": False,
        }

    if verbose:
        print(f"\n>>> {task.id} FAILED with {cheap_model} -- generating tool...")

    gen_costs = {"input_tokens": 0, "output_tokens": 0}
    retry_info = None

    for attempt in range(1, max_attempts + 1):
        if verbose:
            print(f"\n--- Generation attempt {attempt}/{max_attempts} ---")

        existing_tools = tool_library.load_tool_summaries()

        tool_code, gen_in, gen_out = generate_tool(
            task_prompt=task.prompt,
            trajectory=initial_result.trajectory,
            verify_message=_generation_feedback(initial_result, allow_verifier_feedback),
            model=sota_model,
            retry_info=retry_info,
            existing_tools=existing_tools,
        )
        gen_costs["input_tokens"] += gen_in
        gen_costs["output_tokens"] += gen_out

        if verbose:
            print(f"Generated tool ({gen_in + gen_out:,} tokens)")

        valid, result = _validate_tool_code(tool_code, verbose=verbose)
        if not valid:
            if verbose:
                print(f"  Invalid tool: {result}")
            continue

        tool_name = result
        if verbose:
            print(f"  Valid tool: {tool_name}")

        tool_library.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        tool_path = tool_library.GENERATED_DIR / f"{tool_name}.py"
        tool_path.write_text(tool_code)

        tool_library.register_tool(
            name=tool_name,
            file_path=tool_path,
            task_id=task.id,
            generator_model=sota_model,
            verified=True,
            verified_with=cheap_model,
        )

        if verbose:
            print(f"\n  Re-running {task.id} with {cheap_model} + {tool_name}...")

        retry_result = _run_with_library_tools(task, cheap_model, verbose)

        if retry_result.passed:
            if verbose:
                print(f"\n>>> SUCCESS: {task.id} PASSED with {cheap_model} + {tool_name}")
                print(f"    Tool saved: {tool_path}")
                print(f"    Generation cost: {gen_costs['input_tokens'] + gen_costs['output_tokens']:,} tokens")
            return {
                "task_id": task.id,
                "status": "tool_generated",
                "initial_result": initial_result,
                "retry_result": retry_result,
                "tool_name": tool_name,
                "tool_code": tool_code,
                "tool_generated": True,
                "generation_cost": gen_costs,
                "attempts": attempt,
            }
        else:
            if verbose:
                print(f"  Still failed with tool. Removing and retrying...")
            retry_info = {
                "tool_name": tool_name,
                "verify_message": _generation_feedback(retry_result, allow_verifier_feedback),
            }
            tool_library.remove_tool(tool_name)

    if verbose:
        print(f"\n>>> FAILED: Could not generate a working tool for {task.id} after {max_attempts} attempts")
        print(f"    Total generation cost: {gen_costs['input_tokens'] + gen_costs['output_tokens']:,} tokens")

    return {
        "task_id": task.id,
        "status": "generation_failed",
        "initial_result": initial_result,
        "tool_generated": False,
        "generation_cost": gen_costs,
        "attempts": max_attempts,
    }
