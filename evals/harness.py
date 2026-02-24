import sys
import time
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agent_loop"))

from agent import Agent, AgentResult, LLMClient
from tools.read_file import SCHEMA as read_file_schema, read_file as _read_file
from tools.write_file import SCHEMA as write_file_schema, write_file as _write_file
from tools.run_shell import SCHEMA as run_shell_schema, run_shell as _run_shell

from .task import EvalTask, TaskResult, ToolCallRecord
from .command_runner import CommandRunner, HostCommandRunner


def _build_toolbox(workspace: Path, command_runner: CommandRunner):
    def resolve(path: str) -> str:
        p = Path(path)
        return str(p if p.is_absolute() else workspace / p)

    def read_file(path: str) -> str:
        try:
            with open(resolve(path), "r") as f:
                return f.read()
        except Exception as e:
            return f"Error: {e}"

    def write_file(path: str, content: str) -> str:
        import os
        try:
            resolved = resolve(path)
            os.makedirs(os.path.dirname(resolved), exist_ok=True)
            with open(resolved, "w") as f:
                f.write(content)
            return f"Wrote {len(content)} characters to {resolved}"
        except Exception as e:
            return f"Error: {e}"

    def run_shell(command: str) -> str:
        result = command_runner.run(command, workspace, timeout=30)
        if result.timed_out:
            return "Error: command timed out after 30 seconds"
        if result.error:
            return f"Error: {result.error}"
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"
        return output.strip() or "(no output)"

    schemas = [read_file_schema, write_file_schema, run_shell_schema]
    handlers = {"read_file": read_file, "write_file": write_file, "run_shell": run_shell}

    def dispatch(name: str, args: dict) -> str:
        if name not in handlers:
            return f"Unknown tool: {name}"
        return handlers[name](**args)

    return schemas, dispatch


class EvalHarness:
    def __init__(self, client: LLMClient, verbose: bool = True, model_name: str = "",
                 extra_tools: tuple = None, system_prompt: str = None,
                 command_runner: CommandRunner | None = None):
        self.client = client
        self.verbose = verbose
        self.model_name = model_name
        self.extra_tools = extra_tools
        self.system_prompt = system_prompt
        self.command_runner = command_runner or HostCommandRunner()

    def run_task(self, task: EvalTask) -> TaskResult:
        workspace = Path(tempfile.mkdtemp(prefix=f"eval_{task.id}_"))
        trajectory: list[ToolCallRecord] = []

        try:
            task.setup(workspace)

            schemas, base_dispatch = _build_toolbox(workspace, self.command_runner)

            if self.extra_tools:
                extra_schemas, extra_handlers = self.extra_tools
                schemas = schemas + extra_schemas

                def merged_dispatch(name: str, args: dict) -> str:
                    if name in extra_handlers:
                        try:
                            return str(extra_handlers[name](**args))
                        except Exception as e:
                            return f"Error: {e}"
                    return base_dispatch(name, args)
                dispatch_fn = merged_dispatch
            else:
                dispatch_fn = base_dispatch

            def recording_dispatch(name: str, args: dict) -> str:
                t0 = time.monotonic()
                result = dispatch_fn(name, args)
                trajectory.append(ToolCallRecord(
                    name=name,
                    args=args,
                    result=result,
                    duration_ms=(time.monotonic() - t0) * 1000,
                ))
                return result

            agent = Agent(
                client=self.client,
                tools=schemas,
                dispatch_fn=recording_dispatch,
                verbose=self.verbose,
                system_prompt=self.system_prompt,
            )

            if self.verbose:
                print(f"\n{'='*60}", flush=True)
                print(f"Task: {task.id}", flush=True)
                print(f"{'='*60}", flush=True)

            t0 = time.monotonic()
            input_tokens = output_tokens = 0
            try:
                agent_result = agent.run(task.prompt)
                final_response = agent_result.content
                input_tokens = agent_result.input_tokens
                output_tokens = agent_result.output_tokens
                error = None
            except Exception as e:
                final_response = ""
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
                model=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error=error,
            )

            status = "PASS" if verify_result.passed else "FAIL"
            if self.verbose:
                print(f"\n{status} {task.id} — {len(trajectory)} tool calls, {total_ms/1000:.1f}s", flush=True)
                print(f"     {verify_result.message}", flush=True)
                print(f"     tokens: {input_tokens:,} in / {output_tokens:,} out — ${result.estimated_cost:.4f}", flush=True)
            else:
                print(f"  {status}  {task.id:<25} {total_ms/1000:.1f}s  ${result.estimated_cost:.4f}", flush=True)

            return result

        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def run_all(self, tasks: list[EvalTask]) -> list[TaskResult]:
        results = []
        for i, task in enumerate(tasks, 1):
            if not self.verbose:
                print(f"[{i}/{len(tasks)}] {task.id}...", end=" ", flush=True)
            results.append(self.run_task(task))
        self._print_summary(results)
        return results

    @staticmethod
    def _print_summary(results: list[TaskResult]):
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        total_cost = sum(r.estimated_cost for r in results)
        total_in = sum(r.input_tokens for r in results)
        total_out = sum(r.output_tokens for r in results)

        print(f"\n{'='*60}")
        print(f"Results: {passed}/{total} passed ({100*passed/total:.0f}%)")
        print(f"Tokens:  {total_in:,} in / {total_out:,} out — ${total_cost:.4f}")
        print(f"{'='*60}")
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  {status}  {r.task_id:<25} ${r.estimated_cost:.4f}")

    @staticmethod
    def compare(runs: dict[str, list[TaskResult]]):
        task_ids = [r.task_id for r in next(iter(runs.values()))]
        col_w = max(len(c) for c in runs) + 4
        col_w = max(col_w, 18)
        header = f"{'task':<25}" + "".join(f"{c:>{col_w}}" for c in runs)
        sep = "-" * len(header)
        print(f"\n{'=' * len(header)}")
        print(header)
        print(sep)

        for tid in task_ids:
            row = f"{tid:<25}"
            for config, results in runs.items():
                r = next(r for r in results if r.task_id == tid)
                status = "PASS" if r.passed else "FAIL"
                cell = f"{status} ${r.estimated_cost:.4f}"
                row += f"{cell:>{col_w}}"
            print(row)

        print(sep)
        row = f"{'TOTAL':<25}"
        for config, results in runs.items():
            p = sum(1 for r in results if r.passed)
            cost = sum(r.estimated_cost for r in results)
            cell = f"{p}/{len(results)} ${cost:.4f}"
            row += f"{cell:>{col_w}}"
        print(row)
        print(f"{'=' * len(header)}")

    @staticmethod
    def compare_multi_run(all_runs: dict[str, list[list[TaskResult]]]):
        first_config = next(iter(all_runs.values()))
        task_ids = [r.task_id for r in first_config[0]]
        num_runs = len(first_config)

        col_w = max(len(c) for c in all_runs) + 4
        col_w = max(col_w, 18)
        header = f"{'task':<25}" + "".join(f"{c:>{col_w}}" for c in all_runs)
        sep = "-" * len(header)

        print(f"\n{'=' * len(header)}")
        print(f"Benchmark: {num_runs} runs per config, {len(task_ids)} tasks")
        print(f"{'=' * len(header)}")
        print(header)
        print(sep)

        for tid in task_ids:
            row = f"{tid:<25}"
            for config, run_list in all_runs.items():
                passes = 0
                total_cost = 0.0
                for run_results in run_list:
                    r = next(r for r in run_results if r.task_id == tid)
                    if r.passed:
                        passes += 1
                    total_cost += r.estimated_cost
                avg_cost = total_cost / len(run_list)
                cell = f"{passes}/{len(run_list)} ${avg_cost:.4f}"
                row += f"{cell:>{col_w}}"
            print(row)

        print(sep)

        row = f"{'PASS RATE':<25}"
        for config, run_list in all_runs.items():
            total_passes = sum(
                1 for run_results in run_list for r in run_results if r.passed
            )
            total_tasks = sum(len(run_results) for run_results in run_list)
            rate = total_passes / total_tasks if total_tasks else 0
            cell = f"{100 * rate:.1f}%"
            row += f"{cell:>{col_w}}"
        print(row)

        row = f"{'AVG COST/RUN':<25}"
        for config, run_list in all_runs.items():
            total_cost = sum(
                r.estimated_cost for run_results in run_list for r in run_results
            )
            avg = total_cost / len(run_list) if run_list else 0
            cell = f"${avg:.4f}"
            row += f"{cell:>{col_w}}"
        print(row)

        print(f"{'=' * len(header)}")
