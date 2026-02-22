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


def _build_toolbox(workspace: Path):
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
        import subprocess
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=30, cwd=str(workspace)
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR: {result.stderr}"
            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: command timed out after 30 seconds"
        except Exception as e:
            return f"Error: {e}"

    schemas = [read_file_schema, write_file_schema, run_shell_schema]
    handlers = {"read_file": read_file, "write_file": write_file, "run_shell": run_shell}

    def dispatch(name: str, args: dict) -> str:
        if name not in handlers:
            return f"Unknown tool: {name}"
        return handlers[name](**args)

    return schemas, dispatch


class EvalHarness:
    def __init__(self, client: LLMClient, verbose: bool = True, model_name: str = "",
                 extra_tools: tuple = None, system_prompt: str = None):
        self.client = client
        self.verbose = verbose
        self.model_name = model_name
        self.extra_tools = extra_tools
        self.system_prompt = system_prompt

    def run_task(self, task: EvalTask) -> TaskResult:
        workspace = Path(tempfile.mkdtemp(prefix=f"eval_{task.id}_"))
        trajectory: list[ToolCallRecord] = []

        try:
            task.setup(workspace)

            schemas, base_dispatch = _build_toolbox(workspace)

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
        header = f"{'task':<25}" + "".join(f"{'':>3}{m:<18}" for m in runs)
        print(f"\n{'='*len(header)}")
        print(header)
        print(f"{'-'*len(header)}")

        for tid in task_ids:
            row = f"{tid:<25}"
            for model, results in runs.items():
                r = next(r for r in results if r.task_id == tid)
                status = "PASS" if r.passed else "FAIL"
                row += f"   {status}  ${r.estimated_cost:.4f}      "
            print(row)

        print(f"{'-'*len(header)}")
        summary = f"{'TOTAL':<25}"
        for model, results in runs.items():
            p = sum(1 for r in results if r.passed)
            cost = sum(r.estimated_cost for r in results)
            summary += f"   {p}/{len(results)}   ${cost:.4f}      "
        print(summary)
        print(f"{'='*len(header)}")
