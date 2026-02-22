import sys
import time
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agent_loop"))

from agent import Agent, LLMClient
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
    def __init__(self, client: LLMClient, verbose: bool = True):
        self.client = client
        self.verbose = verbose

    def run_task(self, task: EvalTask) -> TaskResult:
        workspace = Path(tempfile.mkdtemp(prefix=f"eval_{task.id}_"))
        trajectory: list[ToolCallRecord] = []

        try:
            task.setup(workspace)

            schemas, base_dispatch = _build_toolbox(workspace)

            def recording_dispatch(name: str, args: dict) -> str:
                t0 = time.monotonic()
                result = base_dispatch(name, args)
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
            )

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Task: {task.id}")
                print(f"{'='*60}")

            t0 = time.monotonic()
            try:
                final_response = agent.run(task.prompt)
                error = None
            except Exception as e:
                final_response = ""
                error = str(e)

            total_ms = (time.monotonic() - t0) * 1000

            verify_result = task.verify(workspace)

            if self.verbose:
                status = "PASS" if verify_result.passed else "FAIL"
                print(f"\n{status} {task.id} — {len(trajectory)} tool calls, {total_ms/1000:.1f}s")
                print(f"     {verify_result.message}")

            return TaskResult(
                task_id=task.id,
                passed=verify_result.passed,
                verify_message=verify_result.message,
                trajectory=trajectory,
                final_response=final_response,
                total_duration_ms=total_ms,
                error=error,
            )

        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def run_all(self, tasks: list[EvalTask]) -> list[TaskResult]:
        results = []
        for task in tasks:
            results.append(self.run_task(task))

        passed = sum(1 for r in results if r.passed)
        total = len(results)

        print(f"\n{'='*60}")
        print(f"Results: {passed}/{total} passed ({100*passed/total:.0f}%)")
        print(f"{'='*60}")
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  {status}  {r.task_id}")

        return results
