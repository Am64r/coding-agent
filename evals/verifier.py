from abc import ABC, abstractmethod
from pathlib import Path

from .task import VerifyResult
from .command_runner import CommandRunner, HostCommandRunner


_COMMAND_RUNNER: CommandRunner = HostCommandRunner()


def set_command_runner(command_runner: CommandRunner):
    global _COMMAND_RUNNER
    _COMMAND_RUNNER = command_runner


class Verifier(ABC):
    @abstractmethod
    def check(self, workspace: Path) -> VerifyResult:
        pass


class FileExists(Verifier):
    def __init__(self, path: str):
        self.path = path

    def check(self, workspace: Path) -> VerifyResult:
        if (workspace / self.path).exists():
            return VerifyResult(passed=True, message=f"{self.path} exists")
        return VerifyResult(passed=False, message=f"{self.path} not found")


class FileContains(Verifier):
    def __init__(self, path: str, pattern: str):
        self.path = path
        self.pattern = pattern

    def check(self, workspace: Path) -> VerifyResult:
        full_path = workspace / self.path
        if not full_path.exists():
            return VerifyResult(passed=False, message=f"{self.path} not found")
        content = full_path.read_text()
        if self.pattern in content:
            return VerifyResult(passed=True, message=f"{self.path} contains expected pattern")
        return VerifyResult(passed=False, message=f"{self.path} missing pattern: {self.pattern!r}")


class ShellOutput(Verifier):
    def __init__(self, command: str, expected: str, exact: bool = False):
        self.command = command
        self.expected = expected
        self.exact = exact

    def check(self, workspace: Path) -> VerifyResult:
        result = _COMMAND_RUNNER.run(self.command, workspace, timeout=30)
        if result.timed_out:
            return VerifyResult(passed=False, message="Verification command timed out")
        if result.error:
            return VerifyResult(passed=False, message=f"Verification command error: {result.error}")
        output = result.stdout.strip()
        passed = (output == self.expected) if self.exact else (self.expected in output)
        if passed:
            return VerifyResult(passed=True, message=f"Output matched: {output!r}")
        return VerifyResult(
            passed=False,
            message=f"Expected {self.expected!r} in output, got: {output!r}"
            + (f"\nSTDERR: {result.stderr.strip()}" if result.stderr.strip() else "")
        )


class TestsPasses(Verifier):
    def __init__(self, command: str):
        self.command = command

    def check(self, workspace: Path) -> VerifyResult:
        result = _COMMAND_RUNNER.run(self.command, workspace, timeout=60)
        if result.timed_out:
            return VerifyResult(passed=False, message="Test command timed out")
        if result.error:
            return VerifyResult(passed=False, message=f"Test command error: {result.error}")
        if result.returncode == 0:
            return VerifyResult(passed=True, message=f"Tests passed\n{result.stdout.strip()}")
        output = (result.stdout + result.stderr).strip()
        return VerifyResult(passed=False, message=f"Tests failed (exit {result.returncode})\n{output}")


class AllOf(Verifier):
    def __init__(self, *verifiers: Verifier):
        self.verifiers = verifiers

    def check(self, workspace: Path) -> VerifyResult:
        for v in self.verifiers:
            result = v.check(workspace)
            if not result.passed:
                return result
        return VerifyResult(passed=True, message="All checks passed")
