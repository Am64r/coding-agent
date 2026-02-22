import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from .task import VerifyResult


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
        try:
            result = subprocess.run(
                self.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(workspace)
            )
            output = result.stdout.strip()
            passed = (output == self.expected) if self.exact else (self.expected in output)
            if passed:
                return VerifyResult(passed=True, message=f"Output matched: {output!r}")
            return VerifyResult(
                passed=False,
                message=f"Expected {self.expected!r} in output, got: {output!r}"
                + (f"\nSTDERR: {result.stderr.strip()}" if result.stderr.strip() else "")
            )
        except subprocess.TimeoutExpired:
            return VerifyResult(passed=False, message="Verification command timed out")


class TestsPasses(Verifier):
    def __init__(self, command: str):
        self.command = command

    def check(self, workspace: Path) -> VerifyResult:
        try:
            result = subprocess.run(
                self.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(workspace)
            )
            if result.returncode == 0:
                return VerifyResult(passed=True, message=f"Tests passed\n{result.stdout.strip()}")
            output = (result.stdout + result.stderr).strip()
            return VerifyResult(passed=False, message=f"Tests failed (exit {result.returncode})\n{output}")
        except subprocess.TimeoutExpired:
            return VerifyResult(passed=False, message="Test command timed out")


class AllOf(Verifier):
    def __init__(self, *verifiers: Verifier):
        self.verifiers = verifiers

    def check(self, workspace: Path) -> VerifyResult:
        for v in self.verifiers:
            result = v.check(workspace)
            if not result.passed:
                return result
        return VerifyResult(passed=True, message="All checks passed")
