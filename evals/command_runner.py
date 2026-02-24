import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    error: str = ""


class CommandRunner(ABC):
    @abstractmethod
    def run(self, command: str, cwd: Path, timeout: int) -> CommandResult:
        pass


class HostCommandRunner(CommandRunner):
    def run(self, command: str, cwd: Path, timeout: int) -> CommandResult:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(cwd),
            )
            return CommandResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(returncode=124, stdout="", stderr="", timed_out=True)
        except Exception as e:
            return CommandResult(returncode=1, stdout="", stderr="", error=str(e))


class DockerCommandRunner(CommandRunner):
    def __init__(self, image: str):
        self.image = image

    def run(self, command: str, cwd: Path, timeout: int) -> CommandResult:
        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{cwd}:/workspace",
            "-w",
            "/workspace",
            self.image,
            "sh",
            "-lc",
            command,
        ]
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return CommandResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(returncode=124, stdout="", stderr="", timed_out=True)
        except Exception as e:
            return CommandResult(returncode=1, stdout="", stderr="", error=str(e))


def build_docker_image(image: str, dockerfile: Path, context: Path) -> CommandResult:
    cmd = [
        "docker",
        "build",
        "-f",
        str(dockerfile),
        "-t",
        image,
        str(context),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return CommandResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except Exception as e:
        return CommandResult(returncode=1, stdout="", stderr="", error=str(e))
