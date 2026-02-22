import subprocess
import os
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent / "agent" / "created_files"
WORKSPACE.mkdir(parents=True, exist_ok=True)


def resolve(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(WORKSPACE / p)

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file."
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating it and any missing parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file."
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file."
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command and return its stdout and stderr. Use for running tests, installing packages, listing directories, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute."
                    }
                },
                "required": ["command"]
            }
        }
    }
]


def read_file(path: str) -> str:
    try:
        with open(resolve(path), "r") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"


def write_file(path: str, content: str) -> str:
    try:
        resolved = resolve(path)
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        with open(resolved, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} characters to {resolved}"
    except Exception as e:
        return f"Error: {e}"


def run_shell(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(WORKSPACE)
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


def dispatch(name: str, args: dict) -> str:
    handlers = {
        "read_file": read_file,
        "write_file": write_file,
        "run_shell": run_shell,
    }
    if name not in handlers:
        return f"Unknown tool: {name}"
    return handlers[name](**args)
