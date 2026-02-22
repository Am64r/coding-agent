from pathlib import Path
from ..task import EvalTask
from ..verifier import TestsPasses

_HIDDEN_TESTS = """\
from cli import parse_args, CLIError

def test_simple_command():
    result = parse_args(["init"])
    assert result["command"] == "init"

def test_positional_arg():
    result = parse_args(["clone", "https://github.com/foo/bar"])
    assert result["command"] == "clone"
    assert result["args"] == ["https://github.com/foo/bar"]

def test_flag():
    result = parse_args(["build", "--verbose"])
    assert result["flags"]["verbose"] == True

def test_key_value():
    result = parse_args(["deploy", "--env", "production"])
    assert result["flags"]["env"] == "production"

def test_short_flag():
    result = parse_args(["test", "-v"])
    assert result["flags"]["v"] == True

def test_short_key_value():
    result = parse_args(["run", "-n", "5"])
    assert result["flags"]["n"] == "5"

def test_mixed():
    result = parse_args(["deploy", "myapp", "--env", "staging", "-v", "--retries", "3"])
    assert result["command"] == "deploy"
    assert result["args"] == ["myapp"]
    assert result["flags"]["env"] == "staging"
    assert result["flags"]["v"] == True
    assert result["flags"]["retries"] == "3"

def test_double_dash_stops_flags():
    result = parse_args(["run", "--", "--not-a-flag", "-x"])
    assert result["args"] == ["--not-a-flag", "-x"]
    assert "not-a-flag" not in result.get("flags", {})

def test_no_args():
    result = parse_args([])
    assert result["command"] is None
    assert result["args"] == []

def test_equals_syntax():
    result = parse_args(["config", "--name=value"])
    assert result["flags"]["name"] == "value"

def test_error_on_dangling_value_flag():
    try:
        parse_args(["build", "--output"])
        result = parse_args(["build", "--output"])
        assert result["flags"]["output"] == True
    except CLIError:
        pass

def test_multiple_positional():
    result = parse_args(["copy", "src.txt", "dst.txt"])
    assert result["args"] == ["src.txt", "dst.txt"]
"""

def setup(workspace: Path) -> None:
    (workspace / "test_cli.py").write_text(_HIDDEN_TESTS)

task = EvalTask(
    id="cli_parser",
    prompt=(
        "Create cli.py with a command-line argument parser (don't use argparse — build it from scratch).\n\n"
        "Custom exception: CLIError(Exception)\n\n"
        "Function parse_args(argv: list[str]) -> dict that returns:\n"
        "  - 'command': first non-flag argument (or None if empty)\n"
        "  - 'args': list of positional arguments (after command)\n"
        "  - 'flags': dict of parsed flags\n\n"
        "Flag parsing rules:\n"
        "  - --flag (boolean, set to True)\n"
        "  - --key value (key-value pair)\n"
        "  - --key=value (key-value with equals)\n"
        "  - -f (short boolean flag)\n"
        "  - -k value (short key-value)\n"
        "  - -- (double dash stops flag parsing; everything after is positional)\n"
        "  - A --flag at the end of argv with no value after it should be treated as boolean True\n"
    ),
    setup=setup,
    verify=TestsPasses("python3 -m pytest test_cli.py -v").check,
    tags=["parsing", "design", "python", "hidden-tests"],
)
