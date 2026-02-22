from .read_file import SCHEMA as _read_file_schema, read_file
from .write_file import SCHEMA as _write_file_schema, write_file
from .run_shell import SCHEMA as _run_shell_schema, run_shell

TOOL_SCHEMAS = [_read_file_schema, _write_file_schema, _run_shell_schema]

_HANDLERS = {
    "read_file": read_file,
    "write_file": write_file,
    "run_shell": run_shell,
}


def dispatch(name: str, args: dict) -> str:
    if name not in _HANDLERS:
        return f"Unknown tool: {name}"
    return _HANDLERS[name](**args)
