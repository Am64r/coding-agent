import os
from ._workspace import resolve

SCHEMA = {
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
}


def write_file(path: str, content: str) -> str:
    try:
        resolved = resolve(path)
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        with open(resolved, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} characters to {resolved}"
    except Exception as e:
        return f"Error: {e}"
