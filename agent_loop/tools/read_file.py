from ._workspace import resolve

SCHEMA = {
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
}


def read_file(path: str) -> str:
    try:
        with open(resolve(path), "r") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"
