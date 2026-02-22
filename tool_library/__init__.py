import json
import importlib.util
from pathlib import Path
from datetime import datetime

LIBRARY_DIR = Path(__file__).parent
GENERATED_DIR = LIBRARY_DIR / "generated"
REGISTRY_PATH = LIBRARY_DIR / "registry.json"


def _load_registry():
    if not REGISTRY_PATH.exists():
        return {"tools": []}
    return json.loads(REGISTRY_PATH.read_text())


def _save_registry(registry):
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2) + "\n")


def _load_tool_module(file_path):
    spec = importlib.util.spec_from_file_location("tool_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tools():
    """Load all verified tools. Returns (schemas_list, handlers_dict)."""
    registry = _load_registry()
    schemas = []
    handlers = {}

    for entry in registry["tools"]:
        if not entry.get("verified"):
            continue
        tool_path = LIBRARY_DIR / entry["file"]
        if not tool_path.exists():
            continue
        try:
            module = _load_tool_module(tool_path)
            name = module.SCHEMA["function"]["name"]
            schemas.append(module.SCHEMA)
            handlers[name] = getattr(module, name)
        except Exception:
            continue

    return schemas, handlers


def register_tool(name, file_path, task_id, generator_model, verified=False, verified_with=""):
    registry = _load_registry()
    registry["tools"] = [t for t in registry["tools"] if t["name"] != name]
    registry["tools"].append({
        "name": name,
        "file": str(Path(file_path).relative_to(LIBRARY_DIR)),
        "generated_from_task": task_id,
        "generated_by_model": generator_model,
        "verified": verified,
        "verified_with_model": verified_with,
        "created_at": datetime.now().isoformat(),
    })
    _save_registry(registry)


def mark_verified(name, verified_with_model):
    registry = _load_registry()
    for entry in registry["tools"]:
        if entry["name"] == name:
            entry["verified"] = True
            entry["verified_with_model"] = verified_with_model
            break
    _save_registry(registry)


def remove_tool(name):
    registry = _load_registry()
    registry["tools"] = [t for t in registry["tools"] if t["name"] != name]
    _save_registry(registry)
    tool_path = GENERATED_DIR / f"{name}.py"
    if tool_path.exists():
        tool_path.unlink()


def list_tools():
    return _load_registry()["tools"]
