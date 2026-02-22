from pathlib import Path

WORKSPACE = Path(__file__).parent.parent.parent / "agent" / "created_files"
WORKSPACE.mkdir(parents=True, exist_ok=True)


def resolve(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(WORKSPACE / p)
