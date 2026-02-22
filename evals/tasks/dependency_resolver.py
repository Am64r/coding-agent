from pathlib import Path
from ..task import EvalTask
from ..verifier import TestsPasses

_HIDDEN_TESTS = """\
from resolver import resolve_dependencies, CircularDependencyError

def test_simple_chain():
    deps = {"a": ["b"], "b": ["c"], "c": []}
    result = resolve_dependencies(deps)
    assert result.index("c") < result.index("b") < result.index("a")

def test_no_deps():
    deps = {"a": [], "b": [], "c": []}
    result = resolve_dependencies(deps)
    assert set(result) == {"a", "b", "c"}

def test_diamond():
    deps = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}
    result = resolve_dependencies(deps)
    assert result.index("d") < result.index("b")
    assert result.index("d") < result.index("c")
    assert result.index("b") < result.index("a")
    assert result.index("c") < result.index("a")

def test_circular_raises():
    deps = {"a": ["b"], "b": ["c"], "c": ["a"]}
    try:
        resolve_dependencies(deps)
        assert False, "Should raise CircularDependencyError"
    except CircularDependencyError:
        pass

def test_self_cycle():
    deps = {"a": ["a"]}
    try:
        resolve_dependencies(deps)
        assert False, "Should raise CircularDependencyError"
    except CircularDependencyError:
        pass

def test_complex_graph():
    deps = {
        "app": ["api", "ui"],
        "api": ["auth", "db"],
        "ui": ["auth"],
        "auth": ["crypto"],
        "db": [],
        "crypto": [],
    }
    result = resolve_dependencies(deps)
    assert result.index("crypto") < result.index("auth")
    assert result.index("auth") < result.index("api")
    assert result.index("db") < result.index("api")
    assert result.index("api") < result.index("app")
    assert result.index("ui") < result.index("app")
    assert len(result) == 6

def test_single_node():
    deps = {"x": []}
    assert resolve_dependencies(deps) == ["x"]

def test_all_returned():
    deps = {"a": ["b"], "b": ["c"], "c": [], "d": ["c"]}
    result = resolve_dependencies(deps)
    assert set(result) == {"a", "b", "c", "d"}
"""

def setup(workspace: Path) -> None:
    (workspace / "test_resolver.py").write_text(_HIDDEN_TESTS)

task = EvalTask(
    id="dependency_resolver",
    prompt=(
        "Create resolver.py with a topological sort-based dependency resolver.\n\n"
        "Custom exception: CircularDependencyError(Exception)\n\n"
        "Function resolve_dependencies(deps: dict[str, list[str]]) -> list[str]:\n"
        "  - Input: dict mapping each package name to its list of dependencies\n"
        "  - Output: list of package names in install order (dependencies before dependents)\n"
        "  - Raise CircularDependencyError if there's a cycle\n"
        "  - All packages in the input dict must appear in the output\n"
    ),
    setup=setup,
    verify=TestsPasses("python3 -m pytest test_resolver.py -v").check,
    tags=["algorithm", "graph", "python", "hidden-tests"],
)
