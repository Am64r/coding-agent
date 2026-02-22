from pathlib import Path
from ..task import EvalTask
from ..verifier import TestsPasses

_HIDDEN_TESTS = """\
from fibonacci import fibonacci

def test_base_cases():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1

def test_sequence():
    assert fibonacci(2) == 1
    assert fibonacci(5) == 5
    assert fibonacci(10) == 55

def test_larger():
    assert fibonacci(20) == 6765
"""

def setup(workspace: Path) -> None:
    (workspace / "test_fibonacci.py").write_text(_HIDDEN_TESTS)

task = EvalTask(
    id="fibonacci",
    prompt=(
        "Create a file called fibonacci.py that defines a function fibonacci(n) "
        "which returns the nth Fibonacci number (0-indexed: fibonacci(0)=0, fibonacci(1)=1, fibonacci(2)=1, ...)."
    ),
    setup=setup,
    verify=TestsPasses("python3 -m pytest test_fibonacci.py -v").check,
    tags=["algorithm", "python", "hidden-tests"],
)
