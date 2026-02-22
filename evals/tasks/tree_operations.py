from pathlib import Path
from ..task import EvalTask
from ..verifier import TestsPasses

_HIDDEN_TESTS = """\
from bst import BST

def test_insert_and_search():
    t = BST()
    t.insert(5)
    t.insert(3)
    t.insert(7)
    assert t.search(5) == True
    assert t.search(3) == True
    assert t.search(99) == False

def test_inorder():
    t = BST()
    for v in [5, 3, 7, 1, 4, 6, 8]:
        t.insert(v)
    assert t.inorder() == [1, 3, 4, 5, 6, 7, 8]

def test_min_max():
    t = BST()
    for v in [10, 5, 15, 3, 7]:
        t.insert(v)
    assert t.minimum() == 3
    assert t.maximum() == 15

def test_delete_leaf():
    t = BST()
    for v in [5, 3, 7]:
        t.insert(v)
    t.delete(3)
    assert t.inorder() == [5, 7]

def test_delete_one_child():
    t = BST()
    for v in [5, 3, 7, 6]:
        t.insert(v)
    t.delete(7)
    assert t.inorder() == [3, 5, 6]

def test_delete_two_children():
    t = BST()
    for v in [5, 3, 7, 6, 8]:
        t.insert(v)
    t.delete(7)
    assert t.search(7) == False
    assert sorted(t.inorder()) == t.inorder()
    assert set(t.inorder()) == {3, 5, 6, 8}

def test_delete_root():
    t = BST()
    for v in [5, 3, 7]:
        t.insert(v)
    t.delete(5)
    assert t.search(5) == False
    assert len(t.inorder()) == 2

def test_height():
    t = BST()
    assert t.height() == 0
    t.insert(1)
    assert t.height() == 1
    t.insert(2)
    assert t.height() == 2
    t.insert(3)
    assert t.height() == 3

def test_size():
    t = BST()
    assert t.size() == 0
    for v in [5, 3, 7, 1, 4]:
        t.insert(v)
    assert t.size() == 5
    t.delete(3)
    assert t.size() == 4

def test_duplicates_ignored():
    t = BST()
    t.insert(5)
    t.insert(5)
    t.insert(5)
    assert t.size() == 1
    assert t.inorder() == [5]

def test_empty_operations():
    t = BST()
    assert t.inorder() == []
    assert t.search(1) == False
    assert t.minimum() is None
    assert t.maximum() is None
    t.delete(1)
"""

def setup(workspace: Path) -> None:
    (workspace / "test_bst.py").write_text(_HIDDEN_TESTS)

task = EvalTask(
    id="tree_operations",
    prompt=(
        "Implement a binary search tree in bst.py.\n\n"
        "Class BST with methods:\n"
        "  - insert(value): insert a value (ignore duplicates)\n"
        "  - search(value) -> bool: return True if value exists\n"
        "  - delete(value): remove a value (handle all cases: leaf, one child, two children)\n"
        "  - inorder() -> list: return values in sorted order\n"
        "  - minimum() -> value or None: return the minimum value\n"
        "  - maximum() -> value or None: return the maximum value\n"
        "  - height() -> int: return the height of the tree (0 for empty)\n"
        "  - size() -> int: return number of nodes\n"
    ),
    setup=setup,
    verify=TestsPasses("python3 -m pytest test_bst.py -v").check,
    tags=["data-structure", "algorithm", "python", "hidden-tests"],
)
