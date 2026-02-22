from pathlib import Path
from ..task import EvalTask
from ..verifier import TestsPasses

_HIDDEN_TESTS = """\
import time
from cache import LRUCache

def test_get_set():
    c = LRUCache(3)
    c.set("a", 1)
    c.set("b", 2)
    assert c.get("a") == 1
    assert c.get("b") == 2

def test_missing_key():
    c = LRUCache(3)
    assert c.get("x") is None
    assert c.get("x", "default") == "default"

def test_eviction():
    c = LRUCache(3)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    c.set("d", 4)
    assert c.get("a") is None
    assert c.get("d") == 4

def test_access_refreshes():
    c = LRUCache(3)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    c.get("a")
    c.set("d", 4)
    assert c.get("a") == 1
    assert c.get("b") is None

def test_update_existing():
    c = LRUCache(2)
    c.set("a", 1)
    c.set("b", 2)
    c.set("a", 10)
    assert c.get("a") == 10
    c.set("c", 3)
    assert c.get("b") is None
    assert c.get("a") == 10

def test_size():
    c = LRUCache(3)
    assert len(c) == 0
    c.set("a", 1)
    c.set("b", 2)
    assert len(c) == 2
    c.set("c", 3)
    c.set("d", 4)
    assert len(c) == 3

def test_ttl():
    c = LRUCache(10, default_ttl=0.1)
    c.set("a", 1)
    assert c.get("a") == 1
    time.sleep(0.15)
    assert c.get("a") is None

def test_per_key_ttl():
    c = LRUCache(10)
    c.set("short", 1, ttl=0.1)
    c.set("long", 2, ttl=10.0)
    time.sleep(0.15)
    assert c.get("short") is None
    assert c.get("long") == 2

def test_no_ttl():
    c = LRUCache(10)
    c.set("forever", 42)
    assert c.get("forever") == 42

def test_delete():
    c = LRUCache(5)
    c.set("a", 1)
    c.delete("a")
    assert c.get("a") is None
    assert len(c) == 0

def test_clear():
    c = LRUCache(5)
    c.set("a", 1)
    c.set("b", 2)
    c.clear()
    assert len(c) == 0
    assert c.get("a") is None
"""

def setup(workspace: Path) -> None:
    (workspace / "test_cache.py").write_text(_HIDDEN_TESTS)

task = EvalTask(
    id="implement_cache",
    prompt=(
        "Implement an LRU cache in cache.py with the following API:\n\n"
        "LRUCache(capacity, default_ttl=None)\n"
        "  - capacity: max number of items\n"
        "  - default_ttl: optional default time-to-live in seconds (None means no expiry)\n\n"
        "Methods:\n"
        "  - set(key, value, ttl=None): store a value. If ttl is given, use it; otherwise use default_ttl. "
        "If capacity is exceeded, evict the least recently used item. Setting an existing key updates it and refreshes its position.\n"
        "  - get(key, default=None): return the value or default if missing/expired. Accessing refreshes the item's position.\n"
        "  - delete(key): remove a key if it exists.\n"
        "  - clear(): remove all items.\n"
        "  - __len__(): return current number of (non-expired) items.\n\n"
        "Use only the standard library (no third-party packages). Must be O(1) for get/set."
    ),
    setup=setup,
    verify=TestsPasses("python3 -m pytest test_cache.py -v").check,
    tags=["data-structure", "performance", "python", "hidden-tests"],
)
