from pathlib import Path
from ..task import EvalTask
from ..verifier import TestsPasses

_APP_CODE = """\
import json

class Config:
    def __init__(self, path):
        with open(path) as f:
            self._data = json.load(f)

    def get(self, key, default=None):
        keys = key.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict):
                val = val[k]
            else:
                return default
        return val


class UserService:
    def __init__(self, config):
        self.config = config
        self.max_users = config.get("limits.max_users")
        self.users = []

    def add_user(self, name, email):
        if len(self.users) >= self.max_users:
            raise RuntimeError("User limit reached")
        user = {"name": name, "email": email, "role": self.config.get("defaults.role")}
        self.users.append(user)
        return user

    def find_user(self, email):
        for user in self.users:
            if user["email"] == email:
                return user
        return None

    def remove_user(self, email):
        user = self.find_user(email)
        self.users.remove(user)
        return user

    def list_by_role(self, role):
        return [u for u in self.users if u["role"] == role]
"""

_CONFIG_JSON = """\
{
    "limits": {
        "max_users": 100
    },
    "defaults": {
        "role": "viewer"
    }
}
"""

_STACK_TRACE = """\
Here is the bug report with stack trace:

When running the application, we get this error:

  Traceback (most recent call last):
    File "test_app.py", line 34, in test_remove_nonexistent
      svc.remove_user("nobody@test.com")
    File "app.py", line 38, in remove_user
      self.users.remove(user)
  ValueError: list.remove(x): x not in list

Also, the Config.get() method crashes instead of returning the default when a
key doesn't exist:

  Traceback (most recent call last):
    File "test_app.py", line 10, in test_config_missing_key
      result = cfg.get("nonexistent.key", "fallback")
    File "app.py", line 13, in get
      val = val[k]
  KeyError: 'nonexistent'

Please fix both bugs in app.py.
"""

_HIDDEN_TESTS = """\
import json, os
from app import Config, UserService

def _make_config(tmp_path=None):
    path = "test_config.json"
    with open(path, "w") as f:
        json.dump({"limits": {"max_users": 100}, "defaults": {"role": "viewer"}}, f)
    return Config(path)

def test_config_get_existing():
    cfg = _make_config()
    assert cfg.get("limits.max_users") == 100

def test_config_missing_key():
    cfg = _make_config()
    result = cfg.get("nonexistent.key", "fallback")
    assert result == "fallback"

def test_config_partial_missing():
    cfg = _make_config()
    result = cfg.get("limits.nonexistent", 42)
    assert result == 42

def test_add_and_find_user():
    cfg = _make_config()
    svc = UserService(cfg)
    svc.add_user("Alice", "alice@test.com")
    u = svc.find_user("alice@test.com")
    assert u["name"] == "Alice"
    assert u["role"] == "viewer"

def test_remove_existing_user():
    cfg = _make_config()
    svc = UserService(cfg)
    svc.add_user("Bob", "bob@test.com")
    removed = svc.remove_user("bob@test.com")
    assert removed["name"] == "Bob"
    assert svc.find_user("bob@test.com") is None

def test_remove_nonexistent():
    cfg = _make_config()
    svc = UserService(cfg)
    result = svc.remove_user("nobody@test.com")
    assert result is None

def test_list_by_role():
    cfg = _make_config()
    svc = UserService(cfg)
    svc.add_user("A", "a@t.com")
    svc.add_user("B", "b@t.com")
    assert len(svc.list_by_role("viewer")) == 2
    assert len(svc.list_by_role("admin")) == 0
"""

def setup(workspace: Path) -> None:
    (workspace / "app.py").write_text(_APP_CODE)
    (workspace / "config.json").write_text(_CONFIG_JSON)
    (workspace / "bug_report.txt").write_text(_STACK_TRACE)
    (workspace / "test_app.py").write_text(_HIDDEN_TESTS)

task = EvalTask(
    id="debug_stack_trace",
    prompt=(
        "Read the bug report in bug_report.txt. It contains two stack traces showing bugs in app.py. "
        "Fix both bugs in app.py:\n"
        "1. Config.get() should return the default value when a key path doesn't exist, not raise KeyError.\n"
        "2. UserService.remove_user() should handle the case where the user doesn't exist (return None instead of crashing).\n"
        "Do not change any other behavior."
    ),
    setup=setup,
    verify=TestsPasses("python3 -m pytest test_app.py -v").check,
    tags=["debugging", "stack-trace", "python", "hidden-tests"],
)
