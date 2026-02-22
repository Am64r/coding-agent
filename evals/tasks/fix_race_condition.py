from pathlib import Path
from ..task import EvalTask
from ..verifier import TestsPasses

_BUGGY_CODE = """\
import threading

class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance

    def deposit(self, amount):
        current = self.balance
        self.balance = current + amount

    def withdraw(self, amount):
        current = self.balance
        if current >= amount:
            self.balance = current - amount
            return True
        return False

    def transfer(self, other, amount):
        if self.withdraw(amount):
            other.deposit(amount)
            return True
        return False


class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        v = self.value
        self.value = v + 1

    def decrement(self):
        v = self.value
        self.value = v - 1

    def get(self):
        return self.value
"""

_HIDDEN_TESTS = """\
import threading
from thread_safe import BankAccount, Counter

def test_concurrent_deposits():
    account = BankAccount(0)
    threads = []
    for _ in range(100):
        t = threading.Thread(target=account.deposit, args=(10,))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert account.balance == 1000

def test_concurrent_withdrawals():
    account = BankAccount(1000)
    results = []
    def try_withdraw():
        results.append(account.withdraw(10))
    threads = [threading.Thread(target=try_withdraw) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert account.balance == 0
    assert sum(results) == 100

def test_concurrent_transfers():
    a = BankAccount(1000)
    b = BankAccount(1000)
    threads = []
    for _ in range(50):
        threads.append(threading.Thread(target=a.transfer, args=(b, 10)))
        threads.append(threading.Thread(target=b.transfer, args=(a, 10)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert a.balance + b.balance == 2000

def test_counter_concurrent_increment():
    c = Counter()
    threads = [threading.Thread(target=c.increment) for _ in range(500)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert c.get() == 500

def test_counter_concurrent_mixed():
    c = Counter()
    def inc_many():
        for _ in range(100):
            c.increment()
    def dec_many():
        for _ in range(100):
            c.decrement()
    threads = [threading.Thread(target=inc_many) for _ in range(5)]
    threads += [threading.Thread(target=dec_many) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert c.get() == 0

def test_withdraw_overdraft_protection():
    account = BankAccount(100)
    results = []
    def try_withdraw():
        results.append(account.withdraw(100))
    threads = [threading.Thread(target=try_withdraw) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(results) == 1
    assert account.balance == 0
"""

def setup(workspace: Path) -> None:
    (workspace / "thread_safe.py").write_text(_BUGGY_CODE)
    (workspace / "test_thread_safe.py").write_text(_HIDDEN_TESTS)

task = EvalTask(
    id="fix_race_condition",
    prompt=(
        "The file thread_safe.py has BankAccount and Counter classes that are NOT thread-safe. "
        "They use a read-then-write pattern that causes race conditions under concurrent access.\n\n"
        "Fix both classes to be thread-safe using threading.Lock (or RLock). "
        "Key requirements:\n"
        "- deposit() and withdraw() must be atomic\n"
        "- transfer() must not allow the total money supply to change (both accounts must be updated atomically)\n"
        "- Counter.increment() and decrement() must be atomic\n"
        "- withdraw() must still return False if insufficient funds (no overdraft)\n\n"
        "Do not change the public API or method signatures."
    ),
    setup=setup,
    verify=TestsPasses("python3 -m pytest test_thread_safe.py -v").check,
    tags=["concurrency", "threading", "debugging", "python", "hidden-tests"],
)
