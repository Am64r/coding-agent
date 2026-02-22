from pathlib import Path
from ..task import EvalTask
from ..verifier import TestsPasses

_MODELS_PY = """\
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
        self.active = True

    def deactivate(self):
        self.active = False

    def __repr__(self):
        return f"User({self.name!r}, {self.email!r})"


class Product:
    def __init__(self, name, price, stock):
        self.name = name
        self.price = price
        self.stock = stock

    def is_available(self):
        return self.stock > 0

    def __repr__(self):
        return f"Product({self.name!r}, ${self.price})"
"""

_ORDERS_PY = """\
class Order:
    _next_id = 1

    def __init__(self, user, items):
        self.id = Order._next_id
        Order._next_id += 1
        self.user = user
        self.items = items  # list of (product, quantity)
        self.status = "pending"

    def total(self):
        return sum(p.price * q for p, q in self.items)

    def confirm(self):
        for product, qty in self.items:
            if product.stock < qty:
                raise ValueError(f"Not enough stock for {product.name}")
            product.stock -= qty
        self.status = "confirmed"

    def cancel(self):
        if self.status == "confirmed":
            for product, qty in self.items:
                product.stock += qty
        self.status = "cancelled"
"""

_STORE_PY = """\
from models import User, Product
from orders import Order

class Store:
    def __init__(self):
        self.users = []
        self.products = []
        self.orders = []

    def add_user(self, name, email):
        u = User(name, email)
        self.users.append(u)
        return u

    def add_product(self, name, price, stock):
        p = Product(name, price, stock)
        self.products.append(p)
        return p

    def place_order(self, user, items):
        order = Order(user, items)
        order.confirm()
        self.orders.append(order)
        return order

    def get_user_orders(self, user):
        return [o for o in self.orders if o.user == user]

    def revenue(self):
        return sum(o.total() for o in self.orders if o.status == "confirmed")
"""

_HIDDEN_TESTS = """\
from models import User, Product
from orders import Order
from store import Store

def test_discount_basic():
    s = Store()
    u = s.add_user("Alice", "alice@test.com")
    p = s.add_product("Widget", 100.0, 10)
    order = s.place_order(u, [(p, 2)], discount=0.1)
    assert order.total() == 180.0

def test_discount_zero():
    s = Store()
    u = s.add_user("Bob", "bob@test.com")
    p = s.add_product("Gadget", 50.0, 5)
    order = s.place_order(u, [(p, 1)], discount=0.0)
    assert order.total() == 50.0

def test_discount_default():
    s = Store()
    u = s.add_user("Carol", "c@t.com")
    p = s.add_product("Thing", 80.0, 3)
    order = s.place_order(u, [(p, 1)])
    assert order.total() == 80.0

def test_revenue_with_discounts():
    s = Store()
    u = s.add_user("Dave", "d@t.com")
    p1 = s.add_product("A", 100.0, 10)
    p2 = s.add_product("B", 200.0, 10)
    s.place_order(u, [(p1, 1)], discount=0.5)
    s.place_order(u, [(p2, 1)], discount=0.0)
    assert s.revenue() == 250.0

def test_cancel_restores_stock():
    s = Store()
    u = s.add_user("Eve", "e@t.com")
    p = s.add_product("X", 10.0, 5)
    order = s.place_order(u, [(p, 3)], discount=0.0)
    assert p.stock == 2
    order.cancel()
    assert p.stock == 5

def test_order_without_discount_kwarg():
    s = Store()
    u = s.add_user("F", "f@t.com")
    p = s.add_product("Y", 25.0, 10)
    order = s.place_order(u, [(p, 4)])
    assert order.total() == 100.0
    assert order.status == "confirmed"
"""

def setup(workspace: Path) -> None:
    (workspace / "models.py").write_text(_MODELS_PY)
    (workspace / "orders.py").write_text(_ORDERS_PY)
    (workspace / "store.py").write_text(_STORE_PY)
    (workspace / "test_store.py").write_text(_HIDDEN_TESTS)

task = EvalTask(
    id="multi_file_refactor",
    prompt=(
        "This codebase has three files: models.py, orders.py, and store.py. They implement a simple store.\n\n"
        "Add a discount feature: Store.place_order() should accept an optional `discount` parameter (float, 0.0 to 1.0, default 0.0). "
        "The discount should be applied to the order's total. You'll need to:\n"
        "1. Modify Order.__init__() to accept and store a discount parameter (default 0.0)\n"
        "2. Modify Order.total() to apply the discount (multiply by 1 - discount)\n"
        "3. Modify Store.place_order() to pass the discount through to Order\n\n"
        "Don't break existing behavior — orders without a discount should work exactly as before."
    ),
    setup=setup,
    verify=TestsPasses("python3 -m pytest test_store.py -v").check,
    tags=["refactoring", "multi-file", "python", "hidden-tests"],
)
