from pathlib import Path
from ..task import EvalTask
from ..verifier import TestsPasses

_HIDDEN_TESTS = """\
import math
from shapes import Shape, Circle, Rectangle, Triangle, ShapeCollection

def test_circle_area():
    c = Circle(5)
    assert abs(c.area() - math.pi * 25) < 0.001

def test_circle_perimeter():
    c = Circle(5)
    assert abs(c.perimeter() - 2 * math.pi * 5) < 0.001

def test_rectangle_area():
    r = Rectangle(4, 6)
    assert r.area() == 24

def test_rectangle_perimeter():
    r = Rectangle(4, 6)
    assert r.perimeter() == 20

def test_triangle_area():
    t = Triangle(3, 4, 5)
    assert abs(t.area() - 6.0) < 0.001

def test_triangle_perimeter():
    t = Triangle(3, 4, 5)
    assert t.perimeter() == 12

def test_invalid_triangle():
    try:
        Triangle(1, 2, 10)
        assert False, "Should raise ValueError"
    except ValueError:
        pass

def test_shape_is_abstract():
    try:
        s = Shape()
        s.area()
        assert False, "Should raise"
    except (TypeError, NotImplementedError):
        pass

def test_collection_total_area():
    sc = ShapeCollection()
    sc.add(Circle(1))
    sc.add(Rectangle(2, 3))
    expected = math.pi + 6
    assert abs(sc.total_area() - expected) < 0.001

def test_collection_sort_by_area():
    sc = ShapeCollection()
    r = Rectangle(10, 10)
    c = Circle(1)
    t = Triangle(3, 4, 5)
    sc.add(r)
    sc.add(c)
    sc.add(t)
    sorted_shapes = sc.sort_by_area()
    areas = [s.area() for s in sorted_shapes]
    assert areas == sorted(areas)

def test_collection_filter_by_type():
    sc = ShapeCollection()
    sc.add(Circle(1))
    sc.add(Circle(2))
    sc.add(Rectangle(1, 1))
    circles = sc.filter_by_type(Circle)
    assert len(circles) == 2

def test_collection_largest():
    sc = ShapeCollection()
    sc.add(Circle(1))
    sc.add(Rectangle(100, 100))
    sc.add(Triangle(3, 4, 5))
    assert isinstance(sc.largest(), Rectangle)

def test_repr():
    c = Circle(5)
    r = Rectangle(3, 4)
    assert "Circle" in repr(c) and "5" in repr(c)
    assert "Rectangle" in repr(r)
"""

def setup(workspace: Path) -> None:
    (workspace / "test_shapes.py").write_text(_HIDDEN_TESTS)

task = EvalTask(
    id="class_hierarchy",
    prompt=(
        "Create shapes.py with an OOP class hierarchy:\n\n"
        "1. Shape (abstract base class) with abstract methods area() and perimeter(), plus a __repr__.\n"
        "2. Circle(radius) — implements area and perimeter.\n"
        "3. Rectangle(width, height) — implements area and perimeter.\n"
        "4. Triangle(a, b, c) — three side lengths. Validates the triangle inequality in __init__ "
        "(raise ValueError if invalid). Uses Heron's formula for area.\n"
        "5. ShapeCollection — holds a list of shapes with methods:\n"
        "   - add(shape)\n"
        "   - total_area() — sum of all areas\n"
        "   - sort_by_area() — return shapes sorted by area ascending\n"
        "   - filter_by_type(shape_type) — return shapes matching the given class\n"
        "   - largest() — return the shape with the largest area\n"
    ),
    setup=setup,
    verify=TestsPasses("python3 -m pytest test_shapes.py -v").check,
    tags=["oop", "design", "python", "hidden-tests"],
)
