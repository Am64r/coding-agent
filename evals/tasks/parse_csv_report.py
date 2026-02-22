from pathlib import Path
from ..task import EvalTask
from ..verifier import TestsPasses

_CSV_DATA = """\
name,department,salary,start_date
Alice,Engineering,95000,2021-03-15
Bob,Engineering,105000,2019-07-01
Carol,Marketing,82000,2022-01-10
Dave,Marketing,78000,2020-11-20
Eve,Engineering,115000,2018-06-05
Frank,Sales,72000,2023-02-28
Grace,Sales,68000,2022-09-14
Hank,Marketing,91000,2019-04-22
Ivy,Engineering,98000,2021-08-30
"""

_HIDDEN_TESTS = """\
import json
from report import generate_report

def test_report_structure():
    r = json.loads(generate_report("employees.csv"))
    assert "departments" in r
    assert "total_headcount" in r
    assert "total_salary_budget" in r

def test_headcount():
    r = json.loads(generate_report("employees.csv"))
    assert r["total_headcount"] == 9

def test_total_budget():
    r = json.loads(generate_report("employees.csv"))
    assert r["total_salary_budget"] == 804000

def test_department_breakdown():
    r = json.loads(generate_report("employees.csv"))
    eng = r["departments"]["Engineering"]
    assert eng["headcount"] == 4
    assert eng["avg_salary"] == 103250.0
    assert eng["highest_paid"] == "Eve"

def test_marketing():
    r = json.loads(generate_report("employees.csv"))
    mkt = r["departments"]["Marketing"]
    assert mkt["headcount"] == 3
    assert mkt["avg_salary"] == 83666.67 or abs(mkt["avg_salary"] - 83666.67) < 0.01

def test_sales():
    r = json.loads(generate_report("employees.csv"))
    sales = r["departments"]["Sales"]
    assert sales["headcount"] == 2
    assert sales["avg_salary"] == 70000.0
    assert sales["highest_paid"] == "Frank"
"""

def setup(workspace: Path) -> None:
    (workspace / "employees.csv").write_text(_CSV_DATA)
    (workspace / "test_report.py").write_text(_HIDDEN_TESTS)

task = EvalTask(
    id="parse_csv_report",
    prompt=(
        "Create a file called report.py with a function generate_report(csv_path) that reads a CSV file "
        "of employees (columns: name, department, salary, start_date) and returns a JSON string with:\n"
        "- total_headcount: number of employees\n"
        "- total_salary_budget: sum of all salaries (as int)\n"
        "- departments: a dict keyed by department name, each with headcount, avg_salary (float, rounded to 2 decimals), "
        "and highest_paid (name of the highest-paid employee in that department).\n"
        "The CSV file is employees.csv in the current directory."
    ),
    setup=setup,
    verify=TestsPasses("python3 -m pytest test_report.py -v").check,
    tags=["data-processing", "python", "hidden-tests"],
)
