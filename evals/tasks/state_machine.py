from pathlib import Path
from ..task import EvalTask
from ..verifier import TestsPasses

_HIDDEN_TESTS = """\
from workflow import Workflow, InvalidTransition

def test_initial_state():
    w = Workflow()
    assert w.state == "draft"

def test_submit():
    w = Workflow()
    w.submit()
    assert w.state == "pending_review"

def test_approve():
    w = Workflow()
    w.submit()
    w.approve()
    assert w.state == "approved"

def test_reject():
    w = Workflow()
    w.submit()
    w.reject()
    assert w.state == "rejected"

def test_rejected_can_resubmit():
    w = Workflow()
    w.submit()
    w.reject()
    w.submit()
    assert w.state == "pending_review"

def test_publish_from_approved():
    w = Workflow()
    w.submit()
    w.approve()
    w.publish()
    assert w.state == "published"

def test_archive():
    w = Workflow()
    w.submit()
    w.approve()
    w.publish()
    w.archive()
    assert w.state == "archived"

def test_cannot_approve_draft():
    w = Workflow()
    try:
        w.approve()
        assert False, "Should raise InvalidTransition"
    except InvalidTransition:
        pass
    assert w.state == "draft"

def test_cannot_publish_pending():
    w = Workflow()
    w.submit()
    try:
        w.publish()
        assert False, "Should raise InvalidTransition"
    except InvalidTransition:
        pass

def test_cannot_submit_approved():
    w = Workflow()
    w.submit()
    w.approve()
    try:
        w.submit()
        assert False, "Should raise InvalidTransition"
    except InvalidTransition:
        pass

def test_history():
    w = Workflow()
    w.submit()
    w.approve()
    w.publish()
    assert w.history == ["draft", "pending_review", "approved", "published"]

def test_full_reject_resubmit_flow():
    w = Workflow()
    w.submit()
    w.reject()
    w.submit()
    w.approve()
    w.publish()
    assert w.state == "published"
    assert w.history == ["draft", "pending_review", "rejected", "pending_review", "approved", "published"]
"""

def setup(workspace: Path) -> None:
    (workspace / "test_workflow.py").write_text(_HIDDEN_TESTS)

task = EvalTask(
    id="state_machine",
    prompt=(
        "Create workflow.py implementing a document workflow state machine.\n\n"
        "States: draft, pending_review, approved, rejected, published, archived\n\n"
        "Transitions:\n"
        "  - draft -> pending_review (submit)\n"
        "  - pending_review -> approved (approve)\n"
        "  - pending_review -> rejected (reject)\n"
        "  - rejected -> pending_review (submit)\n"
        "  - approved -> published (publish)\n"
        "  - published -> archived (archive)\n\n"
        "Class Workflow:\n"
        "  - Starts in 'draft' state\n"
        "  - Methods: submit(), approve(), reject(), publish(), archive()\n"
        "  - Invalid transitions raise InvalidTransition (custom exception)\n"
        "  - Property `state` returns current state\n"
        "  - Property `history` returns list of all states visited (including initial)\n"
    ),
    setup=setup,
    verify=TestsPasses("python3 -m pytest test_workflow.py -v").check,
    tags=["state-machine", "design-pattern", "python", "hidden-tests"],
)
