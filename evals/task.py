from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


@dataclass
class EvalTask:
    id: str
    prompt: str
    setup: Callable[[Path], None]
    verify: Callable[[Path], "VerifyResult"]
    tags: list[str] = field(default_factory=list)


@dataclass
class VerifyResult:
    passed: bool
    message: str


@dataclass
class ToolCallRecord:
    name: str
    args: dict
    result: str
    duration_ms: float


@dataclass
class TaskResult:
    task_id: str
    passed: bool
    verify_message: str
    trajectory: list[ToolCallRecord]
    final_response: str
    total_duration_ms: float
    error: Optional[str] = None

    @property
    def num_tool_calls(self) -> int:
        return len(self.trajectory)
