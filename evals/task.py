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


COST_PER_1K = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4.1": {"input": 0.002, "output": 0.008},
    "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
    "gpt-4.1-nano": {"input": 0.0001, "output": 0.0004},
    "o3-mini": {"input": 0.0011, "output": 0.0044},
    "gpt-5.2": {"input": 0.00175, "output": 0.014},
    "gpt-5.2-2025-12-11": {"input": 0.00175, "output": 0.014},
    "gpt-5.2-thinking": {"input": 0.00175, "output": 0.014},
}


@dataclass
class TaskResult:
    task_id: str
    passed: bool
    verify_message: str
    trajectory: list[ToolCallRecord]
    final_response: str
    total_duration_ms: float
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None
    extra_cost: float = 0.0
    tools_used: list[str] = field(default_factory=list)

    @property
    def num_tool_calls(self) -> int:
        return len(self.trajectory)

    @property
    def estimated_cost(self) -> float:
        rates = COST_PER_1K.get(self.model, {"input": 0.0, "output": 0.0})
        base = (self.input_tokens / 1000) * rates["input"] + (self.output_tokens / 1000) * rates["output"]
        return base + self.extra_cost
