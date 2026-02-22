import json
import os
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from tools import TOOL_SCHEMAS, dispatch as _default_dispatch

SYSTEM_PROMPT = """\
You are a coding agent. You have tools to read files, write files, and run shell commands.

All files you create and shell commands you run operate inside the workspace: agent/created_files/
Use relative paths (e.g. "solution.py") and they will be placed there automatically.

Work step by step. Use tools to explore and gather information before making changes.
When the task is complete, give a clear summary of what you did without calling any more tools.\
"""


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class AgentResponse:
    content: Optional[str]
    tool_calls: list[ToolCall]
    raw_message: dict


class LLMClient:
    def chat(self, messages: list, tools: list = None) -> AgentResponse:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4o", api_key: str = None):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    def chat(self, messages: list, tools: list = None) -> AgentResponse:
        kwargs = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    args=json.loads(tc.function.arguments)
                ))

        raw_message = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            raw_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]

        return AgentResponse(
            content=message.content,
            tool_calls=tool_calls,
            raw_message=raw_message
        )


class Agent:
    def __init__(
        self,
        client: LLMClient,
        max_iterations: int = 20,
        verbose: bool = True,
        tools: list = None,
        dispatch_fn: callable = None,
    ):
        self.client = client
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.tools = tools if tools is not None else TOOL_SCHEMAS
        self.dispatch_fn = dispatch_fn or _default_dispatch

    def run(self, task: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task}
        ]

        for iteration in range(self.max_iterations):
            response = self.client.chat(messages, tools=self.tools)
            messages.append(response.raw_message)

            if not response.tool_calls:
                return response.content or "(no response)"

            for tc in response.tool_calls:
                if self.verbose:
                    print(f"  [{iteration + 1}] {tc.name}({json.dumps(tc.args)})")

                result = self.dispatch_fn(tc.name, tc.args)

                if self.verbose:
                    preview = result[:300] + "..." if len(result) > 300 else result
                    print(f"      → {preview}\n")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })

        return "Reached maximum iterations without completing the task."
