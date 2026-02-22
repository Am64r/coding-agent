import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agent_loop"))

from agent import OpenAIClient

GENERATION_PROMPT = """\
You are a tool engineering expert. An AI coding agent (using a cheap, weak model) attempted a coding task and FAILED.

Your job: analyze the failure and generate a reusable Python tool that the cheap model can call to solve this type of task correctly. The tool should encapsulate the complex reasoning and edge-case handling that the cheap model failed to do on its own.

## Failed Task
{task_prompt}

## Agent's Tool Call Trajectory
{trajectory}

## Verification Result (why it failed)
{verify_message}
{retry_context}
{existing_tools_context}
## Failure Analysis Guidelines

Before generating the tool, reason about:
1. What specific logic or edge cases did the cheap model get wrong?
2. What is the GENERAL category of this failure? (e.g. state management, parsing, \
data structure invariants, cross-file coordination, concurrency, algorithm correctness)
3. What reusable abstraction would prevent this class of failure for similar future tasks?

## What Makes a Good Generated Tool

The tool should:
1. Encapsulate the complex reasoning/edge-case handling the cheap model missed
2. Be a pure function: takes structured input, returns a string (generated Python code)
3. Be GENERAL — parameterize names, types, structure so it works for similar tasks, not just this one
4. Have a clear, descriptive name and description so the agent knows when to use it
5. Return COMPLETE, CORRECT Python source code that handles ALL edge cases
6. Mentally verify: would the generated code pass ALL the edge cases the agent originally failed on?

The tool should NOT:
- Access the filesystem (the agent has read_file/write_file for that)
- Be trivially simple (it should encode real logic the cheap model can't do alone)
- Import external packages (only stdlib)
- Hardcode task-specific values (names, strings, counts) — accept them as parameters

## Output Format

Return ONLY a valid Python file. No markdown fences. No explanation. Just code.

The file must define:
1. A `SCHEMA` dict in OpenAI function calling format
2. An implementation function whose name matches `SCHEMA["function"]["name"]`
3. The function must accept keyword arguments matching the schema parameters
4. The function must return a string (the generated source code)
5. A `USAGE_EXAMPLE` string showing a sample function call and what it returns

Short example of the expected file structure:

SCHEMA = {{
    "type": "function",
    "function": {{
        "name": "generate_dataclass",
        "description": "Generates a Python dataclass with typed fields and optional defaults.",
        "parameters": {{
            "type": "object",
            "properties": {{
                "class_name": {{"type": "string", "description": "Name of the class"}},
                "fields": {{
                    "type": "array",
                    "items": {{
                        "type": "object",
                        "properties": {{
                            "name": {{"type": "string"}},
                            "type": {{"type": "string"}},
                            "default": {{"type": "string"}}
                        }}
                    }},
                    "description": "List of fields with name, type, and optional default"
                }}
            }},
            "required": ["class_name", "fields"]
        }}
    }}
}}

USAGE_EXAMPLE = '''
generate_dataclass(class_name="Point", fields=[{{"name": "x", "type": "float"}}, {{"name": "y", "type": "float", "default": "0.0"}}])
# Returns:
# from dataclasses import dataclass
#
# @dataclass
# class Point:
#     x: float
#     y: float = 0.0
'''

def generate_dataclass(class_name, fields):
    lines = ["from dataclasses import dataclass", "", "@dataclass", f"class {{class_name}}:"]
    for f in fields:
        line = f"    {{f['name']}}: {{f['type']}}"
        if "default" in f:
            line += f" = {{f['default']}}"
        lines.append(line)
    return "\\n".join(lines)

This is just an example of the structure. Your generated tool should address the specific failure above.
Now generate the tool. Return ONLY Python code.\
"""


RETRY_CONTEXT_TEMPLATE = """
## Previous Attempt Failed
The previously generated tool was called by the agent but STILL produced wrong code.
Tool name: {tool_name}
Agent's output with the tool still failed verification: {retry_verify_message}
Fix the tool so the generated code passes. Think carefully about what edge case is still broken.
"""


def format_trajectory(trajectory):
    lines = []
    for i, tc in enumerate(trajectory, 1):
        args_str = json.dumps(tc.args, indent=2) if isinstance(tc.args, dict) else str(tc.args)
        result_preview = tc.result[:500] + "..." if len(tc.result) > 500 else tc.result
        lines.append(f"Step {i}: {tc.name}({args_str})\n  -> {result_preview}")
    return "\n\n".join(lines) if lines else "(no tool calls recorded)"


def format_existing_tools(existing_tools):
    if not existing_tools:
        return ""
    lines = ["## Existing Tool Library",
             "",
             "These tools are already available in the agent's toolkit. If one of them can serve "
             "as a building block for the new tool, your generated tool may import and call it.",
             ""]
    for schema in existing_tools:
        func = schema.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {}).get("properties", {})
        param_list = ", ".join(
            f"{k}: {v.get('type', 'any')}" for k, v in params.items()
        )
        lines.append(f"- **{name}**({param_list}): {desc}")
    lines.append("")
    return "\n".join(lines)


def generate_tool(task_prompt, trajectory, verify_message, model="gpt-4o",
                  retry_info=None, existing_tools=None):
    client = OpenAIClient(model=model)

    retry_context = ""
    if retry_info:
        retry_context = RETRY_CONTEXT_TEMPLATE.format(
            tool_name=retry_info["tool_name"],
            retry_verify_message=retry_info["verify_message"][:500],
        )

    prompt = GENERATION_PROMPT.format(
        task_prompt=task_prompt,
        trajectory=format_trajectory(trajectory),
        verify_message=verify_message,
        retry_context=retry_context,
        existing_tools_context=format_existing_tools(existing_tools),
    )

    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        tools=None,
    )

    code = response.content.strip()
    if code.startswith("```"):
        code = re.sub(r'^```\w*\n', '', code)
        code = re.sub(r'\n```\s*$', '', code)

    return code, response.input_tokens, response.output_tokens
