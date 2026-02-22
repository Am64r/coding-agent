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
## What Makes a Good Generated Tool

The tool should:
1. Encapsulate the complex reasoning/edge-case handling the cheap model missed
2. Be a pure function: takes structured input, returns a string (generated Python code)
3. Be general enough to help with similar tasks (don't hardcode task-specific values)
4. Have a clear, descriptive name and description so the agent knows when to use it
5. Return COMPLETE, CORRECT Python source code that handles ALL edge cases

The tool should NOT:
- Access the filesystem (the agent has read_file/write_file for that)
- Be trivially simple (it should encode real logic the cheap model can't do alone)
- Import external packages (only stdlib)

CRITICAL IMPLEMENTATION RULES:
- If the tool generates a class with methods, and multiple input transitions share the same method \
name (e.g. two transitions both named "submit" from different states), you MUST merge them into ONE \
method that checks `self._state in [all_valid_from_states]`. NEVER generate duplicate method \
definitions — Python will silently drop the earlier ones.
- Always group/merge by method name before generating code.
- Test your logic mentally: would the generated code pass ALL the edge cases the agent originally failed on?

## Output Format

Return ONLY a valid Python file. No markdown fences. No explanation. Just code.

The file must define:
1. A SCHEMA dict in OpenAI function calling format
2. An implementation function whose name matches schema.function.name
3. The function must accept keyword arguments matching the schema parameters
4. The function must return a string (the generated source code)

Example — a tool that generates a state machine class (note how transitions with the same method are MERGED):

SCHEMA = {{
    "type": "function",
    "function": {{
        "name": "build_fsm",
        "description": "Generates a Python finite state machine class with states, transitions, validation, and history tracking. Handles multiple from-states per method correctly.",
        "parameters": {{
            "type": "object",
            "properties": {{
                "class_name": {{"type": "string", "description": "Name of the generated class"}},
                "states": {{"type": "array", "items": {{"type": "string"}}, "description": "Valid states"}},
                "initial_state": {{"type": "string", "description": "Starting state"}},
                "transitions": {{
                    "type": "array",
                    "items": {{
                        "type": "object",
                        "properties": {{
                            "method": {{"type": "string"}},
                            "from_states": {{"type": "array", "items": {{"type": "string"}}}},
                            "to_state": {{"type": "string"}}
                        }}
                    }},
                    "description": "Transitions. Multiple entries with same method name will be merged into one method."
                }},
                "exception_name": {{"type": "string", "description": "Custom exception class name"}}
            }},
            "required": ["class_name", "states", "initial_state", "transitions", "exception_name"]
        }}
    }}
}}


def build_fsm(class_name, states, initial_state, transitions, exception_name):
    # Group transitions by method name so we never generate duplicate methods
    from collections import defaultdict
    method_map = defaultdict(list)  # method_name -> [(from_states, to_state), ...]
    for t in transitions:
        method_map[t["method"]].append((t["from_states"], t["to_state"]))

    lines = [f"class {{exception_name}}(Exception):", "    pass", "", ""]
    lines += [f"class {{class_name}}:", "    def __init__(self):", f"        self._state = '{{initial_state}}'", f"        self._history = ['{{initial_state}}']", ""]
    lines += ["    @property", "    def state(self):", "        return self._state", ""]
    lines += ["    @property", "    def history(self):", "        return list(self._history)", ""]

    for method_name, entries in method_map.items():
        # Merge all from_states for this method
        all_from = []
        for from_states, to_state in entries:
            all_from.extend(from_states)
        # Build condition -> to_state mapping
        lines.append(f"    def {{method_name}}(self):")
        if len(entries) == 1:
            from_states, to_state = entries[0]
            lines.append(f"        if self._state not in {{all_from}}:")
            lines.append(f"            raise {{exception_name}}(f'Cannot {{{{method_name}}}} from {{{{self._state}}}}')")
            lines.append(f"        self._state = '{{to_state}}'")
        else:
            for i, (from_states, to_state) in enumerate(entries):
                kw = "if" if i == 0 else "elif"
                lines.append(f"        {{kw}} self._state in {{from_states}}:")
                lines.append(f"            self._state = '{{to_state}}'")
            lines.append(f"        else:")
            lines.append(f"            raise {{exception_name}}(f'Cannot {{{{method_name}}}} from {{{{self._state}}}}')")
        lines.append(f"        self._history.append(self._state)")
        lines.append("")

    return "\\n".join(lines)

Now generate the tool for the failed task above. Return ONLY Python code.\
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


def generate_tool(task_prompt, trajectory, verify_message, model="gpt-4o",
                  retry_info=None):
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
