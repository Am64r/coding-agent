SCHEMA = {
    "type": "function",
    "function": {
        "name": "build_workflow_state_machine",
        "description": "Generates a Python workflow state machine class with states, transitions, validation, and history tracking. Handles multiple from-states per method correctly.",
        "parameters": {
            "type": "object",
            "properties": {
                "class_name": {"type": "string", "description": "Name of the generated class"},
                "states": {"type": "array", "items": {"type": "string"}, "description": "Valid states"},
                "initial_state": {"type": "string", "description": "Starting state"},
                "transitions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "method": {"type": "string"},
                            "from_states": {"type": "array", "items": {"type": "string"}},
                            "to_state": {"type": "string"}
                        }
                    },
                    "description": "Transitions. Multiple entries with same method name will be merged into one method."
                },
                "exception_name": {"type": "string", "description": "Custom exception class name"}
            },
            "required": ["class_name", "states", "initial_state", "transitions", "exception_name"]
        }
    }
}


def build_workflow_state_machine(class_name, states, initial_state, transitions, exception_name):
    from collections import defaultdict
    method_map = defaultdict(list)  # method_name -> [(from_states, to_state), ...]

    # Group transitions by method name
    for t in transitions:
        method_map[t["method"]].append((t["from_states"], t["to_state"]))

    lines = [f"class {exception_name}(Exception):", "    pass", "", ""]
    lines += [f"class {class_name}:", "    def __init__(self):", f"        self._state = '{initial_state}'", f"        self._history = ['{initial_state}']", ""]
    lines += ["    @property", "    def state(self):", "        return self._state", ""]
    lines += ["    @property", "    def history(self):", "        return list(self._history)", ""]

    for method_name, entries in method_map.items():
        # Merge all from_states for this method
        all_from = []
        for from_states, to_state in entries:
            all_from.extend(from_states)
        unique_from_states = list(set(all_from))

        # Build condition -> to_state mapping
        lines.append(f"    def {method_name}(self):")
        if len(entries) == 1:
            from_states, to_state = entries[0]
            lines.append(f"        if self._state not in {unique_from_states}:")
            lines.append(f"            raise {exception_name}(f'Cannot {method_name} from {{self._state}}')")
            lines.append(f"        self._state = '{to_state}'")
        else:
            for i, (from_states, to_state) in enumerate(entries):
                kw = "if" if i == 0 else "elif"
                lines.append(f"        {kw} self._state in {from_states}:")
                lines.append(f"            self._state = '{to_state}'")
            lines.append(f"        else:")
            lines.append(f"            raise {exception_name}(f'Cannot {method_name} from {{self._state}}')")
        lines.append(f"        self._history.append(self._state)")
        lines.append("")

    return "\n".join(lines)