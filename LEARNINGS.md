# Agent Engineering — Learnings

## Project Vision

Building a coding agent that dynamically generates task-specific tools, uses them, and accumulates a reusable tool library. The core thesis: **a cheap model + a library of purpose-built tools can match or beat a SOTA model without tools**, at a fraction of the cost.

The tool library is generated once (by a strong model or the agent itself), eval-verified, then reused across future tasks. The moat is the library — not the model.

**Planned modules:**
```
agent_loop/    # ReAct loop, tool calling, model-agnostic LLM client  ✓ done
evals/         # Eval harness, trajectory recording, hidden verifiers  ✓ done
context/       # Context management and file retrieval
memory/        # In-context, external, and episodic memory + tool library
multiagent/    # Orchestrator/subagent pattern (tool generator + tool user)
local_model/   # vLLM deployment on H100, model comparison benchmark
```

**The benchmark goal:** Run the same task suite on (A) cheap model alone, (B) SOTA model alone, (C) cheap model + generated tool library. Prove C ≥ B at lower cost.

**Next up:** Add cost/token tracking to `TaskResult` and `EvalHarness`, multi-model comparison support, and a harder task suite where cheap models fail ~40-60% without tools.

---

## Module 1: Agent Loop

### What We Built
A minimal ReAct agent using OpenAI function calling with three files:
- `agent_loop/tools/` — tool definitions and implementations (each tool is its own file)
- `agent_loop/agent.py` — LLM client abstraction and ReAct loop
- `agent_loop/main.py` — interactive CLI entry point

### Architecture

**Tools** (`tools/`):
- Each tool is its own file (`read_file.py`, `write_file.py`, `run_shell.py`), each exporting a `SCHEMA` dict and an implementation function
- `tools/_workspace.py` — shared `WORKSPACE` path and `resolve(path)` helper
- `tools/__init__.py` — aggregates `TOOL_SCHEMAS` list and `dispatch(name, args)` from all tool files
- All tools operate within `agent/created_files/`
- Adding a new tool = drop a file in `tools/`, register in `__init__.py`

**LLM Client** (`agent.py`):
- Abstract `LLMClient` base class with `chat(messages, tools) -> AgentResponse`
- `OpenAIClient` implementation (default model: `gpt-4o`)
- `ToolCall` dataclass: `id`, `name`, `args`
- `AgentResponse` dataclass: `content`, `tool_calls`, `raw_message`

**ReAct Loop** (`agent.py`):
- `Agent.run(task)` iterates up to `max_iterations` (default 20)
- Each iteration: call LLM → if text only, return it → if tool calls, execute + append results → repeat
- System prompt instructs agent to work in `agent/created_files/`, use relative paths, explore before changing

### Key Design Decisions
- `LLMClient` abstraction makes it easy to swap providers (important for module 4 local models)
- Sandboxed workspace prevents agent from touching the rest of the filesystem
- Tool results returned as strings (including errors) so the agent can self-correct

---

## Module 2: Evals

### What We Built
An industry-style eval framework modeled on SWE-bench / terminal-bench.

**Core principle**: the agent never sees the verification logic — it only gets a task prompt. Verification runs after the fact against the agent's output.

### Architecture

**`evals/task.py`** — Data models:
- `EvalTask`: `id`, `prompt`, `setup(workspace)`, `verify(workspace)`, `tags`
- `VerifyResult`: `passed`, `message`
- `ToolCallRecord`: `name`, `args`, `result`, `duration_ms`
- `TaskResult`: `task_id`, `passed`, `verify_message`, `trajectory`, `final_response`, `total_duration_ms`

**`evals/verifier.py`** — Composable verifiers (agent never sees these):
- `FileExists(path)` — checks file was created
- `FileContains(path, pattern)` — checks file content
- `ShellOutput(command, expected)` — runs a command, checks output
- `TestsPasses(command)` — runs test suite, checks exit code
- `AllOf(*verifiers)` — chains checks, short-circuits on first failure

**`evals/harness.py`** — `EvalHarness`:
1. Creates a fresh `tempfile.mkdtemp()` per task (full isolation)
2. Calls `task.setup(workspace)` to plant initial files
3. Builds a custom toolbox scoped to that temp dir (agent operates in isolation)
4. Wraps `dispatch` to record every tool call + duration (trajectory)
5. Runs agent, then calls `task.verify(workspace)`
6. Cleans up temp dir

**`evals/tasks/`** — Three tasks of increasing difficulty:
- `hello_world` — write a script that prints "Hello, World!" (baseline)
- `fibonacci` — implement `fibonacci(n)`, verified by hidden pytest suite
- `fix_the_bug` — fix `count_words()` in provided buggy code, verified by hidden tests

### Key Design Decisions
- **Workspace injection**: `Agent` now accepts `tools` and `dispatch_fn` params so the harness can swap in an isolated toolbox without changing agent code
- **Hidden tests**: tasks that use `TestsPasses` plant a test file during `setup` — the agent doesn't know what the tests check
- **Trajectory recording**: wrapping `dispatch` captures the full tool call log without modifying the agent at all

### Running
```bash
python3 -m evals.run --task hello_world
python3 -m evals.run --all
python3 -m evals.run --all --quiet   # suppress per-step output
```

### What's Missing (Next Steps for Module 2)

1. **Cost/token tracking** — `TaskResult` needs `input_tokens`, `output_tokens`, `estimated_cost`. The OpenAI response already returns usage data, just not captured yet. Required to make the economic argument measurable.

2. **Multi-model comparison** — `EvalHarness` should accept a list of models, run the same suite on each, and output a side-by-side comparison table (pass rate + cost per model).

3. **Harder task suite** — Current tasks (hello_world, fibonacci, fix_the_bug) are too easy for any model. Need 10-15 tasks where cheap models fail 40-60% without tools:
   - Multi-file codebase understanding before editing
   - Debugging from a stack trace
   - Refactoring without breaking existing tests
   - Cross-file dependency reasoning
   - Performance-constrained implementation

4. **Tool library integration** — Once built (module 3b/memory), the harness needs a mode where it loads a pre-built tool library into the agent before running tasks. This enables the A/B/C comparison.

---
