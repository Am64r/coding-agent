# Agent Engineering — Learnings

## Module 1: Agent Loop

### What We Built
A minimal ReAct agent using OpenAI function calling with three files:
- `agent_loop/tools.py` — tool definitions and implementations
- `agent_loop/agent.py` — LLM client abstraction and ReAct loop
- `agent_loop/main.py` — interactive CLI entry point

### Architecture

**Tools** (`tools.py`):
- `read_file(path)` — reads from sandboxed workspace
- `write_file(path, content)` — writes to workspace, creates parent dirs
- `run_shell(command)` — executes shell commands in workspace, 30s timeout
- All tools operate within `agent/created_files/`
- `dispatch(name, args)` routes tool calls by name

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
