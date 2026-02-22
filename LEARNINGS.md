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

_Coming next._
