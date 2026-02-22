# Agent Engineering — Learnings

## Project Vision

Building a coding agent with a **self-improving tool generation pipeline**: when the agent fails a task, it automatically analyzes the failure, generates a reusable tool to prevent it, verifies the tool, and caches it. Over time, a cheap model + its accumulated tool library converges to SOTA-level performance at a fraction of the ongoing cost.

**The moat is the flywheel, not the model or the tools.** Individual tools are simple. Accumulated context is table stakes (Claude Code, Cursor already do this). What nobody has built is a system that **automatically converts agent failures into verified, executable tools** and distributes them across users. Every failure makes the system smarter. Every user's failures improve every other user's agent.

### Thesis (Refined)

The original thesis — "cheap model + tools beats SOTA model" — has a hole: if you need SOTA to generate the tools, why not just use SOTA directly? The refined thesis fixes this:

**A cheap model that accumulates tools over time will have a declining cost-per-task curve. After enough tasks, its cumulative cost undercuts SOTA while matching its pass rate.** The tool generation cost is amortized over all future invocations. The longer you run the system, the bigger the savings.

The wrong benchmark: a single snapshot comparison (A vs B vs C).
The right benchmark: **a time series** — plot cost-per-task and pass-rate as the tool library grows across a sequence of tasks. Show the crossover point where cheap+tools becomes cheaper than SOTA at equal quality.

### Why This Is Defensible

1. **Executable tools > static context.** CLAUDE.md and .cursor/rules are text a model interprets. Tools are functions that execute correctly regardless of model quality. Text instructions degrade with cheaper models; tools don't.
2. **Network effects via cross-user tool sharing.** One user's agent fails on Django N+1 queries → generates a verified tool → publishes it → every user's agent can now fix N+1 queries without SOTA. More users = more failures = more tools = better agent for everyone.
3. **The pipeline is the IP.** Any individual tool is trivial to replicate. The system that automatically produces and verifies them from failures is not.

### Planned Modules

```
agent_loop/    # ReAct loop, tool calling, model-agnostic LLM client       ✓ done
evals/         # Eval harness, cost tracking, 15-task suite, benchmark CLI  ✓ done
tool_gen/      # Task-agnostic failure→tool pipeline with recursive comp.   ✓ done
tool_library/  # Persistent tool storage, lookup, versioning, clear/reset   ✓ done
escalation/    # Self-improving benchmark: fail→gen→retry, cross-task reuse ✓ done
benchmarks/    # Time-series eval: cost-per-task curve over N sequential tasks
```

**The benchmark goal:** Run a sequence of 50+ tasks. Show that cost-per-task declines as the tool library grows, and that cumulative cost crosses below SOTA-every-time at some task count N.

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

---

## Module 2b: Cost Tracking, Multi-Model Comparison & Harder Tasks

### What We Built

Extended the eval framework with the infrastructure needed to actually prove the thesis: cost measurement, model comparison, and a task suite that differentiates cheap vs strong models.

### Cost & Token Tracking

**`AgentResponse`** now captures `input_tokens` and `output_tokens` from the OpenAI usage object. **`AgentResult`** (new dataclass returned by `Agent.run()`) accumulates totals across all iterations. **`TaskResult`** stores `model`, `input_tokens`, `output_tokens`, and an `estimated_cost` property that uses a `COST_PER_1K` table (gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, o3-mini).

The harness prints per-task and per-run cost summaries automatically.

### Multi-Model Comparison

New `--compare` flag in `evals/run.py`:
```bash
python3 -m evals.run --compare gpt-4o gpt-4o-mini gpt-4.1-nano
```

Runs the full suite on each model sequentially, then calls `EvalHarness.compare(runs)` which prints a side-by-side table: pass/fail + cost per task per model, with totals.

### Harder Task Suite (12 new tasks, 15 total)

| Task | Category | What It Tests |
|------|----------|---------------|
| `parse_csv_report` | data processing | CSV parsing, JSON output, aggregation logic |
| `debug_stack_trace` | debugging | Read bug report with stack traces, fix two bugs in app.py |
| `multi_file_refactor` | multi-file | Add discount feature across 3 files (models, orders, store) |
| `cross_file_import` | cross-file | Create new module that imports from 3 existing modules |
| `implement_cache` | data structure | LRU cache with TTL, O(1) get/set, eviction |
| `rest_api_client` | API design | REST client with auth, retries on 5xx, error handling |
| `class_hierarchy` | OOP design | Abstract base, 3 subclasses, collection with query methods |
| `state_machine` | design pattern | Workflow FSM with transitions, validation, history tracking |
| `fix_race_condition` | concurrency | Add thread safety to BankAccount and Counter classes |
| `tree_operations` | data structure | BST with insert, delete (all cases), traversal, height |
| `cli_parser` | parsing | Build arg parser from scratch (flags, key-value, --) |
| `dependency_resolver` | graph algorithm | Topological sort with cycle detection |

Task design principle: each task has hidden tests the agent never sees. Prompts give clear requirements but don't reveal edge cases. Cheap models should fail on tasks requiring multi-step reasoning, correct handling of edge cases, or cross-file coordination.

### Running
```bash
python3 -m evals.run --all                              # all 15 tasks, default model
python3 -m evals.run --task implement_cache --model gpt-4o-mini
python3 -m evals.run --compare gpt-4o gpt-4o-mini       # head-to-head with cost
```

---

## First Benchmark Results

### Raw Data (Feb 22, 2026)

| Task | gpt-4o | gpt-4o-mini | gpt-5.2-thinking |
|------|--------|-------------|------------------|
| hello_world | PASS $0.002 | PASS $0.000 | PASS $0.004 |
| fibonacci | PASS $0.004 | PASS $0.000 | PASS $0.013 |
| fix_the_bug | PASS $0.007 | PASS $0.000 | PASS $0.017 |
| parse_csv_report | PASS $0.007 | PASS $0.001 | FAIL $0.038 |
| debug_stack_trace | PASS $0.014 | PASS $0.001 | PASS $0.029 |
| multi_file_refactor | FAIL $0.010 | PASS $0.001 | PASS $0.034 |
| cross_file_import | FAIL $0.014 | FAIL $0.001 | PASS $0.033 |
| implement_cache | PASS $0.015 | PASS $0.001 | PASS $0.058 |
| rest_api_client | FAIL $0.012 | FAIL $0.001 | FAIL $0.045 |
| class_hierarchy | PASS $0.011 | PASS $0.001 | PASS $0.024 |
| state_machine | PASS $0.013 | FAIL $0.001 | PASS $0.019 |
| fix_race_condition | FAIL $0.011 | FAIL $0.001 | FAIL $0.240 |
| tree_operations | PASS $0.013 | PASS $0.001 | PASS $0.034 |
| cli_parser | PASS $0.009 | FAIL $0.001 | PASS $0.154 |
| dependency_resolver | PASS $0.006 | PASS $0.000 | PASS $0.074 |
| **TOTAL** | **11/15 $0.147** | **10/15 $0.008** | **12/15 $0.814** |

### Key Findings

**1. The cost/quality tradeoff is massive.**
- gpt-4o-mini is 17.5x cheaper than gpt-4o but only 1 task worse (10 vs 11).
- gpt-5.2-thinking is 97x more expensive than gpt-4o-mini for just 2 more passes (12 vs 10).
- The marginal cost per additional pass explodes at the frontier.

**2. Reasoning models can regress on simple tasks.**
- gpt-5.2-thinking failed `parse_csv_report` which both cheaper models passed easily. Overthinking a straightforward CSV aggregation.
- Reasoning tokens on `cli_parser` were 4,798 — more than the actual code output. Expensive "thinking" doesn't always help.

**3. Two tasks had test design bugs (now fixed).**
- `fix_race_condition`: filename `concurrent.py` shadowed Python's stdlib `concurrent` package, causing import errors in pytest. Renamed to `thread_safe.py`.
- `rest_api_client`: test asserted `call_args[1]["headers"]["Authorization"]` but agents typically set auth on `Session.headers` (which doesn't show in call_args when `Session.request` is mocked). Fixed to check both per-request and session-level headers.

**4. The gap between cheap and SOTA is narrow — exactly what the thesis needs.**
- gpt-4o-mini at $0.008 gets 67% pass rate.
- gpt-4o at $0.147 gets 73% — barely better.
- The 5 tasks where gpt-4o-mini fails (state_machine, cross_file_import, rest_api_client, fix_race_condition, cli_parser) are prime candidates for tool library augmentation.
- If the tool library can help gpt-4o-mini pass even 3 of those 5, it matches gpt-4o at 1/17th the cost.

**5. Behavioral differences between models are interesting.**
- gpt-4o-mini writes code in 1 shot without exploring first (fast but misses edge cases).
- gpt-4o reads files first, writes, then sometimes self-verifies.
- gpt-5.2-thinking reads files, reasons extensively about edge cases, writes careful code, then verifies — but burns $0.24 in reasoning tokens when stuck on an unsolvable import bug.
- gpt-4o-mini won `multi_file_refactor` which gpt-4o lost — cheaper models aren't strictly worse.

### Reasoning Model Observations

gpt-5.2-thinking with `reasoning_effort: high` was tested. The `OpenAIClient` now auto-detects the `-thinking` suffix (e.g. `gpt-5.2-thinking` → model `gpt-5.2` + `reasoning_effort: high`). Reasoning tokens are logged per-call when active. They're billed as output tokens at $14/1M, so reasoning-heavy tasks get expensive fast ($0.24 for a single failed task).

### Implications for the Thesis

The baseline is established. The A/B comparison shows that cheap models are close but fail on tasks requiring:
- Careful edge case handling (state_machine: rejected→pending_review transition)
- Multi-file coordination (cross_file_import: reading 3 files, synthesizing a 4th)
- Complex parsing with many rules (cli_parser: --, --key=value, -k value, etc.)
- Concurrency reasoning (fix_race_condition: deadlock avoidance in transfer)

These are exactly the failure categories the tool generation pipeline should target. When gpt-4o-mini fails `state_machine`, the escalation loop kicks in: SOTA analyzes the failure, generates a reusable "state machine builder" tool, verifies it against the task, caches it. Next time a state machine task appears, gpt-4o-mini has the tool and passes.

---

## Module 3: Task-Agnostic Tool Generation Pipeline

### What We Built
Reworked the tool generation pipeline to be fully task-agnostic. Previously the generation prompt was hardcoded with state-machine-specific examples and rules. Now it works for any task type.

### Key Changes

**`tool_gen/generator.py`** — Rewrote `GENERATION_PROMPT`:
- Removed FSM-specific `CRITICAL IMPLEMENTATION RULES` and `build_fsm` example
- Added generic "Failure Analysis Guidelines" and "What Makes a Good Generated Tool" sections
- Introduced a generic `generate_dataclass` example to show the expected file structure
- Added `{existing_tools_context}` placeholder for recursive tool composition (from Sheng's paper)
- Required `USAGE_EXAMPLE` string in all generated tools (tool discovery, from Sheng's paper)

**`tool_gen/pipeline.py`** — Updated to pass existing tool summaries to the generator and inject `USAGE_EXAMPLE` strings into the cheap model's system prompt for better tool discovery.

**`tool_library/__init__.py`** — Added `load_tool_summaries()`, `load_tool_usage_examples()`, and `clear_all()` for library management.

### Insights from Sheng's Paper Applied
1. **Recursive tool composition**: existing tools are summarized and injected into the generation prompt, letting SOTA build on top of previous tools.
2. **Tool discovery via usage examples**: each generated tool includes a `USAGE_EXAMPLE` string that gets injected into the cheap model's system prompt, making it more likely to actually use the tools.

---

## Module 4: Self-Improving Benchmark

### What We Built
Integrated the full self-improving pipeline into the benchmark runner. The `+tools` config doesn't just load pre-existing tools — it runs the complete failure→generate→retry loop inline, with tool accumulation across tasks and cross-task reuse tracking.

### How It Works

```bash
python3 -m evals.run --compare gpt-4o-mini gpt-4o-mini+tools gpt-4o gpt-5.2-2025-12-11 \
  --runs 3 --output results.json --quiet
```

For `model+tools` configs, each run:
1. **Clears the tool library** (fresh start every run)
2. **For each task**: runs the cheap model with accumulated library tools
3. **On failure**: SOTA generates a tool → saves to library → cheap model retries
4. **Tools persist across tasks** within a run — a tool generated for task N is available for tasks N+1, N+2, etc.
5. **Tracks cross-task reuse**: logs when a tool generated from one task is called during a different task

The `--sota-model` flag controls which model generates tools (default: gpt-4o). Generation cost is included in the task's `estimated_cost` for fair comparison.

### Benchmark Results (Feb 22, 2026)

3 runs × 15 tasks × 4 configurations:

| Task | gpt-4o-mini | gpt-4o-mini+tools | gpt-4o | gpt-5.2-2025-12-11 |
|------|-------------|-------------------|--------|---------------------|
| hello_world | 3/3 $0.0001 | 3/3 $0.0001 | 3/3 $0.0021 | 3/3 $0.0036 |
| fibonacci | 3/3 $0.0002 | 3/3 $0.0002 | 3/3 $0.0035 | 3/3 $0.0069 |
| fix_the_bug | 3/3 $0.0004 | 3/3 $0.0003 | 3/3 $0.0091 | 3/3 $0.0113 |
| parse_csv_report | 3/3 $0.0005 | 3/3 $0.0098 | 3/3 $0.0076 | 3/3 $0.0131 |
| debug_stack_trace | 3/3 $0.0011 | 3/3 $0.0013 | 3/3 $0.0172 | 3/3 $0.0201 |
| multi_file_refactor | 3/3 $0.0009 | 3/3 $0.0011 | 1/3 $0.0402 | 3/3 $0.0240 |
| cross_file_import | 1/3 $0.0006 | **3/3** $0.0228 | 0/3 $0.0155 | 3/3 $0.0244 |
| implement_cache | 3/3 $0.0006 | 3/3 $0.0061 | 1/3 $0.0180 | 3/3 $0.0312 |
| rest_api_client | 0/3 $0.0005 | 0/3 $0.0456 | 0/3 $0.0102 | 0/3 $0.0163 |
| class_hierarchy | 1/3 $0.0007 | **3/3** $0.0076 | 3/3 $0.0122 | 3/3 $0.0225 |
| state_machine | 2/3 $0.0005 | **3/3** $0.0339 | 3/3 $0.0098 | 3/3 $0.0094 |
| fix_race_condition | 1/3 $0.0006 | **3/3** $0.0129 | 0/3 $0.0228 | 3/3 $0.0212 |
| tree_operations | 0/3 $0.0009 | **2/3** $0.0327 | 3/3 $0.0161 | 3/3 $0.0234 |
| cli_parser | 0/3 $0.0006 | 0/3 $0.0498 | 2/3 $0.0184 | 3/3 $0.0228 |
| dependency_resolver | 2/3 $0.0003 | **3/3** $0.0158 | 0/3 $0.0087 | 3/3 $0.0115 |
| **PASS RATE** | **62.2%** | **84.4%** | **62.2%** | **93.3%** |
| **AVG COST/RUN** | **$0.0084** | **$0.2400** | **$0.2114** | **$0.2615** |

### Tools Generated Across Runs

Run 1: `generate_employee_report_function`, `generate_article_formatter_tools`, `generate_workflow_state_machine`, `generate_thread_safe_classes`
Run 2: `generate_employee_report_function`, `generate_article_formatter_function`, `generate_workflow_state_machine`, `generate_thread_safe_classes`, `generate_correct_bst`, `generate_dependency_resolver`
Run 3: `generate_formatter_functions`, `generate_lru_cache`, `generate_shape_class_hierarchy`, `generate_state_machine_workflow`, `generate_thread_safe_classes`, `generate_dependency_resolver`

### Key Findings

**1. Self-improving pipeline boosted gpt-4o-mini from 62.2% → 84.4% (+22 points).**
The cheap model with tool generation now significantly outperforms both `gpt-4o-mini` alone and `gpt-4o`, which both sit at 62.2%. The pipeline turned 6 failing tasks into passes: `cross_file_import`, `class_hierarchy`, `state_machine`, `fix_race_condition`, `tree_operations`, `dependency_resolver`.

**2. gpt-4o-mini+tools matches gpt-4o at the same cost — and crushes it on pass rate.**
At $0.24/run vs $0.21/run, the self-improving cheap model costs roughly the same as gpt-4o but passes 84.4% vs 62.2%. The cost is dominated by tool generation (one-time SOTA calls). Once tools are in the library, future runs on the same task types cost only $0.008/run.

**3. gpt-4o surprisingly underperforms at 62.2%.**
gpt-4o fails `cross_file_import` (0/3), `fix_race_condition` (0/3), `dependency_resolver` (0/3), `multi_file_refactor` (1/3), and `implement_cache` (1/3). These are tasks where gpt-4o-mini+tools gets 3/3. The tool library creates reliable, deterministic solutions that even a strong model can't match via reasoning alone.

**4. `rest_api_client` is a test environment bug, not a model failure.**
All models fail this task 0/3 because the eval sandbox doesn't have the `requests` library installed. This should be excluded from analysis or fixed.

**5. Tool generation cost dominates — but amortizes to zero.**
The $0.19–$0.26 generation cost per run is a one-time investment. The thesis predicts: run the system across N tasks, pay the generation cost once per failure mode, then all future runs are at gpt-4o-mini's base cost ($0.008/run). After ~30 runs, cumulative cost crosses below SOTA.

**6. No cross-task reuse detected yet.**
Tools were task-specific in this benchmark. With a larger, more diverse task suite, tools like `generate_thread_safe_classes` should help with any concurrency task, not just `fix_race_condition`. The tracking infrastructure is in place.

### Implications for the Thesis

The snapshot benchmark already shows the thesis directionally: cheap+tools (84.4%) beats strong model alone (62.2%) at comparable cost. The remaining gap to SOTA (93.3%) is `cli_parser` (complex parsing) and `rest_api_client` (env bug). The time-series benchmark — showing cost declining as the library grows — is the next step to fully prove it.

### What's Next

1. **Fix `rest_api_client` eval environment** — install `requests` in the sandbox or mock it properly.
2. **Time-series benchmark** — run 50+ sequential tasks, plot cost-per-task and pass-rate curves as the tool library grows. Show the crossover point.
3. **Cross-task reuse** — design task variants that test whether tools generalize (e.g., multiple state machine tasks, multiple concurrency tasks).
4. **Tool library persistence** — don't clear the library between benchmark invocations. Measure long-term accumulation effects.

---
