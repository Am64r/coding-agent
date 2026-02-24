# Agent Engineering

Building a coding agent from scratch, with an eval framework, memory, multi-agent orchestration, and local model inference on H100s.

## Project Structure

```
agent_loop/          # ReAct loop, tool calling, model-agnostic LLM client
evals/               # Eval harness, trajectory assessment, observability tracer
context/             # Context management and file retrieval
memory/              # In-context, external, and episodic memory
multiagent/          # Orchestrator/subagent pattern
local_model/         # vLLM deployment on H100, model comparison
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your API key:

```
OPENAI_API_KEY=your_key_here
```

## Running the agent

```bash
python3 agent_loop/main.py
```

## Running eval benchmarks

### Host runner (default)

```bash
python3 -m evals.run --all
python3 -m evals.run --compare gpt-4o-mini gpt-4o --runs 1 --quiet
```

### Docker runner (reproducible)

Build the benchmark image (pinned runtime/deps):

```bash
python3 -m evals.run --build-image --docker-image coding-agent-evals:latest
```

Run a fast docker smoke suite:

```bash
python3 -m evals.run --docker-smoke --runner docker --docker-image coding-agent-evals:latest --quiet
```

Run the full docker rebaseline matrix:

```bash
python3 -m evals.run --compare gpt-4o-mini gpt-4o-mini+tools gpt-4o gpt-5.2-2025-12-11 \
  --runs 3 --output results_docker.json --quiet --runner docker --docker-image coding-agent-evals:latest
```

Capture a live JSONL benchmark log (task results + tool-gen attempts):

```bash
python3 -m evals.run --compare gpt-4o-mini+tools --runs 1 --quiet \
  --runner docker --docker-image coding-agent-evals:latest \
  --benchmark-log benchmark_live.jsonl
```

Key runner flags:

```bash
--runner host|docker
--docker-image <name:tag>
--build-image
--docker-smoke
```

Tool generation uses a no-leakage default: hidden verifier output is not passed into tool generation prompts. To opt into old behavior for debugging:

```bash
python3 -m evals.run ... --allow-verifier-feedback
python3 -m tool_gen.run ... --allow-verifier-feedback
```
