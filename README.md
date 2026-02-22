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
