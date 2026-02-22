# Agent Engineering

Building a coding agent from scratch, with an eval framework, memory, multi-agent orchestration, and local model inference on H100s.

## Project Structure

```
01_agent_loop/       # ReAct loop, tool calling, model-agnostic LLM client
02_evals/            # Eval harness, trajectory assessment, observability tracer
03_context/          # Context management and file retrieval
03b_memory/          # In-context, external, and episodic memory
03c_multiagent/      # Orchestrator/subagent pattern
04_local_model/      # vLLM deployment on H100, model comparison
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
python3 01_agent_loop/main.py
```
