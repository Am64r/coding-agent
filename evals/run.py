import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agent_loop"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from .check_openai_key import check_openai_key
if not check_openai_key():
    sys.exit("OPENAI_API_KEY is not available. Set it in .env or as an environment variable (e.g. GitHub secret).")

from agent import OpenAIClient
from .harness import EvalHarness
from .tasks import ALL_TASKS, TASK_MAP
import tool_library


def main():
    parser = argparse.ArgumentParser(description="Run agent evals")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--task", metavar="ID", help="Run a single task by ID")
    group.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument("--model", default="gpt-4o", help="Model to use (default: gpt-4o)")
    parser.add_argument("--compare", nargs="+", metavar="MODEL",
                        help="Run all tasks on multiple models and compare (e.g. --compare gpt-4o gpt-4o-mini)")
    parser.add_argument("--with-tools", action="store_true",
                        help="Include generated tools from the tool library")
    parser.add_argument("--quiet", action="store_true", help="Suppress agent output")
    args = parser.parse_args()

    extra_tools = None
    augmented_prompt = None
    if args.with_tools:
        lib_schemas, lib_handlers = tool_library.load_tools()
        if lib_schemas:
            extra_tools = (lib_schemas, lib_handlers)
            augmented_prompt = (
                "You are a coding agent. You have tools to read files, write files, and run shell commands.\n"
                "You also have specialized code-generation tools. Check your full tool list — if a specialized "
                "tool matches the task, USE IT instead of writing code from scratch. The specialized tools "
                "generate correct, well-tested code.\n\n"
                "All files you create and shell commands you run operate inside the workspace: agent/created_files/\n"
                "Use relative paths (e.g. \"solution.py\") and they will be placed there automatically.\n\n"
                "Work step by step:\n"
                "1. Read any existing files to understand the codebase\n"
                "2. Check if any of your specialized tools can generate the code you need\n"
                "3. If so, call the tool, then write the result to a file\n"
                "4. If not, write the code yourself\n"
                "When the task is complete, give a clear summary of what you did without calling any more tools."
            )
            print(f"[tool library] Loaded {len(lib_schemas)} tools: {', '.join(s['function']['name'] for s in lib_schemas)}")
        else:
            print("[tool library] No verified tools found.")

    if args.compare:
        runs = {}
        for model in args.compare:
            print(f"\n{'#'*60}")
            print(f"  Model: {model}")
            print(f"{'#'*60}")
            client = OpenAIClient(model=model)
            harness = EvalHarness(client=client, verbose=not args.quiet, model_name=model,
                                  extra_tools=extra_tools, system_prompt=augmented_prompt)
            runs[model] = harness.run_all(ALL_TASKS)
        EvalHarness.compare(runs)
        return

    if not args.task and not args.all:
        parser.print_help()
        print(f"\nAvailable tasks: {', '.join(TASK_MAP.keys())}")
        sys.exit(0)

    client = OpenAIClient(model=args.model)
    harness = EvalHarness(client=client, verbose=not args.quiet, model_name=args.model,
                          extra_tools=extra_tools, system_prompt=augmented_prompt)

    if args.task:
        if args.task not in TASK_MAP:
            print(f"Unknown task: {args.task!r}. Available: {', '.join(TASK_MAP.keys())}")
            sys.exit(1)
        harness.run_task(TASK_MAP[args.task])
    else:
        harness.run_all(ALL_TASKS)


if __name__ == "__main__":
    main()
