import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agent_loop"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from agent import OpenAIClient
from .harness import EvalHarness
from .tasks import ALL_TASKS, TASK_MAP


def main():
    parser = argparse.ArgumentParser(description="Run agent evals")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--task", metavar="ID", help="Run a single task by ID")
    group.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument("--model", default="gpt-4o", help="Model to use (default: gpt-4o)")
    parser.add_argument("--quiet", action="store_true", help="Suppress agent output")
    args = parser.parse_args()

    if not args.task and not args.all:
        parser.print_help()
        print(f"\nAvailable tasks: {', '.join(TASK_MAP.keys())}")
        sys.exit(0)

    client = OpenAIClient(model=args.model)
    harness = EvalHarness(client=client, verbose=not args.quiet)

    if args.task:
        if args.task not in TASK_MAP:
            print(f"Unknown task: {args.task!r}. Available: {', '.join(TASK_MAP.keys())}")
            sys.exit(1)
        harness.run_task(TASK_MAP[args.task])
    else:
        harness.run_all(ALL_TASKS)


if __name__ == "__main__":
    main()
