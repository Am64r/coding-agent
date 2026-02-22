import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agent_loop"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from evals.tasks import ALL_TASKS, TASK_MAP
from .pipeline import run_pipeline
import tool_library


def main():
    parser = argparse.ArgumentParser(description="Failure-to-tool generation pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--task", metavar="ID", help="Run pipeline for a single task")
    group.add_argument("--all", action="store_true", help="Run pipeline for all tasks")
    group.add_argument("--list-tools", action="store_true", help="List all generated tools")
    parser.add_argument("--cheap-model", default="gpt-4o-mini", help="Cheap model (default: gpt-4o-mini)")
    parser.add_argument("--sota-model", default="gpt-4o", help="SOTA model for tool generation (default: gpt-4o)")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max generation attempts per task (default: 3)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    if args.list_tools:
        tools = tool_library.list_tools()
        if not tools:
            print("No tools in library.")
            return
        print(f"\n{'='*60}")
        print(f"Tool Library ({len(tools)} tools)")
        print(f"{'='*60}")
        for t in tools:
            status = "verified" if t.get("verified") else "unverified"
            print(f"  {t['name']:<30} [{status}]")
            print(f"    from: {t['generated_from_task']}")
            print(f"    by:   {t['generated_by_model']}")
            if t.get("verified"):
                print(f"    verified with: {t['verified_with_model']}")
            print()
        return

    if args.task:
        if args.task not in TASK_MAP:
            print(f"Unknown task: {args.task!r}. Available: {', '.join(TASK_MAP.keys())}")
            sys.exit(1)
        tasks = [TASK_MAP[args.task]]
    elif args.all:
        tasks = ALL_TASKS
    else:
        parser.print_help()
        print(f"\nAvailable tasks: {', '.join(TASK_MAP.keys())}")
        sys.exit(0)

    results = []
    for task in tasks:
        result = run_pipeline(
            task=task,
            cheap_model=args.cheap_model,
            sota_model=args.sota_model,
            max_attempts=args.max_attempts,
            verbose=not args.quiet,
        )
        results.append(result)

    print(f"\n{'='*60}")
    print(f"Pipeline Summary")
    print(f"{'='*60}")

    already_passing = [r for r in results if r["status"] == "already_passing"]
    tool_generated = [r for r in results if r["status"] == "tool_generated"]
    gen_failed = [r for r in results if r["status"] == "generation_failed"]

    if already_passing:
        print(f"\n  Already passing ({len(already_passing)}):")
        for r in already_passing:
            print(f"    {r['task_id']}")

    if tool_generated:
        print(f"\n  Tool generated ({len(tool_generated)}):")
        for r in tool_generated:
            print(f"    {r['task_id']} -> {r['tool_name']} (attempt {r['attempts']})")

    if gen_failed:
        print(f"\n  Generation failed ({len(gen_failed)}):")
        for r in gen_failed:
            print(f"    {r['task_id']}")

    total_gen_tokens = sum(
        r.get("generation_cost", {}).get("input_tokens", 0) +
        r.get("generation_cost", {}).get("output_tokens", 0)
        for r in results
    )
    print(f"\n  Total generation tokens: {total_gen_tokens:,}")

    tools = tool_library.list_tools()
    verified = [t for t in tools if t.get("verified")]
    print(f"  Tools in library: {len(verified)} verified")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
