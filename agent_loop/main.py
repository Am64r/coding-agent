from pathlib import Path
from dotenv import load_dotenv
from agent import Agent, OpenAIClient

load_dotenv(Path(__file__).parent.parent / ".env")


def main():
    client = OpenAIClient(model="gpt-4o")
    agent = Agent(client)

    print("Coding agent ready. Type a task or 'quit' to exit.\n")

    while True:
        try:
            task = input("Task: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not task:
            continue
        if task.lower() in ("quit", "exit", "q"):
            break

        print()
        result = agent.run(task)
        print(f"\nDone: {result.content}\n")
        print(f"  tokens: {result.input_tokens:,} in / {result.output_tokens:,} out")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
