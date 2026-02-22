from dotenv import load_dotenv
from agent import Agent, OpenAIClient

load_dotenv("/Users/amr/training/.env")


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
        print(f"\nDone: {result}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
