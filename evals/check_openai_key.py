import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def check_openai_key() -> bool:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return False
    if not key.strip():
        return False
    if not key.startswith("sk-"):
        return False
    return True


def main() -> int:
    if check_openai_key():
        print("OPENAI_API_KEY is available and set as a secret")
        return 0
    print("OPENAI_API_KEY is not available or invalid (must be set and start with sk-)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
