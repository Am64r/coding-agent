from pathlib import Path
from ..task import EvalTask
from ..verifier import TestsPasses

_BUGGY_CODE = """\
def count_words(text):
    if not text:
        return 0
    words = text.split(" ")
    return len(words)


def most_common_word(text):
    words = text.lower().split()
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return max(counts, key=lambda w: counts[w])
"""

_HIDDEN_TESTS = """\
from word_utils import count_words, most_common_word

def test_count_words_basic():
    assert count_words("hello world") == 2

def test_count_words_multiple_spaces():
    assert count_words("hello  world") == 2

def test_count_words_tabs_and_newlines():
    assert count_words("hello\\tworld\\nfoo") == 3

def test_count_words_empty():
    assert count_words("") == 0

def test_count_words_whitespace_only():
    assert count_words("   ") == 0

def test_most_common_word():
    assert most_common_word("the cat sat on the mat") == "the"

def test_most_common_word_case_insensitive():
    assert most_common_word("Apple apple APPLE") == "apple"
"""

def setup(workspace: Path) -> None:
    (workspace / "word_utils.py").write_text(_BUGGY_CODE)
    (workspace / "test_word_utils.py").write_text(_HIDDEN_TESTS)

task = EvalTask(
    id="fix_the_bug",
    prompt=(
        "The file word_utils.py contains two functions: count_words(text) and most_common_word(text). "
        "There is a bug in count_words — it doesn't handle multiple consecutive spaces or other whitespace correctly. "
        "Fix the bug. Do not change most_common_word."
    ),
    setup=setup,
    verify=TestsPasses("python3 -m pytest test_word_utils.py -v").check,
    tags=["debugging", "python", "hidden-tests"],
)
