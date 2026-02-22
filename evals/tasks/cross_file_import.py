from pathlib import Path
from ..task import EvalTask
from ..verifier import TestsPasses

_UTILS_PY = """\
def slugify(text):
    return text.lower().strip().replace(" ", "-")


def truncate(text, max_len=100):
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."
"""

_VALIDATORS_PY = """\
import re

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_slug(slug):
    return bool(re.match(r'^[a-z0-9]+(-[a-z0-9]+)*$', slug))
"""

_MODELS_PY = """\
from utils import slugify, truncate
from validators import validate_email

class Article:
    def __init__(self, title, body, author_email):
        if not validate_email(author_email):
            raise ValueError(f"Invalid email: {author_email}")
        self.title = title
        self.slug = slugify(title)
        self.body = body
        self.summary = truncate(body, 200)
        self.author_email = author_email
        self.tags = []

    def add_tag(self, tag):
        slug = slugify(tag)
        if slug not in self.tags:
            self.tags.append(slug)

    def has_tag(self, tag):
        return slugify(tag) in self.tags

    def word_count(self):
        return len(self.body.split())
"""

_HIDDEN_TESTS = """\
from models import Article
from utils import slugify, truncate
from validators import validate_email, validate_slug
from formatters import format_article, format_listing

def test_format_article_basic():
    a = Article("Hello World", "This is the body.", "a@b.com")
    out = format_article(a)
    assert "Hello World" in out
    assert "hello-world" in out
    assert "This is the body." in out
    assert "a@b.com" in out

def test_format_article_with_tags():
    a = Article("Test Post", "Body text here.", "x@y.com")
    a.add_tag("Python")
    a.add_tag("Coding")
    out = format_article(a)
    assert "python" in out
    assert "coding" in out

def test_format_listing():
    articles = [
        Article("First", "Body one.", "a@b.com"),
        Article("Second", "Body two.", "c@d.com"),
    ]
    out = format_listing(articles)
    assert "first" in out
    assert "second" in out
    assert "2 articles" in out.lower() or "2 article" in out.lower()

def test_format_listing_empty():
    out = format_listing([])
    assert "0 articles" in out.lower() or "no articles" in out.lower() or "0 article" in out.lower()

def test_slugify_used_in_format():
    a = Article("My Great Title", "Content.", "z@w.com")
    out = format_article(a)
    assert "my-great-title" in out

def test_validate_slug_in_formatter():
    from validators import validate_slug
    assert validate_slug("hello-world")
    assert not validate_slug("Hello World")
"""

def setup(workspace: Path) -> None:
    (workspace / "utils.py").write_text(_UTILS_PY)
    (workspace / "validators.py").write_text(_VALIDATORS_PY)
    (workspace / "models.py").write_text(_MODELS_PY)
    (workspace / "test_formatters.py").write_text(_HIDDEN_TESTS)

task = EvalTask(
    id="cross_file_import",
    prompt=(
        "This project has utils.py (slugify, truncate), validators.py (validate_email, validate_slug), "
        "and models.py (Article class). Read all files to understand the codebase.\n\n"
        "Create a new file formatters.py with two functions:\n"
        "1. format_article(article) — returns a formatted string with the article's title, slug, body, "
        "author email, and tags (if any). Use the existing slugify/truncate from utils.py and validate_slug "
        "from validators.py as needed.\n"
        "2. format_listing(articles) — returns a string summarizing a list of articles, including a count "
        "and each article's title and slug.\n\n"
        "The formatters should import from the existing modules and work with the Article class from models.py."
    ),
    setup=setup,
    verify=TestsPasses("python3 -m pytest test_formatters.py -v").check,
    tags=["cross-file", "imports", "python", "hidden-tests"],
)
