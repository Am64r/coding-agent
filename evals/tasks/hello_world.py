from pathlib import Path
from ..task import EvalTask, VerifyResult
from ..verifier import AllOf, FileExists, ShellOutput

task = EvalTask(
    id="hello_world",
    prompt="Write a Python script called hello.py that prints 'Hello, World!' to stdout.",
    setup=lambda workspace: None,
    verify=AllOf(
        FileExists("hello.py"),
        ShellOutput("python3 hello.py", "Hello, World!", exact=True),
    ).check,
    tags=["basic", "python"],
)
