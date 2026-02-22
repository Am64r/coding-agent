from .hello_world import task as hello_world
from .fibonacci import task as fibonacci
from .fix_the_bug import task as fix_the_bug

ALL_TASKS = [hello_world, fibonacci, fix_the_bug]
TASK_MAP = {t.id: t for t in ALL_TASKS}
