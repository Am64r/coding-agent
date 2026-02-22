from .hello_world import task as hello_world
from .fibonacci import task as fibonacci
from .fix_the_bug import task as fix_the_bug
from .parse_csv_report import task as parse_csv_report
from .debug_stack_trace import task as debug_stack_trace
from .multi_file_refactor import task as multi_file_refactor
from .cross_file_import import task as cross_file_import
from .implement_cache import task as implement_cache
from .rest_api_client import task as rest_api_client
from .class_hierarchy import task as class_hierarchy
from .state_machine import task as state_machine
from .fix_race_condition import task as fix_race_condition
from .tree_operations import task as tree_operations
from .cli_parser import task as cli_parser
from .dependency_resolver import task as dependency_resolver

ALL_TASKS = [
    hello_world,
    fibonacci,
    fix_the_bug,
    parse_csv_report,
    debug_stack_trace,
    multi_file_refactor,
    cross_file_import,
    implement_cache,
    rest_api_client,
    class_hierarchy,
    state_machine,
    fix_race_condition,
    tree_operations,
    cli_parser,
    dependency_resolver,
]
TASK_MAP = {t.id: t for t in ALL_TASKS}
