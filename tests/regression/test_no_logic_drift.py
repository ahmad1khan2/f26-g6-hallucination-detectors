"""PRD: the adapted scripts change only marked code. Compares each `_g6` copy with the authors' file by AST."""
import ast

import pytest

from scripts.save_patches import PAIRS
from tests.regression.authors import MARK, adapted_path, functions, original_path, parse, span

NAMES = sorted(PAIRS)
PLAIN = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def changed_functions(name: str) -> tuple:
    """(functions that differ or are new, with their source lines; originals missing from the copy)."""
    _, original_tree = parse(original_path(name))
    ours_lines, ours_tree = parse(adapted_path(name))
    before, after = functions(original_tree), functions(ours_tree)
    changed = {q: ours_lines[span(node).start - 1:span(node).stop - 1] for q, node in after.items()
               if q not in before or ast.dump(node) != ast.dump(before[q])}
    return changed, sorted(set(before) - set(after))


def unmarked_module_statements(name: str) -> list:
    """Module-level statements of the copy (not functions or classes) that lack a mark and are not in the original."""
    _, original_tree = parse(original_path(name))
    ours_lines, ours_tree = parse(adapted_path(name))
    known = {ast.dump(n) for n in original_tree.body if not isinstance(n, PLAIN)}
    return [ast.unparse(n) for n in ours_tree.body
            if not isinstance(n, PLAIN) and ast.dump(n) not in known
            and not any(MARK in line for line in ours_lines[n.lineno - 1:n.end_lineno])]


@pytest.mark.parametrize("name", NAMES)
def test_every_changed_or_new_function_is_marked(name):
    changed, _ = changed_functions(name)
    unmarked = [q for q, lines in changed.items() if not any(MARK in line for line in lines)]
    assert not unmarked, f"{name}: changed without a '{MARK}' mark: {unmarked}"


@pytest.mark.parametrize("name", NAMES)
def test_every_original_function_still_exists(name):
    assert not changed_functions(name)[1]


@pytest.mark.parametrize("name", NAMES)
def test_module_level_changes_are_marked(name):
    assert not unmarked_module_statements(name)
