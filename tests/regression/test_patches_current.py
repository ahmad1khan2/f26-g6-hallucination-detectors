"""The saved diffs are current, and the authors' lines they remove sit only inside functions marked as changed."""
import re

import pytest

from scripts.save_patches import PAIRS, PATCHES, make_diff
from tests.regression.authors import functions, original_path, parse, span
from tests.regression.test_no_logic_drift import changed_functions

HUNK = re.compile(r"^@@ -(\d+)")


def lf(text: str) -> str:
    return text.replace("\r\n", "\n")


def removed_original_lines(diff: str) -> list:
    """1-based line numbers, in the authors' file, of every line the patch removes."""
    removed, line = [], 0
    for text in diff.splitlines():
        if match := HUNK.match(text):
            line = int(match.group(1))
        elif text.startswith("-") and not text.startswith("---"):
            removed.append(line)
            line += 1
        elif text.startswith(" "):
            line += 1
    return removed


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_saved_patch_equals_a_fresh_diff(name):
    saved = lf((PATCHES / f"{name}.diff").read_text(encoding="utf-8"))
    assert saved == lf(make_diff(name))


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_removed_authors_lines_are_inside_functions_that_carry_a_mark(name):
    changed, _ = changed_functions(name)
    ranges = {q: span(node) for q, node in functions(parse(original_path(name))[1]).items()}
    diff = lf((PATCHES / f"{name}.diff").read_text(encoding="utf-8"))
    stray = [n for n in removed_original_lines(diff) if not any(n in ranges[q] for q in changed if q in ranges)]
    assert not stray, f"{name}: removed authors' lines {stray} are outside every changed function"
