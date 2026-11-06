"""Helpers for the regression tests: the authors' original modules and their function tables."""
import ast
import importlib.util
import sys
from functools import lru_cache

from scripts.save_patches import ORIGINALS, PAIRS
from src.common.io import ROOT

MARK = "# G6:"


def original_path(name: str):
    return ORIGINALS / PAIRS[name]


def adapted_path(name: str):
    return ROOT / "src" / "detectors" / f"{name}.py"


@lru_cache(maxsize=None)
def load_authors(name: str):
    """The authors' module for adapted script `name`, imported by path under its own module name."""
    path = original_path(name)
    spec = importlib.util.spec_from_file_location(f"authors_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    keep, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # no __pycache__ inside the read-only submodule
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = keep
    return module


def functions(tree: ast.Module) -> dict:
    """{qualified name: node} for module-level functions and class methods (Class.method)."""
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            found[node.name] = node
        elif isinstance(node, ast.ClassDef):
            found |= {f"{node.name}.{m.name}": m for m in node.body if isinstance(m, ast.FunctionDef)}
    return found


def parse(path) -> tuple:
    """(source lines, tree) of a file."""
    text = path.read_text(encoding="utf-8")
    return text.splitlines(), ast.parse(text)


def span(node) -> range:
    """1-based line numbers of a function including its decorators."""
    first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
    return range(first, node.end_lineno + 1)
