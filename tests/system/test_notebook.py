"""System test: F26-6.ipynb runs top to bottom from the saved results (no GPU, no model)."""
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[2]


def test_notebook_executes_without_errors():
    notebook = nbformat.read(ROOT / "F26-6.ipynb", as_version=4)

    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(notebook, {"metadata": {"path": str(ROOT)}})
