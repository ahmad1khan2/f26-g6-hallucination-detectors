"""Environment check: every Phase 2 package imports. Run: python scripts/check_env.py"""
import importlib
import sys

PACKAGES = ["pandas", "pyarrow", "numpy", "scipy", "sklearn", "matplotlib",
            "yaml", "tqdm", "datasets", "nbconvert", "pytest"]

print("python", sys.version.split()[0])
for name in PACKAGES:
    module = importlib.import_module(name)
    print(f"ok {name} {getattr(module, '__version__', '')}")
