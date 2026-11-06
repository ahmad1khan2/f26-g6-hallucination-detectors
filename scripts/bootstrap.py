"""One-shot setup of the GPU machine from a fresh clone: py -3.11 scripts/bootstrap.py (stdlib only, safe to re-run)."""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# CUDA wheel indexes that carry torch for cp311 (checked at download.pytorch.org/whl/torch for 2.15); newest last.
CUDA_INDEXES = [((12, 6), "cu126"), ((12, 8), "cu128"), ((13, 0), "cu130"), ((13, 2), "cu132"), ((13, 4), "cu134")]
PIP_FLAGS = ["--timeout", "300", "--retries", "8"]


def parse_cuda_version(smi_text: str) -> tuple:
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", smi_text)
    if not match:
        raise RuntimeError("nvidia-smi output has no 'CUDA Version' field. Fix: update the NVIDIA driver, then re-run.")
    return int(match[1]), int(match[2])


def torch_index(version: tuple) -> str:
    fits = [tag for floor, tag in CUDA_INDEXES if floor <= version]
    if not fits:
        raise RuntimeError(f"NVIDIA driver supports CUDA {version[0]}.{version[1]}, below the oldest wheel (12.6). "
                           "Fix: update the NVIDIA driver, then re-run.")
    return f"https://download.pytorch.org/whl/{fits[-1]}"


def torch_pin(lock_text: str):
    """'torch==X' from a lock file (local tag like +cu132 dropped, the index decides it), or None."""
    match = re.search(r"^torch==([^\s+;]+)", lock_text, re.MULTILINE)
    return f"torch=={match[1]}" if match else None


def venv_python(root: Path) -> Path:
    return root / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def fail(message: str):
    sys.exit(f"bootstrap: {message}")


def run(command, **kwargs):
    print("+", " ".join(str(c) for c in command), flush=True)
    return subprocess.run([str(c) for c in command], cwd=ROOT, check=True, **kwargs)


def check_prerequisites() -> str:
    if sys.version_info[:2] != (3, 11):
        fail(f"Python 3.11 is required, found {sys.version.split()[0]}. Fix: run it as `py -3.11 scripts/bootstrap.py`.")
    for tool, fix in (("git", "install Git and reopen the terminal"), ("nvidia-smi", "install the NVIDIA driver")):
        if not shutil.which(tool):
            fail(f"`{tool}` is not on PATH. Fix: {fix}.")
    return subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout


def install_torch(python, pin, index: str) -> None:
    """The tested torch if this driver's index has it, else that index's newest (older drivers lack new wheels)."""
    command = [python, "-m", "pip", "install", "--index-url", index, *PIP_FLAGS]
    if pin:
        try:
            run([*command, pin])
            return
        except subprocess.CalledProcessError:
            print(f"bootstrap: {pin} has no wheel on {index}; using its newest torch. "
                  "Tests ran on the pinned version, so updating the NVIDIA driver is safer.", flush=True)
    run([*command, "torch"])


def install_requirements(python: Path, smi_text: str) -> None:
    lock = ROOT / "requirements-gpu.lock.txt"
    source = lock if lock.exists() else ROOT / "requirements-gpu.txt"
    pin = torch_pin(lock.read_text(encoding="utf-8")) if lock.exists() else None
    install_torch(python, pin, torch_index(parse_cuda_version(smi_text)))
    # torch is already in; a temp copy in ROOT keeps the `-r requirements.txt` line resolving.
    rest = [line for line in source.read_text(encoding="utf-8").splitlines() if not re.match(r"torch(==|\s*$)", line)]
    filtered = ROOT / ".bootstrap-requirements.txt"
    filtered.write_text("\n".join(rest) + "\n", encoding="utf-8")
    try:
        run([python, "-m", "pip", "install", "-r", filtered, *PIP_FLAGS])
    finally:
        filtered.unlink()


def has_hf_token(python: Path) -> bool:
    if os.environ.get("HF_TOKEN"):
        return True
    code = "from huggingface_hub import get_token; print(bool(get_token()))"
    out = subprocess.run([str(python), "-c", code], capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip() == "True"


def main() -> None:
    smi_text = check_prerequisites()
    os.environ.setdefault("HF_HOME", str(Path.home() / "hf_cache"))
    os.environ["PYTHONUTF8"] = "1"
    run(["git", "submodule", "update", "--init"])
    python = venv_python(ROOT)
    if not python.exists():
        run([sys.executable, "-m", "venv", ROOT / ".venv"])
    install_requirements(python, smi_text)
    if not has_hf_token(python):
        run([python.parent / ("hf.exe" if sys.platform == "win32" else "hf"), "auth", "login"])
    print(f"\nSet these permanently so later runs find them:\n  setx HF_HOME \"{os.environ['HF_HOME']}\"\n  setx PYTHONUTF8 1")
    run([python, "scripts/check_gpu.py"])


if __name__ == "__main__":
    main()
