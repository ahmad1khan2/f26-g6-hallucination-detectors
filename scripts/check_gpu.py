"""Load Llama-3.1-8B-Instruct in 4-bit, answer one English mTREx question, log the machine's versions."""
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io import ROOT
from src.common.schema import read_jsonl
from src.detectors.shared import AUTHORS_PROMPTS, chat_text, load_generator

MAX_NEW_TOKENS = 50
README = ROOT / "results" / "README.md"


def require_cuda() -> None:
    if not torch.cuda.is_available():
        sys.exit("check_gpu: CUDA is not available. Fix: install a CUDA build of torch and update the NVIDIA driver "
                 "(scripts/bootstrap.py does both), then re-run.")


def answer_one(model, tokenizer, record, prompts) -> str:
    inputs = tokenizer(chat_text(record, tokenizer, prompts), return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                             pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def versions_section(machine: str, versions: dict) -> str:
    return f"## {machine}\n" + "".join(f"- {key}: {value}\n" for key, value in versions.items())


def log_versions(path, machine: str, versions: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)  # results/ does not exist on a fresh clone before the first run
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write("\n" + versions_section(machine, versions))


def driver_version() -> str:
    out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def main() -> None:
    import transformers

    require_cuda()
    record = next(r for r in read_jsonl(ROOT / "data" / "processed" / "mtrex.jsonl") if r.language == "en")
    with open(AUTHORS_PROMPTS, encoding="utf-8") as f:
        prompts = {"mtrex": json.load(f)}
    model, tokenizer = load_generator()
    answer = answer_one(model, tokenizer, record, prompts)
    gpu, peak = torch.cuda.get_device_name(0), torch.cuda.max_memory_allocated() / 2**30
    print(f"Q: {record.meta['source']}\nA: {answer}\nGPU: {gpu}\nPeak VRAM: {peak:.2f} GiB")
    versions = {"torch": torch.__version__, "transformers": transformers.__version__,
                "CUDA": torch.version.cuda, "driver": driver_version(), "GPU": gpu, "peak VRAM (GiB)": f"{peak:.2f}"}
    log_versions(README, platform.node(), versions)


if __name__ == "__main__":
    main()
