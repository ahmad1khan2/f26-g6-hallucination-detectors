"""Raw data -> data/processed/{mtrex,xfactr,psiloqa}.jsonl, plus build_log.txt with the step counts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io import ROOT, load_config
from src.common.schema import write_jsonl
from src.datasets import mtrex, psiloqa, xfactr

OUT = ROOT / "data" / "processed"


def main():
    cfg = load_config("data")
    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    for name, module in (("mtrex", mtrex), ("xfactr", xfactr), ("psiloqa", psiloqa)):
        write_jsonl(OUT / f"{name}.jsonl", module.build(cfg, log))
    (OUT / "build_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
