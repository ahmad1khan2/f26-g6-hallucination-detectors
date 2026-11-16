"""Phase 5 entropy table: first-token entropy per setting and language -> results/tables/entropy.csv."""
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io import ROOT, load_config

ORDER = {
    "mtrex": ["en", "de", "hi", "bn", "ur"],
    "xfactr": ["en", "zh", "ja", "vi", "tl", "sw", "yo", "ilo"],
    "psiloqa": ["en", "hi", "zh", "fi", "ar"],
}
MEASURE = ("entropy of the first generated token softmax distribution, saved during generation "
           "(the run stored this entropy instead of the full 128k-wide logits)")
COLUMNS = ["setting", "language", "resource_group", "n", "mean", "median", "std",
           "mean_correct", "mean_hallucinated", "measure"]


def _mean(values):
    return round(statistics.fmean(values), 4) if values else ""


def main():
    groups = {"en": "english", **{l["code"]: l["group"] for l in load_config("languages")["languages"]}}
    rows = []
    for dataset, languages in ORDER.items():
        for lang in languages:
            path = ROOT / "results" / dataset / lang / "answers.jsonl"
            data = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
            e = [r["first_token_entropy"] for r in data]
            rows.append({
                "setting": dataset, "language": lang,
                "resource_group": groups[lang] if dataset == "xfactr" else "",
                "n": len(e), "mean": _mean(e), "median": round(statistics.median(e), 4),
                "std": round(statistics.stdev(e), 4),
                "mean_correct": _mean([r["first_token_entropy"] for r in data if r["label"] == 0]),
                "mean_hallucinated": _mean([r["first_token_entropy"] for r in data if r["label"] == 1]),
                "measure": MEASURE})
    out = ROOT / "results" / "tables" / "entropy.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(*(r[c] for c in COLUMNS[:-1]))


if __name__ == "__main__":
    main()
