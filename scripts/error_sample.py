"""Phase 5 error sample: 15 missed hallucinations and 15 false flags per setting.

Primary detector = MAM fully_connected, seed 42, test split. Per language, a missed
hallucination is a hallucinated item scoring below the median score of the correct
items; a false flag is a correct item scoring above the median of the hallucinated
items. Candidates are ordered by hash_key(42, uid) and taken round-robin over
languages so every language appears. Output: results/errors/sample.csv.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io import ROOT, hash_key

SEED = 42
CAP = 15
ORDER = {
    "mtrex": ["en", "de", "hi", "bn", "ur"],
    "xfactr": ["en", "zh", "ja", "vi", "tl", "sw", "yo", "ilo"],
    "psiloqa": ["en", "hi", "zh", "fi", "ar"],
}
TYPES = ("missed_hallucination", "false_flag")
COLUMNS = ["setting", "language", "uid", "failure_type", "p_hallucinated", "label", "prompt", "gold",
           "answer", "spans", "generator_model", "code_a", "coder_a", "code_b", "coder_b",
           "final_code", "note"]


def find_failures(items) -> list:
    """items: dicts with uid, label (1 = hallucinated), score, for one language's test split."""
    correct = [i["score"] for i in items if i["label"] == 0]
    wrong = [i["score"] for i in items if i["label"] == 1]
    if not correct or not wrong:
        return []
    low, high = statistics.median(correct), statistics.median(wrong)
    out = [dict(i, failure_type=TYPES[0]) for i in items if i["label"] == 1 and i["score"] < low]
    return out + [dict(i, failure_type=TYPES[1]) for i in items if i["label"] == 0 and i["score"] > high]


def select(items_by_language, order, cap=CAP, seed=SEED) -> list:
    """Round-robin over languages (in `order`), hash-ordered within each, up to `cap` per type."""
    chosen = []
    for failure_type in TYPES:
        queues = [sorted((f for f in find_failures(items_by_language.get(lang, []))
                          if f["failure_type"] == failure_type),
                         key=lambda f: hash_key(seed, f["uid"]))
                  for lang in order]
        picked = []
        while len(picked) < cap and any(queues):
            for queue in queues:
                if queue and len(picked) < cap:
                    picked.append(queue.pop(0))
        chosen += picked
    return chosen


def _rows(path) -> list:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_language(dataset, language, processed) -> list:
    folder = ROOT / "results" / dataset / language
    answers = {r["uid"]: r for r in _rows(folder / "answers.jsonl")}
    items = []
    for r in _rows(folder / "mam_preds_seed42.jsonl"):
        if r["artifact"] == "fully_connected" and r.get("split", "test") == "test":
            a, p = answers[r["uid"]], processed[r["uid"]]
            spans = " | ".join(p["answer"][s["start"]:s["end"]] for s in p.get("spans") or [])
            items.append({"language": language, "uid": r["uid"], "label": a["label"],
                          "score": r["p_hallucinated"], "prompt": p["prompt"],
                          "gold": " | ".join(p.get("gold") or []), "answer": a["answer"],
                          "spans": spans, "generator_model": (p.get("meta") or {}).get("model", "")})
    return items


def main():
    rows = []
    for dataset, order in ORDER.items():
        processed = {r["uid"]: r for r in _rows(ROOT / "data" / "processed" / f"{dataset}.jsonl")}
        by_language = {lang: _load_language(dataset, lang, processed) for lang in order}
        for f in select(by_language, order):
            rows.append({**f, "setting": dataset, "p_hallucinated": round(f["score"], 4)})
    out = ROOT / "results" / "errors" / "sample.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, COLUMNS, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for r in rows:
        key = (r["setting"], r["failure_type"], r["language"])
        counts[key] = counts.get(key, 0) + 1
    for key in sorted(counts):
        print(*key, counts[key])


if __name__ == "__main__":
    main()
