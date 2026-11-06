"""Hand check of X-FACTR answers: `draw` writes the sheet, a person fills checked_label, `score` reports agreement.

Labels: 1 = hallucinated (the answer matches no gold alias), 0 = correct, as everywhere in this project.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io import ROOT, hash_key, load_config
from src.common.text import matches

COLUMNS = ["uid", "language", "prompt", "gold", "answer", "matcher_label", "checked_label", "checked_by",
           "confidence", "note"]
PER_LANGUAGE = 50
MIN_AGREEMENT = 0.9
SHEET = ROOT / "results" / "xfactr" / "handcheck.csv"


def _read_csv(path) -> list:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # BOM so Excel opens it as UTF-8
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path) -> list:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def draw(results_dir, processed_path, out_path, seed=None, per_language=PER_LANGUAGE) -> list:
    """Write the sheet: `per_language` answers per language in hash order. Refuses to overwrite filled labels."""
    if Path(out_path).exists() and any(r["checked_label"].strip() for r in _read_csv(out_path)):
        raise FileExistsError(f"{out_path} already has checked labels; not overwriting")
    seed = load_config("data")["seed"] if seed is None else seed
    items = {r["uid"]: r for r in _read_jsonl(processed_path)}
    rows = []
    for answers_path in sorted(Path(results_dir).glob("*/answers.jsonl")):
        answers = [a for a in _read_jsonl(answers_path) if a["uid"] in items]
        for a in sorted(answers, key=lambda a: hash_key(seed, a["uid"]))[:per_language]:
            item = items[a["uid"]]
            rows.append({"uid": a["uid"], "language": item["language"], "prompt": item["prompt"],
                         "gold": " | ".join(item["gold"]), "answer": a["answer"],
                         "matcher_label": 0 if matches(a["answer"], item["gold"]) else 1,
                         "checked_label": "", "checked_by": "", "confidence": "", "note": ""})
    _write_csv(out_path, rows)
    return rows


def score(path, threshold=MIN_AGREEMENT):
    """(agreement per language, languages under the threshold). Refuses while any label is blank."""
    by_language = defaultdict(list)
    for row in _read_csv(path):
        by_language[row["language"]].append(row)
    blank = sorted(lang for lang, rows in by_language.items() if any(not r["checked_label"].strip() for r in rows))
    if blank:
        raise ValueError(f"blank checked_label in: {', '.join(blank)}")
    agreement = {lang: sum(r["checked_label"].strip() == str(r["matcher_label"]) for r in rows) / len(rows)
                 for lang, rows in by_language.items()}
    return agreement, [lang for lang, value in agreement.items() if value < threshold]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["draw", "score"])
    command = parser.parse_args().command
    if command == "draw":
        rows = draw(SHEET.parent, ROOT / "data" / "processed" / "xfactr.jsonl", SHEET)
        print(f"wrote {len(rows)} rows to {SHEET}")
        return
    agreement, low = score(SHEET)
    for language, value in sorted(agreement.items()):
        print(f"{language}: {value:.0%}")
    print(f"under {MIN_AGREEMENT:.0%} (add the stem-based second label): {', '.join(low) or 'none'}")


if __name__ == "__main__":
    main()
