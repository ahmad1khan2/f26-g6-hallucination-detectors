"""matches() against the authors' mTREx rule on the authors' own data; disagreements are logged (PRD)."""
import pandas as pd

from src.common.io import ROOT, load_config
from src.common.text import matches, normalize
from tests.regression.authors import load_authors

DATA = ROOT / "third_party" / "low-resource-hallucination-detection" / "data" / "mTREx"
LOG = ROOT / "results" / "raw" / "matcher_disagreements.txt"
ROWS_PER_FILE = 500
WRONG_ANSWER = "zzz an unrelated reply"
MAX_DISAGREEMENT_RATE = 0.01


def gold_phrases(target: str) -> list:
    """The same split src/datasets/mtrex.py makes."""
    return [g for g in (normalize(x) for x in target.split("<OR>")) if g]


def compare() -> tuple:
    authors = load_authors("ModelArtifacts_g6")
    config = load_config("data")["mtrex"]
    disagreements, total = [], 0
    for folder in config["languages"]:
        for category in config["categories"]:
            table = pd.read_csv(DATA / folder / f"{category}.csv", dtype=str, keep_default_na=False)
            for number, target in enumerate(table["target"].head(ROWS_PER_FILE)):
                first_phrase = target.split("<OR>")[0].strip()
                for kind, answer in (("gold", first_phrase), ("wrong", WRONG_ANSWER)):
                    theirs = authors.post_process_mtrex_response(target, answer, "")[1]
                    ours = matches(answer, gold_phrases(target))
                    total += 1
                    if theirs != ours:
                        disagreements.append(f"{folder}/{category} row {number} {kind}: authors={theirs} "
                                             f"ours={ours} target={target!r}")
    return disagreements, total


def test_matcher_agrees_with_the_authors_rule_on_mtrex():
    disagreements, total = compare()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("".join(line + "\n" for line in disagreements), encoding="utf-8", newline="\n")
    print(f"matcher disagreements: {len(disagreements)} of {total} ({len(disagreements) / total:.4%}); see {LOG}")
    assert len(disagreements) / total < MAX_DISAGREEMENT_RATE
