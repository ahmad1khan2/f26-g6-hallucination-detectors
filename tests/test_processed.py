"""Invariants of the built files in data/processed/ (run scripts/build_processed.py first)."""
from collections import Counter

import pytest

from src.common.io import ROOT, load_config
from src.common.schema import read_jsonl

PROCESSED = ROOT / "data" / "processed"


@pytest.fixture(scope="module")
def cfg():
    return load_config("data")


def load(name):
    return read_jsonl(PROCESSED / f"{name}.jsonl")


def test_mtrex_equal_rows_per_language_and_category(cfg):
    records = load("mtrex")
    counts = Counter((r.language, r.relation) for r in records)
    assert len(counts) == len(cfg["mtrex"]["languages"]) * len(cfg["mtrex"]["categories"])
    assert len(set(counts.values())) == 1
    assert len({r.uid for r in records}) == len(records)


def test_xfactr_every_language_fact_has_an_english_pair():
    records = load("xfactr")
    english = {r.uid for r in records if r.language == "en"}
    others = [r for r in records if r.language != "en"]
    assert others and all(r.pair_uid in english for r in others)
    assert all("[X]" not in r.prompt and "[Y]" not in r.prompt for r in records)
    assert all("___" in r.prompt for r in records)


def test_psiloqa_equal_items_per_language_and_label_matches_spans(cfg):
    records = load("psiloqa")
    counts = Counter(r.language for r in records)
    assert set(counts) == set(cfg["psiloqa"]["languages"])
    assert set(counts.values()) == {cfg["psiloqa"]["items_per_language"]}
    assert all(r.label == int(bool(r.spans)) and r.answer.strip() for r in records)
    assert len({(r.language, r.prompt, r.answer) for r in records}) == len(records)
