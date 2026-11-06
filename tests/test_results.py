"""Checks the real run outputs in results/<dataset>/<language>/ against data/processed/ and configs/run.yaml.

Hard problems raise AssertionError; softer ones come back as flags, which are printed in the pytest warnings
summary and written to results/flags.txt.
"""
import math
import os
import warnings
from collections import Counter

import pytest

from src.common.io import ROOT, load_config
from src.common.schema import DATASETS, read_jsonl
from src.detectors.shared import read_results

RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed"
MIN_PER_CLASS = 10  # a test split with fewer items of either class gives an unstable AUROC


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _no_duplicates(rows, where, key=lambda row: row["uid"]):
    counts = Counter(key(row) for row in rows)
    repeated = [k for k, n in counts.items() if n > 1]
    assert not repeated, f"{where}: duplicate uid {repeated[0]}"


def _known(uid, uids, where):
    assert uid in uids, f"{where}: uid {uid} is not in the processed file"


def _check_answers(rows, uids, where):
    for row in rows:
        _known(row["uid"], uids, where)
        assert row["label"] in (0, 1), f"{where}: label {row['label']!r} is not 0 or 1"
        assert _finite(row["first_token_entropy"]), f"{where}: first_token_entropy is not finite"
    _no_duplicates(rows, where)


def _check_sgm(rows, uids, where):
    for row in rows:
        _known(row["uid"], uids, where)
        assert _finite(row["score"]), f"{where}: score is not finite"
    _no_duplicates(rows, where)


def _check_sem(rows, uids, where):
    for row in rows:
        for uid in [row["uid"], *row.get("uids", [])]:
            _known(uid, uids, where)
        assert all(_finite(v) for v in row["entropies"].values()), f"{where}: an entropy is not finite"
    _no_duplicates(rows, where)


def _check_preds(rows, uids, where):
    for row in rows:
        _known(row["uid"], uids, where)
        assert _finite(row["p_hallucinated"]) and 0 <= row["p_hallucinated"] <= 1, \
            f"{where}: p_hallucinated {row['p_hallucinated']!r} is not finite in [0, 1]"
    _no_duplicates(rows, where, key=lambda row: (row["uid"], row["artifact"]))


def _expected(limit, available):
    return available if limit is None else min(limit, available)


def _covered(rows, errors, stages):
    """Distinct uids with a row or an error: an item that failed and then succeeded on resume counts once."""
    return len({r["uid"] for r in rows} | {e["uid"] for e in errors if e["stage"] in stages})


def _check_count(name, what, got, want, shortfall_allowed, flags):
    if got < want and shortfall_allowed:
        flags.append(f"{name}: {what} incomplete, {got} of {want}")
    else:
        assert got == want, f"{name}: {what} = {got}, expected {want}"


def _thin_split_flags(rows, labels, where):
    """Flag a test split whose rarer class has fewer than MIN_PER_CLASS items."""
    test = {r["uid"] for r in rows if r.get("split", "test") == "test" and r["uid"] in labels}
    counts = Counter(labels[uid] for uid in test)
    if min(counts[0], counts[1]) < MIN_PER_CLASS:
        return [f"{where}: fewer than {MIN_PER_CLASS} items of one class in the test split "
                f"({counts[0]} correct, {counts[1]} hallucinated)"]
    return []


def _check_language(directory, name, uids, run_cfg, dataset, partial):
    flags = []
    answers = read_results(directory / "answers.jsonl")
    errors = read_results(directory / "errors.jsonl")
    if errors:
        flags.append(f"{name}: {len(errors)} error rows in errors.jsonl")
    if not answers:
        flags.append(f"{name}: missing answers.jsonl")
    else:
        _check_answers(answers, uids, f"{name}/answers.jsonl")
        want = _expected(run_cfg.get("mam_limit"), len(uids))
        _check_count(name, "answers + mam errors", _covered(answers, errors, ("mam",)), want, partial, flags)
    for stage, file_name, check in (("sgm", "sgm.jsonl", _check_sgm), ("sem", "sem.jsonl", _check_sem)):
        rows = read_results(directory / file_name)
        if not rows:
            flags.append(f"{name}: missing {file_name}")
            continue
        check(rows, uids, f"{name}/{file_name}")
        if stage == "sgm":
            want = _expected((run_cfg.get("sampling_limit") or {}).get(dataset), len(uids))
            # A failed repeated PsiloQA question logs only its first uid, so a sampling error can leave a shortfall.
            sampling_failed = any(e["stage"] == "sampling" for e in errors)
            _check_count(name, "sgm rows + sgm errors", _covered(rows, errors, ("sgm", "sampling")), want,
                         partial or sampling_failed, flags)
    labels = {row["uid"]: row["label"] for row in answers}
    pred_files = sorted(directory.glob("mam_preds_seed*.jsonl"))
    if not pred_files:
        flags.append(f"{name}: missing mam_preds_seed*.jsonl")
    for path in pred_files:
        rows = read_results(path)
        _check_preds(rows, uids, f"{name}/{path.name}")
        if labels:
            flags += _thin_split_flags(rows, labels, f"{name}/{path.name}")
    return flags


def check_results(results_root, processed_dir, run_cfg, partial=False) -> list:
    """Validate every results/<dataset>/<language>/ folder; return flags, raise AssertionError on hard failures.

    partial=True is for a run stopped early: unfinished counts become flags. Bad values still fail.
    """
    flags = []
    for dataset in DATASETS:
        dataset_dir = results_root / dataset
        if not dataset_dir.is_dir():
            continue
        by_language = {}
        for record in read_jsonl(processed_dir / f"{dataset}.jsonl"):
            by_language.setdefault(record.language, set()).add(record.uid)
        for folder in sorted(p.name for p in dataset_dir.iterdir() if p.is_dir()):
            assert folder in by_language, f"{dataset}/{folder}: language is not in the processed file"
        for language, uids in by_language.items():
            if not (dataset_dir / language).is_dir():
                flags.append(f"{dataset}/{language}: not run yet")
                continue
            flags += _check_language(dataset_dir / language, f"{dataset}/{language}", uids, run_cfg, dataset, partial)
    return flags


def _real_run_cfg() -> dict:
    try:
        return load_config("run")
    except FileNotFoundError:
        return {"mam_limit": None, "sampling_limit": {}}


def test_real_results_are_consistent_with_the_processed_files():
    if not any((RESULTS / d).is_dir() for d in DATASETS):
        pytest.skip("results/ has no dataset folders yet")
    flags = check_results(RESULTS, PROCESSED, _real_run_cfg(), partial=os.environ.get("G6_PARTIAL") == "1")
    (RESULTS / "flags.txt").write_text("".join(f + "\n" for f in flags), encoding="utf-8")
    if flags:
        warnings.warn("results flags:\n" + "\n".join(flags), stacklevel=1)
