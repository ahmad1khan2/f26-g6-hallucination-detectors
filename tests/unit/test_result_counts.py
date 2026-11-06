"""Result counting across restarts and early stops, plus small run_phase3 and shared edge cases."""
import json

import pytest
import torch

from scripts import run_phase3
from src.common.schema import Record, write_jsonl
from src.detectors import shared
from tests.test_results import check_results


def _processed(tmp_path, n=4):
    folder = tmp_path / "processed"
    folder.mkdir()
    write_jsonl(folder / "xfactr.jsonl", [Record(uid=f"u{i}", dataset="xfactr", language="en", prompt="p ___",
                                                 relation="r", gold=["g"]) for i in range(n)])
    return folder


def _answer(uid):
    return {"uid": uid, "answer": "a", "label": 0, "label_rule": "matches", "first_token_entropy": 1.0}


def _language_dir(tmp_path):
    return tmp_path / "results" / "xfactr" / "en"


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _check(tmp_path, partial=False, limit=None):
    cfg = {"mam_limit": None, "sampling_limit": {"xfactr": limit}}
    return check_results(tmp_path / "results", tmp_path / "processed", cfg, partial=partial)


def test_a_failed_item_that_succeeds_on_resume_counts_once(tmp_path):
    _processed(tmp_path)
    directory = _language_dir(tmp_path)
    _write(directory / "answers.jsonl", [_answer(f"u{i}") for i in range(4)])
    _write(directory / "errors.jsonl", [{"uid": "u1", "stage": "mam", "error": "OutOfMemoryError: x"}])
    assert any("1 error rows" in flag for flag in _check(tmp_path))


def test_an_item_that_failed_sampling_is_a_flag_and_not_a_failure(tmp_path):
    _processed(tmp_path)
    directory = _language_dir(tmp_path)
    _write(directory / "answers.jsonl", [_answer(f"u{i}") for i in range(4)])
    _write(directory / "sgm.jsonl", [{"uid": f"u{i}", "score": 0.5} for i in range(3)])
    _write(directory / "errors.jsonl", [{"uid": "u3", "stage": "sampling", "error": "OutOfMemoryError: x"}])
    assert any("error rows" in flag for flag in _check(tmp_path))


def test_a_failed_repeated_question_flags_its_uncounted_uids(tmp_path):
    _processed(tmp_path)
    directory = _language_dir(tmp_path)
    _write(directory / "answers.jsonl", [_answer(f"u{i}") for i in range(4)])
    _write(directory / "sgm.jsonl", [{"uid": f"u{i}", "score": 0.5} for i in range(2)])
    _write(directory / "errors.jsonl", [{"uid": "u2", "stage": "sampling", "error": "x"}])  # u3 shares its question
    assert any("sgm" in flag and "3 of 4" in flag for flag in _check(tmp_path))


def test_unfinished_counts_fail_a_final_check(tmp_path):
    _processed(tmp_path)
    _write(_language_dir(tmp_path) / "answers.jsonl", [_answer("u0")])
    with pytest.raises(AssertionError, match="1, expected 4"):
        _check(tmp_path)


def test_unfinished_counts_are_flags_in_a_partial_check(tmp_path):
    _processed(tmp_path)
    directory = _language_dir(tmp_path)
    _write(directory / "answers.jsonl", [_answer("u0")])
    _write(directory / "sgm.jsonl", [{"uid": "u0", "score": 0.5}])
    flags = _check(tmp_path, partial=True)
    assert any("answers" in f and "1 of 4" in f for f in flags) and any("sgm" in f and "1 of 4" in f for f in flags)


def test_a_partial_check_still_fails_on_too_many_rows(tmp_path):
    _processed(tmp_path, n=4)
    directory = _language_dir(tmp_path)
    _write(directory / "answers.jsonl", [_answer(f"u{i}") for i in range(4)])
    _write(directory / "sgm.jsonl", [{"uid": f"u{i}", "score": 0.5} for i in range(2)])
    with pytest.raises(AssertionError, match="2, expected 1"):
        _check(tmp_path, partial=True, limit=1)


def test_a_partial_check_still_fails_on_bad_values(tmp_path):
    _processed(tmp_path)
    _write(_language_dir(tmp_path) / "answers.jsonl", [{**_answer("u0"), "label": 7}])
    with pytest.raises(AssertionError, match="label"):
        _check(tmp_path, partial=True)


def test_write_run_json_survives_a_value_json_cannot_store(tmp_path):
    shared.write_run_json(tmp_path, dtype=torch.float16)
    assert "float16" in json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))["dtype"]


def test_eta_counts_no_sampling_when_the_limit_is_zero():
    throughput = {"sampling": {"mtrex": 2.0}}
    assert run_phase3._sampling_seconds(throughput, {"mtrex": 100}, {"mtrex": 1}, {"mtrex": 0}) == 0
    assert run_phase3._sampling_seconds(throughput, {"mtrex": 100}, {"mtrex": 1}, {"mtrex": None}) == 200


def test_smoke_accepts_empty_predictions_when_every_group_was_one_class(tmp_path, monkeypatch):
    root = tmp_path
    directory = root / "mtrex" / "bn"
    _write(directory / "answers.jsonl", [{**_answer("u0"), "label": 0}, {**_answer("u1"), "label": 1}])
    for name in ("sgm.jsonl", "sem.jsonl"):
        _write(directory / name, [{"uid": "u0"}])
    (directory / "run.json").write_text("{}", encoding="utf-8")
    (directory / "mam_preds_seed42.jsonl").write_text("", encoding="utf-8")
    _write(directory / "errors.jsonl", [{"uid": "capitals", "stage": "classifier", "error": "ValueError: one class"}])
    for name in ("features.pkl", "samples.jsonl"):
        (root / "raw" / "mtrex" / "bn").mkdir(parents=True, exist_ok=True)
        (root / "raw" / "mtrex" / "bn" / name).write_text("x", encoding="utf-8")
    problems, warnings = run_phase3._smoke_layout_problems(root, {"seeds": [42]}, [("mtrex", "bn")])
    assert not problems and warnings


def test_verify_partial_flag_reaches_the_checker(monkeypatch):
    seen = {}

    def fake_run(command, cwd=None, env=None):
        seen["env"] = env
        return type("Done", (), {"returncode": 0})()

    monkeypatch.setattr(run_phase3.subprocess, "run", fake_run)
    args = run_phase3._parse(["verify", "--partial"])
    assert run_phase3.run_verify(args, {}, None, None) == 0
    assert seen["env"]["G6_PARTIAL"] == "1"
