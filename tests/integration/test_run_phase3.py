"""run_phase3 stages against the tiny model: run.json fields, accuracy report, resume, round-robin order."""
import json

import pytest

from scripts import run_phase3 as rp
from src.detectors import selfcheckGPT_g6 as sgm
from src.detectors import shared

CFG = {"mam_limit": 3, "sampling_limit": {"mtrex": 2, "xfactr": 0, "psiloqa": 0}, "psiloqa_max_new_tokens": 6,
       "num_samples": 2, "sample_chunk": 2, "sample_step": 1, "seeds": [42],
       "languages": {"mtrex": ["en", "hi"], "xfactr": ["en"], "psiloqa": ["en"]}}


@pytest.fixture(autouse=True)
def results_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(shared, "RESULTS", tmp_path)


@pytest.fixture
def prompts():
    return shared.load_prompts()


def test_mam_stage_writes_run_json_with_the_generation_config_and_prints_the_paper_comparison(
        tmp_path, tiny_model, tiny_tokenizer, prompts, capsys):
    rp.stage_mam(CFG, tiny_model, tiny_tokenizer, prompts)

    info = json.loads((tmp_path / "mtrex" / "en" / "run.json").read_text(encoding="utf-8"))
    assert info["items"] == 3 and info["seeds"] == [42] and info["gpu"]
    assert info["submodule_commit"].startswith("644d655")
    assert isinstance(info["generation_config"], dict) and info["started"] <= info["ended"]
    assert "mTREx en capitals: accuracy" in capsys.readouterr().out


def test_a_second_mam_pass_redoes_nothing_and_keeps_the_first_start(tmp_path, tiny_model, tiny_tokenizer, prompts):
    rp.stage_mam(CFG, tiny_model, tiny_tokenizer, prompts)
    directory = tmp_path / "mtrex" / "hi"
    first = shared.read_results(directory / "answers.jsonl")
    started = json.loads((directory / "run.json").read_text(encoding="utf-8"))["started"]

    rp.stage_mam(CFG, tiny_model, tiny_tokenizer, prompts)

    assert shared.read_results(directory / "answers.jsonl") == first
    assert json.loads((directory / "run.json").read_text(encoding="utf-8"))["started"] == started


def test_sampling_alternates_languages_one_step_at_a_time(tiny_model, tiny_tokenizer, prompts, fake_encoder,
                                                         monkeypatch):
    rp.stage_mam(CFG, tiny_model, tiny_tokenizer, prompts)
    real, turns = sgm.sample_schema, []

    def record_turn(records, *args, **kwargs):
        turns.append((records[0].language, len(records)))
        return real(records, *args, **kwargs)

    monkeypatch.setattr(sgm, "sample_schema", record_turn)

    rp.stage_sampling(CFG, tiny_model, tiny_tokenizer, prompts, lambda device: fake_encoder)

    assert turns == [("en", 1), ("hi", 1), ("en", 1), ("hi", 1)]


def test_classifier_stage_logs_an_error_when_a_language_has_no_features(tmp_path):
    rp.stage_classifiers({**CFG, "languages": {"mtrex": ["bn"], "xfactr": ["en"], "psiloqa": ["en"]}})

    errors = shared.read_results(tmp_path / "mtrex" / "bn" / "errors.jsonl")
    assert [e["stage"] for e in errors] == ["classifier"]
