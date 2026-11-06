"""Unit tests for the SelfCheckGPT and Semantic Entropy scorers (no model needed) and configs/prompts.yaml."""
import inspect
import json
import math

import pytest
import torch

from src.common.io import load_config
from src.common.schema import Record
from src.detectors import selfcheckGPT_g6 as sgm
from src.detectors import semanticEntropy_g6 as sem
from src.detectors import shared

LANGUAGES = "en de hi bn ur ja zh vi tl sw yo ilo fi ar".split()


def write_rows(path, rows):
    for row in rows:
        shared.append_jsonl(path, row)


def sample_row(uid, samples, uids=None, log_liks=None):
    return {"uid": uid, "uids": uids or [uid], "samples": samples,
            "log_liks": log_liks or [[-1.0] for _ in samples]}


class StubEncoder:
    """'a' and 'b' are orthogonal, so each distinct string is its own cluster."""

    def encode(self, texts, convert_to_tensor=True):
        return torch.tensor([[1.0, 0.0] if t == "a" else [0.0, 1.0] for t in texts])


def test_unigram_score_matches_hand_computation(tmp_path):
    # counts over passage + samples: a=3, b=2, c=1 of 6 tokens; passage "a b" -> avg nll (ln2 + ln3)/2
    write_rows(tmp_path / "samples.jsonl", [sample_row("u1", ["a b", "a c"])])
    write_rows(tmp_path / "answers.jsonl", [{"uid": "u1", "answer": "a b"}])
    sgm.score_schema(tmp_path / "samples.jsonl", tmp_path / "answers.jsonl", tmp_path, "en")
    (row,) = shared.read_results(tmp_path / "sgm.jsonl")
    expected = 1.0 - math.exp(-(math.log(2) + math.log(3)) / 2)
    assert row["uid"] == "u1" and row["score"] == pytest.approx(expected)


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_language_scores_without_crashing(tmp_path, language):
    write_rows(tmp_path / "samples.jsonl", [sample_row("u1", ["東京 です", "hello world", "東京"])])
    write_rows(tmp_path / "answers.jsonl", [{"uid": "u1", "answer": "東京 です"}])
    sgm.score_schema(tmp_path / "samples.jsonl", tmp_path / "answers.jsonl", tmp_path, language)
    (row,) = shared.read_results(tmp_path / "sgm.jsonl")
    assert math.isfinite(row["score"]) and not (tmp_path / "errors.jsonl").exists()


@pytest.mark.parametrize("answer", ["", "   "])
def test_empty_answer_goes_to_errors_never_to_output(tmp_path, answer):
    write_rows(tmp_path / "samples.jsonl", [sample_row("u1", ["a", "b"])])
    write_rows(tmp_path / "answers.jsonl", [{"uid": "u1", "answer": answer}])
    sgm.score_schema(tmp_path / "samples.jsonl", tmp_path / "answers.jsonl", tmp_path, "en")
    assert shared.read_results(tmp_path / "sgm.jsonl") == []
    (error,) = shared.read_results(tmp_path / "errors.jsonl")
    assert error["uid"] == "u1" and error["stage"] == "sgm"


def test_uid_without_answer_is_skipped_silently_then_scored_on_resume(tmp_path):
    write_rows(tmp_path / "samples.jsonl", [sample_row("u1", ["a"]), sample_row("u2", ["b"])])
    write_rows(tmp_path / "answers.jsonl", [{"uid": "u1", "answer": "a"}])
    args = (tmp_path / "samples.jsonl", tmp_path / "answers.jsonl", tmp_path, "en")
    sgm.score_schema(*args)
    assert [r["uid"] for r in shared.read_results(tmp_path / "sgm.jsonl")] == ["u1"]
    assert not (tmp_path / "errors.jsonl").exists()
    write_rows(tmp_path / "answers.jsonl", [{"uid": "u2", "answer": "b"}])
    sgm.score_schema(*args)
    assert [r["uid"] for r in shared.read_results(tmp_path / "sgm.jsonl")] == ["u1", "u2"]


def test_sgm_resume_does_not_duplicate_and_repairs_a_cut_tail(tmp_path):
    write_rows(tmp_path / "samples.jsonl", [sample_row("u1", ["a"]), sample_row("u2", ["b"])])
    write_rows(tmp_path / "answers.jsonl", [{"uid": "u1", "answer": "a"}, {"uid": "u2", "answer": "b"}])
    args = (tmp_path / "samples.jsonl", tmp_path / "answers.jsonl", tmp_path, "en")
    sgm.score_schema(*args)
    with open(tmp_path / "sgm.jsonl", "a", encoding="utf-8") as f:
        f.write('{"uid": "u3", "sco')
    sgm.score_schema(*args)
    assert [r["uid"] for r in shared.read_results(tmp_path / "sgm.jsonl")] == ["u1", "u2"]


def test_question_row_scores_each_of_its_uids_against_its_own_answer(tmp_path):
    write_rows(tmp_path / "samples.jsonl", [sample_row("q1", ["a b", "a"], uids=["q1", "q2"])])
    write_rows(tmp_path / "answers.jsonl", [{"uid": "q1", "answer": "a b"}, {"uid": "q2", "answer": "a"}])
    sgm.score_schema(tmp_path / "samples.jsonl", tmp_path / "answers.jsonl", tmp_path, "en")
    rows = shared.read_results(tmp_path / "sgm.jsonl")
    assert [r["uid"] for r in rows] == ["q1", "q2"] and rows[0]["score"] != rows[1]["score"]


def test_unigram_model_uses_the_shared_tokenizer_for_languages_spacy_lacks():
    model = sgm.UnigramModel("sw")
    assert model.nlp is shared.blank_spacy("sw")


# ---- Semantic Entropy ----

def test_entropies_match_hand_computed_values(tmp_path):
    # clusters: {a, a, a} and {b}; per-sample log-likelihoods -1,-1,-1 and -2
    write_rows(tmp_path / "samples.jsonl",
               [sample_row("u1", ["a", "a", "b", "a"], log_liks=[[-1.0], [-1.0], [-2.0], [-1.0]])])
    sem.score_schema(tmp_path / "samples.jsonl", tmp_path, StubEncoder())
    (row,) = shared.read_results(tmp_path / "sem.jsonl")
    e = row["entropies"]
    assert row["uid"] == "u1" and row["uids"] == ["u1"] and row["level"] == "item"
    assert len(e) == 11 and all(isinstance(v, float) for v in e.values())
    assert e["cluster_assignment_entropy"] == pytest.approx(-(0.75 * math.log(0.75) + 0.25 * math.log(0.25)))
    assert e["regular_entropy"] == pytest.approx(1.25) and e["regular_entropy_sum"] == pytest.approx(1.25)
    # 'sum' aggregation subtracts 5.0 in the authors' code
    assert e["semantic_entropy"] == pytest.approx(-((math.log(3) - 1 - 5) + (-2 - 5)) / 2)
    assert e["semantic_entropy_cmean"] == pytest.approx(1.5)
    total = 3 * math.exp(-1) + math.exp(-2)
    p0, p1 = 3 * math.exp(-1) / total, math.exp(-2) / total
    assert e["semantic_entropy_sum-normalized"] == pytest.approx(-(math.log(p0) + math.log(p1)) / 2)
    assert e["semantic_entropy_sum-normalized-rao"] == pytest.approx(-(p0 * math.log(p0) + p1 * math.log(p1)))


def test_entropies_of_multi_token_log_liks_use_mean_and_sum(tmp_path):
    write_rows(tmp_path / "samples.jsonl",
               [sample_row("u1", ["a", "b"], log_liks=[[-1.0, -3.0], [-2.0, -2.0]])])
    sem.score_schema(tmp_path / "samples.jsonl", tmp_path, StubEncoder())
    e = shared.read_results(tmp_path / "sem.jsonl")[0]["entropies"]
    assert e["regular_entropy"] == pytest.approx(2.0)       # means -2, -2
    assert e["regular_entropy_sum"] == pytest.approx(4.0)   # sums -4, -4


def test_sem_level_is_passed_through_and_resume_skips_done_uids(tmp_path):
    write_rows(tmp_path / "samples.jsonl", [sample_row("q1", ["a", "b"], uids=["q1", "q2"])])
    for _ in range(2):
        sem.score_schema(tmp_path / "samples.jsonl", tmp_path, StubEncoder(), level="question")
    (row,) = shared.read_results(tmp_path / "sem.jsonl")
    assert row["level"] == "question" and row["uids"] == ["q1", "q2"]


def test_non_finite_entropy_goes_to_errors_not_to_output(tmp_path):
    write_rows(tmp_path / "samples.jsonl", [sample_row("u1", ["a", "b"], log_liks=[[float("-inf")], [-1.0]])])
    sem.score_schema(tmp_path / "samples.jsonl", tmp_path, StubEncoder())
    assert shared.read_results(tmp_path / "sem.jsonl") == []
    assert shared.read_results(tmp_path / "errors.jsonl")[0]["stage"] == "sem"


def test_sgm_and_sem_read_the_same_samples_file(tmp_path):
    write_rows(tmp_path / "samples.jsonl", [sample_row("u1", ["a", "b", "a"]), sample_row("u2", ["b", "b"])])
    write_rows(tmp_path / "answers.jsonl", [{"uid": "u1", "answer": "a"}, {"uid": "u2", "answer": "b"}])
    sgm.score_schema(tmp_path / "samples.jsonl", tmp_path / "answers.jsonl", tmp_path, "en")
    sem.score_schema(tmp_path / "samples.jsonl", tmp_path, StubEncoder())
    assert [r["uid"] for r in shared.read_results(tmp_path / "sgm.jsonl")] == \
           [r["uid"] for r in shared.read_results(tmp_path / "sem.jsonl")] == ["u1", "u2"]


def test_load_sbert_asks_for_labse_on_the_given_device(monkeypatch):
    seen = {}
    monkeypatch.setattr(sem, "SentenceTransformer", lambda name, device=None: seen.update(name=name, device=device))
    sem.load_sbert("cpu")
    assert seen == {"name": "sentence-transformers/LaBSE", "device": "cpu"}


def test_urdu_code_added_to_the_semantic_entropy_language_table():
    source = inspect.getsource(sem.processing_dataset)
    assert '"urdu": "ur"' in source and "# G6:" in source


@pytest.mark.parametrize("function", [sgm.UnigramModel.__init__, sgm.NgramModel.__init__, sgm.sample_schema,
                                      sgm.score_schema, sem.score_schema, sem.compute_entropies,
                                      sem.processing_dataset])
def test_every_changed_or_added_function_carries_a_g6_mark(function):
    assert "# G6:" in inspect.getsource(function)


# ---- configs/prompts.yaml ----

def prompt_languages():
    xfactr = ["en"] + [lang["code"] for lang in load_config("languages")["languages"] if lang["chosen"]]
    return xfactr, load_config("data")["psiloqa"]["languages"]


def test_prompts_cover_every_language_and_start_unchecked():
    prompts = load_config("prompts")
    xfactr, psiloqa = prompt_languages()
    assert set(prompts["xfactr"]) == set(xfactr) and set(prompts["psiloqa"]) == set(psiloqa)
    assert set(prompts["checked_by"]["xfactr"]) == set(xfactr)
    assert set(prompts["checked_by"]["psiloqa"]) == set(psiloqa)
    assert {v for group in prompts["checked_by"].values() for v in group.values()} == {"pending"}


def test_xfactr_prompts_format_with_real_records_through_build_messages():
    prompts = shared.load_prompts()
    for language in prompt_languages()[0]:
        record = Record(uid=f"xfactr-{language}-P1-000000", dataset="xfactr", language=language,
                        prompt="SUBJECT ___ .", relation="P1", gold=["g"])
        system, user, asst = shared.build_messages(record, prompts)
        assert "SUBJECT ___ ." in user["content"]
        assert "___" in system["content"] and asst["content"].strip()
        assert "{" not in user["content"] + asst["content"] + system["content"]


def test_psiloqa_prompts_format_with_real_records_through_build_messages():
    prompts = shared.load_prompts()
    for language in prompt_languages()[1]:
        record = Record(uid=f"psiloqa-{language}-000000", dataset="psiloqa", language=language,
                        prompt="QUESTION?", answer="a", label=0, meta={"passage": "PASSAGE TEXT"})
        system, user = shared.build_messages(record, prompts)
        assert "PASSAGE TEXT" in user["content"] and "QUESTION?" in user["content"]
        assert "{" not in user["content"] + system["content"]
        assert "asst" not in prompts["psiloqa"][language]


def test_prompts_yaml_uses_only_the_documented_placeholders():
    prompts = load_config("prompts")
    allowed = {"xfactr": {"{sentence}"}, "psiloqa": {"{passage}", "{question}"}}
    for dataset, placeholders in allowed.items():
        for entry in prompts[dataset].values():
            text = "".join(entry.values())
            assert {p for p in ("{sentence}", "{passage}", "{question}") if p in text} <= placeholders
            assert text.replace("{sentence}", "").replace("{passage}", "").replace("{question}", "").count("{") == 0
