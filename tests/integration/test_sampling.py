"""Sampling with the tiny random Llama: row shape, chunking, reproducibility, equality with the authors' path."""
import torch

from src.detectors import selfcheckGPT_g6 as sgm
from src.detectors import shared

NEW_TOKENS = 6


def sample(records, model, tokenizer, prompts, raw_dir, **kwargs):
    sgm.sample_schema(records, model, tokenizer, prompts, raw_dir, max_new_tokens=NEW_TOKENS, **kwargs)
    return shared.read_results(raw_dir / "samples.jsonl")


def test_row_has_twenty_samples_and_a_log_lik_list_per_sample(tmp_path, tiny_model, tiny_tokenizer, fake_prompts,
                                                              mtrex_records):
    (row, *_) = sample(mtrex_records[:1], tiny_model, tiny_tokenizer, fake_prompts, tmp_path)
    assert row["uid"] == mtrex_records[0].uid and row["uids"] == [row["uid"]]
    assert len(row["samples"]) == 20 and len(row["log_liks"]) == 20
    assert all(isinstance(s, str) for s in row["samples"])
    assert all(1 <= len(ll) <= NEW_TOKENS and all(isinstance(x, float) for x in ll) for ll in row["log_liks"])


def test_chunked_sampling_still_gives_twenty_samples(tmp_path, tiny_model, tiny_tokenizer, fake_prompts,
                                                     xfactr_records):
    (row,) = sample(xfactr_records[:1], tiny_model, tiny_tokenizer, fake_prompts, tmp_path, chunk=7)
    assert len(row["samples"]) == 20 and len(row["log_liks"]) == 20


def test_same_item_seed_reproduces_the_samples(tmp_path, tiny_model, tiny_tokenizer, fake_prompts, mtrex_records):
    first = sample(mtrex_records[:2], tiny_model, tiny_tokenizer, fake_prompts, tmp_path / "a")
    second = sample(mtrex_records[:2], tiny_model, tiny_tokenizer, fake_prompts, tmp_path / "b")
    assert first == second


def test_unchunked_path_equals_the_authors_generate_responses(tmp_path, tiny_model, tiny_tokenizer, fake_prompts,
                                                               mtrex_records):
    record = mtrex_records[0]
    (row,) = sample([record], tiny_model, tiny_tokenizer, fake_prompts, tmp_path, chunk=20)
    torch.manual_seed(shared.item_seed(record.uid))
    _, _, authors, _ = sgm.generate_responses(
        question=shared.chat_text(record, tiny_tokenizer, fake_prompts), num_generations=20,
        max_new_tokens=NEW_TOKENS, tokenizer=tiny_tokenizer, model=tiny_model, device="cpu", lang_code="en")
    assert row["samples"] == [text for text, _ in authors]
    assert row["log_liks"] == [ll for _, ll in authors]


def test_psiloqa_gets_one_row_per_question_with_every_uid(tmp_path, tiny_model, tiny_tokenizer, fake_prompts,
                                                          psiloqa_records):
    rows = sample(psiloqa_records[:2], tiny_model, tiny_tokenizer, fake_prompts, tmp_path)
    assert len(rows) == 1
    assert rows[0]["uid"] == psiloqa_records[0].uid
    assert rows[0]["uids"] == [psiloqa_records[0].uid, psiloqa_records[1].uid]


def test_psiloqa_scoring_gives_one_sgm_row_per_uid(tmp_path, tiny_model, tiny_tokenizer, fake_prompts,
                                                   psiloqa_records):
    sample(psiloqa_records[:2], tiny_model, tiny_tokenizer, fake_prompts, tmp_path)
    for record in psiloqa_records[:2]:
        shared.append_jsonl(tmp_path / "answers.jsonl", {"uid": record.uid, "answer": record.answer})
    sgm.score_schema(tmp_path / "samples.jsonl", tmp_path / "answers.jsonl", tmp_path, "en")
    assert [r["uid"] for r in shared.read_results(tmp_path / "sgm.jsonl")] == [r.uid for r in psiloqa_records[:2]]


def test_resume_adds_no_duplicates_and_repairs_a_cut_tail(tmp_path, tiny_model, tiny_tokenizer, fake_prompts,
                                                          mtrex_records):
    sample(mtrex_records[:2], tiny_model, tiny_tokenizer, fake_prompts, tmp_path)
    with open(tmp_path / "samples.jsonl", "a", encoding="utf-8") as f:
        f.write('{"uid": "cut')
    rows = sample(mtrex_records, tiny_model, tiny_tokenizer, fake_prompts, tmp_path)
    assert [r["uid"] for r in rows] == [r.uid for r in mtrex_records]


def test_a_failing_item_is_logged_and_the_run_goes_on(tmp_path, tiny_model, tiny_tokenizer, fake_prompts,
                                                      mtrex_records, monkeypatch):
    real = tiny_model.generate
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(tiny_model, "generate", flaky)
    rows = sample(mtrex_records[:2], tiny_model, tiny_tokenizer, fake_prompts, tmp_path, out_dir=tmp_path / "out")
    assert [r["uid"] for r in rows] == [mtrex_records[1].uid]
    (error,) = shared.read_results(tmp_path / "out" / "errors.jsonl")
    assert error["uid"] == mtrex_records[0].uid and error["stage"] == "sampling" and "boom" in error["error"]
