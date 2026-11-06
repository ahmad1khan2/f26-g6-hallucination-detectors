"""MAM feature extraction on the tiny offline model (schema records, resume, hooks, PsiloQA teacher forcing)."""
import copy
import json
import pickle
from dataclasses import replace

import numpy as np
import pytest
import torch

from src.detectors import ModelArtifacts_g6 as mam
from src.detectors import ModelArtifacts_multiple_gen_tokens_g6 as mam_multi
from src.detectors import shared


@pytest.fixture
def dirs(tmp_path):
    return tmp_path / "out", tmp_path / "raw"


def _run(records, model, tokenizer, prompts, dirs):
    mam.run_schema(records, model, tokenizer, prompts, *dirs, max_new_tokens=8, min_new_tokens=5)


def _tokens(tokenizer, text):
    return tokenizer(text, return_tensors="pt")["input_ids"]


def _capture(module, model, ids):
    """Independent forward pass; returns {layer index: (fc, attn)} arrays of shape (seq, dim)."""
    handles = module.register_g6_hooks(model)
    module.FULLY_CONNECTED_HIDDEN_LAYERS.clear()
    module.ATTENTION_HIDDEN_LAYERS.clear()
    try:
        with torch.no_grad():
            model(ids)
        return ({k: v[-1] for k, v in module.FULLY_CONNECTED_HIDDEN_LAYERS.items()},
                {k: v[-1] for k, v in module.ATTENTION_HIDDEN_LAYERS.items()})
    finally:
        for handle in handles:
            handle.remove()


def test_run_schema_writes_both_files_with_the_agreed_rows(dirs, tiny_model, tiny_tokenizer, fake_prompts,
                                                           mtrex_records):
    _run(mtrex_records, tiny_model, tiny_tokenizer, fake_prompts, dirs)
    answers = shared.read_results(dirs[0] / "answers.jsonl")
    features = shared.read_features(dirs[1] / "features.pkl")
    assert [a["uid"] for a in answers] == [r.uid for r in mtrex_records] == [f["uid"] for f in features]
    assert set(answers[0]) == {"uid", "answer", "label", "label_rule", "first_token_entropy"}
    assert answers[0]["label_rule"] == "matches" and isinstance(answers[0]["first_token_entropy"], float)
    assert set(features[0]) == {"uid", "relation", "label", "first_fully_connected", "first_attention"}
    assert features[0]["relation"] == "capitals" and features[0]["label"] == answers[0]["label"]


def test_feature_shapes_are_middle_and_last_layer_by_width(dirs, tiny_model, tiny_tokenizer, fake_prompts,
                                                           mtrex_records):
    _run(mtrex_records[:1], tiny_model, tiny_tokenizer, fake_prompts, dirs)
    feature = shared.read_features(dirs[1] / "features.pkl")[0]
    assert feature["first_attention"].shape == (2, tiny_model.config.hidden_size)  # (2, 32)
    # up_proj is the authors' "fully connected" hook point: its width is the MLP size, not the hidden size
    assert feature["first_fully_connected"].shape == (2, tiny_model.config.intermediate_size)


def test_features_equal_an_independent_pass_at_the_first_generated_position(dirs, tiny_model, tiny_tokenizer,
                                                                            fake_prompts, mtrex_records):
    record = mtrex_records[0]
    _run([record], tiny_model, tiny_tokenizer, fake_prompts, dirs)
    answer = shared.read_results(dirs[0] / "answers.jsonl")[0]
    feature = shared.read_features(dirs[1] / "features.pkl")[0]
    chat = shared.chat_text(record, tiny_tokenizer, fake_prompts)
    start = _tokens(tiny_tokenizer, chat).shape[1]
    # the stored answer is the decoded, newline-flattened greedy output; regenerate it the same way
    ids = tiny_model.generate(_tokens(tiny_tokenizer, chat), do_sample=False, min_new_tokens=5, max_new_tokens=8,
                              pad_token_id=tiny_tokenizer.pad_token_id, eos_token_id=tiny_tokenizer.eos_token_id)
    fc, attn = _capture(mam, tiny_model, ids)
    assert tiny_tokenizer.decode(ids[0][start:], skip_special_tokens=True).replace("\n", " ").strip() == answer["answer"]
    np.testing.assert_allclose(feature["first_attention"], np.stack([attn["model.layers.0.self_attn.o_proj"][start],
                                                                     attn["model.layers.1.self_attn.o_proj"][start]]),
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(feature["first_fully_connected"],
                               np.stack([fc["model.layers.0.mlp.up_proj"][start], fc["model.layers.1.mlp.up_proj"][start]]),
                               rtol=1e-5, atol=1e-6)
    with torch.no_grad():
        logits = tiny_model(ids).logits[0, start]
    assert answer["first_token_entropy"] == pytest.approx(mam.token_entropy(logits))


def test_hooks_are_removed_after_a_run_even_when_it_fails(dirs, tiny_model, tiny_tokenizer, fake_prompts,
                                                          mtrex_records, monkeypatch):
    _run(mtrex_records[:1], tiny_model, tiny_tokenizer, fake_prompts, dirs)
    assert not any(m._forward_hooks for m in tiny_model.modules())

    def interrupt(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(mam, "generate_response", interrupt)
    with pytest.raises(KeyboardInterrupt):
        _run(mtrex_records[1:], tiny_model, tiny_tokenizer, fake_prompts, dirs)
    assert not any(m._forward_hooks for m in tiny_model.modules())


def test_label_one_means_hallucinated_and_comes_from_matches(dirs, tiny_model, tiny_tokenizer, fake_prompts,
                                                             mtrex_records, monkeypatch):
    real = mam.generate_response
    said = iter(["It is Mogadishu.", "Berlin", "LIMA, of course"])  # gold: mogadishu, paris, lima

    def scripted(**kwargs):
        _, logits, start = real(**kwargs)
        return next(said), logits, start

    monkeypatch.setattr(mam, "generate_response", scripted)
    _run(mtrex_records, tiny_model, tiny_tokenizer, fake_prompts, dirs)
    assert [a["label"] for a in shared.read_results(dirs[0] / "answers.jsonl")] == [0, 1, 0]
    assert [f["label"] for f in shared.read_features(dirs[1] / "features.pkl")] == [0, 1, 0]


def test_a_failing_item_goes_to_errors_and_the_rest_finish(dirs, tiny_model, tiny_tokenizer, fake_prompts,
                                                           mtrex_records, monkeypatch):
    real = mam.generate_response
    calls = iter(range(3))

    def flaky(**kwargs):
        if next(calls) == 1:
            raise ValueError("boom")
        return real(**kwargs)

    monkeypatch.setattr(mam, "generate_response", flaky)
    _run(mtrex_records, tiny_model, tiny_tokenizer, fake_prompts, dirs)
    errors = shared.read_results(dirs[0] / "errors.jsonl")
    assert errors == [{"uid": mtrex_records[1].uid, "stage": "mam", "error": "ValueError: boom"}]
    assert [a["uid"] for a in shared.read_results(dirs[0] / "answers.jsonl")] == [mtrex_records[0].uid,
                                                                                  mtrex_records[2].uid]


def test_interrupted_run_resumes_without_duplicates(dirs, tiny_model, tiny_tokenizer, fake_prompts, mtrex_records):
    _run(mtrex_records[:2], tiny_model, tiny_tokenizer, fake_prompts, dirs)
    with open(dirs[0] / "answers.jsonl", "ab") as f:
        f.write(b'{"uid": "half-writ')
    with open(dirs[1] / "features.pkl", "ab") as f:
        f.write(pickle.dumps({"uid": "half-writ", "x": list(range(50))})[:-7])  # a record cut mid-write
    _run(mtrex_records, tiny_model, tiny_tokenizer, fake_prompts, dirs)
    uids = [r.uid for r in mtrex_records]
    assert [a["uid"] for a in shared.read_results(dirs[0] / "answers.jsonl")] == uids
    assert [f["uid"] for f in shared.read_features(dirs[1] / "features.pkl")] == uids
    lines = (dirs[0] / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and all(json.loads(line) for line in lines)


def _run_given(records, model, tokenizer, prompts, dirs, **kwargs):
    mam_multi.run_schema_given_answer(records, model, tokenizer, prompts, *dirs, **kwargs)


def _given_expected(record, model, tokenizer, prompts, avg):
    chat = shared.chat_text(record, tokenizer, prompts)
    start = _tokens(tokenizer, chat).shape[1]
    ids = _tokens(tokenizer, chat + record.answer)
    fc, attn = _capture(mam_multi, model, ids)
    end = min(start + avg, ids.shape[1])
    mean = lambda d, name: np.stack([d[f"model.layers.{i}.{name}"][start:end].mean(axis=0) for i in (0, 1)])
    return mean(fc, "mlp.up_proj"), mean(attn, "self_attn.o_proj"), end - start


def test_psiloqa_rows_and_average_over_a_short_answer(dirs, tiny_model, tiny_tokenizer, fake_prompts,
                                                      psiloqa_records):
    short = replace(psiloqa_records[2], answer="Five")
    _run_given([short], tiny_model, tiny_tokenizer, fake_prompts, dirs)
    row = shared.read_results(dirs[0] / "answers.jsonl")[0]
    feature = shared.read_features(dirs[1] / "features.pkl")[0]
    assert row["answer"] == "Five" and row["label"] == short.label and row["label_rule"] == "spans"
    assert set(feature) == {"uid", "relation", "label", "first_fully_connected", "first_attention",
                            "token_pos_fully_connected_avg", "token_pos_attention_avg"}
    fc, attn, count = _given_expected(short, tiny_model, tiny_tokenizer, fake_prompts, 10)
    assert count == 4  # fewer than 10 answer tokens: average over just those
    np.testing.assert_allclose(feature["token_pos_fully_connected_avg"], fc, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(feature["token_pos_attention_avg"], attn, rtol=1e-5, atol=1e-6)
    assert feature["token_pos_attention_avg"].shape == (2, tiny_model.config.hidden_size)


def test_psiloqa_average_is_capped_at_ten_answer_tokens(dirs, tiny_model, tiny_tokenizer, fake_prompts,
                                                        psiloqa_records):
    record = psiloqa_records[0]
    long = replace(record, answer="Shakespeare wrote Hamlet in about 1600.")
    _run_given([long], tiny_model, tiny_tokenizer, fake_prompts, dirs)
    feature = shared.read_features(dirs[1] / "features.pkl")[0]
    fc, attn, count = _given_expected(long, tiny_model, tiny_tokenizer, fake_prompts, 10)
    assert count == 10
    np.testing.assert_allclose(feature["token_pos_fully_connected_avg"], fc, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(feature["token_pos_attention_avg"], attn, rtol=1e-5, atol=1e-6)
    # first-token features are the answer's first position, not the average
    assert not np.allclose(feature["first_attention"], feature["token_pos_attention_avg"])


def test_the_given_answer_is_really_used_in_the_forward_pass(dirs, tiny_model, tiny_tokenizer, fake_prompts,
                                                             psiloqa_records, tmp_path):
    record = psiloqa_records[0]
    other = replace(record, uid="psiloqa-en-other", answer="Zzzzzz qqqq xxxx.")
    _run_given([record, other], tiny_model, tiny_tokenizer, fake_prompts, dirs)
    a, b = shared.read_features(dirs[1] / "features.pkl")
    assert not np.allclose(a["token_pos_attention_avg"], b["token_pos_attention_avg"])
    rows = shared.read_results(dirs[0] / "answers.jsonl")
    assert rows[0]["first_token_entropy"] != rows[1]["first_token_entropy"]


def test_an_empty_answer_is_logged_not_written(dirs, tiny_model, tiny_tokenizer, fake_prompts, psiloqa_records):
    empty = copy.copy(psiloqa_records[0])
    object.__setattr__(empty, "answer", "")  # the schema refuses empty answers; skip its validation
    _run_given([empty, psiloqa_records[1]], tiny_model, tiny_tokenizer, fake_prompts, dirs)
    errors = shared.read_results(dirs[0] / "errors.jsonl")
    assert [e["uid"] for e in errors] == [empty.uid] and errors[0]["stage"] == "mam"
    assert [a["uid"] for a in shared.read_results(dirs[0] / "answers.jsonl")] == [psiloqa_records[1].uid]


def test_psiloqa_run_resumes_without_duplicates(dirs, tiny_model, tiny_tokenizer, fake_prompts, psiloqa_records):
    _run_given(psiloqa_records[:2], tiny_model, tiny_tokenizer, fake_prompts, dirs)
    with open(dirs[1] / "features.pkl", "ab") as f:
        f.write(pickle.dumps({"uid": "half-writ", "x": list(range(50))})[:-7])  # a record cut mid-write
    _run_given(psiloqa_records, tiny_model, tiny_tokenizer, fake_prompts, dirs)
    uids = [r.uid for r in psiloqa_records]
    assert [a["uid"] for a in shared.read_results(dirs[0] / "answers.jsonl")] == uids
    assert [f["uid"] for f in shared.read_features(dirs[1] / "features.pkl")] == uids
