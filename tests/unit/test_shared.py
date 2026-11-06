"""Unit tests for src/detectors/shared.py."""
import json

import pytest

from src.common.io import load_config
from src.common.schema import Record
from src.detectors import shared


def test_mtrex_messages_keep_authors_trailing_whitespace_and_prefix(mtrex_records, fake_prompts):
    messages = shared.build_messages(mtrex_records[0], fake_prompts)
    assert messages == [
        {"role": "system", "content": "Answer in English. \n"},
        {"role": "user", "content": "Question: What is the capital of Somalia? \n"},
        {"role": "assistant", "content": "Answer (in English): "},
    ]


def test_xfactr_messages_put_the_sentence_with_blank_in_user_turn(xfactr_records, fake_prompts):
    messages = shared.build_messages(xfactr_records[0], fake_prompts)
    assert messages[1]["content"] == "Sentence: Spain speaks ___ ."
    assert messages[2] == {"role": "assistant", "content": "Answer: "}


def test_psiloqa_messages_hold_passage_and_question_and_never_the_given_answer(psiloqa_records, fake_prompts):
    messages = shared.build_messages(psiloqa_records[0], fake_prompts)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Short passage." in messages[1]["content"] and "Who wrote Hamlet?" in messages[1]["content"]
    assert "Shakespeare" not in json.dumps(messages)


def test_build_messages_fails_loudly_for_a_language_without_prompts(xfactr_records):
    with pytest.raises(KeyError):
        shared.build_messages(xfactr_records[0], {"xfactr": {}})


def test_chat_text_ends_with_generation_prompt(mtrex_records, fake_prompts, tiny_tokenizer):
    text = shared.chat_text(mtrex_records[0], tiny_tokenizer, fake_prompts)
    assert text.endswith("<|start|>assistant<|sep|>")


def all_language_codes():
    data = load_config("data")
    xfactr = [l["code"] for l in load_config("languages")["languages"] if l["chosen"]]
    return sorted({*data["mtrex"]["languages"].values(), *xfactr, *data["psiloqa"]["languages"], "en"})


@pytest.mark.parametrize("code", all_language_codes())
def test_every_chosen_language_has_a_working_unigram_tokenizer(code):
    tokens = [t.text for t in shared.blank_spacy(code)("abc def")]
    assert tokens


def test_blank_spacy_rejects_an_unknown_language():
    with pytest.raises(KeyError):
        shared.blank_spacy("qq")


def test_append_jsonl_then_done_uids_round_trip_and_keep_unicode(tmp_path):
    path = tmp_path / "a" / "answers.jsonl"
    shared.append_jsonl(path, {"uid": "x", "answer": "दिल्ली"})
    shared.append_jsonl(path, {"uid": "y"})
    assert shared.done_uids(path) == {"x", "y"}
    assert "दिल्ली" in path.read_text(encoding="utf-8")


def test_done_uids_is_empty_for_a_missing_file(tmp_path):
    assert shared.done_uids(tmp_path / "none.jsonl") == set()


def test_read_results_drops_a_half_written_last_line(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text('{"uid": "x"}\n{"uid": "y', encoding="utf-8")
    assert [r["uid"] for r in shared.read_results(path)] == ["x"]
    assert shared.done_uids(path) == {"x"}


def test_append_after_a_half_written_line_does_not_merge_records(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text('{"uid": "x"}\n{"uid": "y', encoding="utf-8")
    shared.repair_jsonl(path)
    shared.append_jsonl(path, {"uid": "z"})
    assert [r["uid"] for r in shared.read_results(path)] == ["x", "z"]


def test_read_results_raises_on_corruption_before_the_last_line(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text('{"uid": "x"\n{"uid": "y"}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        shared.read_results(path)


def test_features_round_trip_and_later_duplicate_uid_wins(tmp_path):
    path = tmp_path / "raw" / "features.pkl"
    shared.append_feature(path, {"uid": "a", "x": 1})
    shared.append_feature(path, {"uid": "b", "x": 2})
    shared.append_feature(path, {"uid": "a", "x": 3})
    assert [(f["uid"], f["x"]) for f in shared.read_features(path)] == [("a", 3), ("b", 2)]


def test_repair_features_cuts_a_half_written_record_so_appends_stay_readable(tmp_path):
    path = tmp_path / "features.pkl"
    shared.append_feature(path, {"uid": "a"})
    with open(path, "ab") as f:
        f.write(b"\x80\x04\x95half")
    shared.repair_features(path)
    shared.append_feature(path, {"uid": "b"})
    assert [f["uid"] for f in shared.read_features(path)] == ["a", "b"]


def test_repair_features_survives_a_tail_claiming_a_huge_frame(tmp_path):
    path = tmp_path / "features.pkl"
    shared.append_feature(path, {"uid": "a"})
    with open(path, "ab") as f:
        f.write(b"\x80\x04\x95" + (2**62).to_bytes(8, "little"))
    shared.repair_features(path)
    assert [f["uid"] for f in shared.read_features(path)] == ["a"]


def test_read_features_is_empty_for_a_missing_file(tmp_path):
    assert shared.read_features(tmp_path / "none.pkl") == []


def test_log_error_records_uid_stage_and_message(tmp_path):
    shared.log_error(tmp_path, "u1", "mam", ValueError("boom"))
    row = shared.read_results(tmp_path / "errors.jsonl")[0]
    assert row["uid"] == "u1" and row["stage"] == "mam" and "boom" in row["error"]


def test_results_dir_separates_raw_from_tracked(tmp_path, monkeypatch):
    monkeypatch.setattr(shared, "RESULTS", tmp_path)
    assert shared.results_dir("mtrex", "en") == tmp_path / "mtrex" / "en"
    assert shared.results_dir("mtrex", "en", raw=True) == tmp_path / "raw" / "mtrex" / "en"
    assert (tmp_path / "raw" / "mtrex" / "en").is_dir()


def test_write_run_json_merges_fields(tmp_path):
    shared.write_run_json(tmp_path, model="m")
    shared.write_run_json(tmp_path, items=3)
    assert json.loads((tmp_path / "run.json").read_text(encoding="utf-8")) == {"model": "m", "items": 3}


def test_item_seed_is_stable_and_differs_by_uid():
    assert shared.item_seed("a") == shared.item_seed("a") != shared.item_seed("b")
    assert 0 <= shared.item_seed("a") < 2**32


def test_take_balanced_cycles_through_relations_in_file_order():
    records = [Record(uid=f"{rel}{i}", dataset="xfactr", language="en", prompt="p ___", relation=rel, gold=["g"])
               for rel in "ab" for i in range(3)]
    assert [r.uid for r in shared.take_balanced(records, 4)] == ["a0", "b0", "a1", "b1"]
    assert len(shared.take_balanced(records, None)) == 6


def test_take_balanced_uses_the_generator_for_psiloqa(psiloqa_records):
    psiloqa_records[0].meta["model"] = "other"
    assert {r.meta["model"] for r in shared.take_balanced(psiloqa_records, 2)} == {"other", "m"}
