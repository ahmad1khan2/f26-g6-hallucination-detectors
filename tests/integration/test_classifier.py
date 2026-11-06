"""run_schema of both MAM classifier scripts, end to end on synthetic feature files (CPU, no model)."""
import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.detectors import ModelArtifacts_Classifier_g6 as by_first_token
from src.detectors import ModelArtifacts_Classifier_multiple_gen_tokens_g6 as by_answer_tokens
from src.detectors import shared

HIDDEN = 8
SEEDS = (42, 43, 44)


def feature(rng, uid, relation, label, signal_key):
    """Arrays are [middle layer, last layer]; only the last layer of `signal_key` carries the label."""
    def noise():
        return rng.normal(size=(2, HIDDEN)).astype(np.float32)
    row = {"uid": uid, "relation": relation, "label": label}
    for key in ("first_fully_connected", "first_attention", "token_pos_fully_connected_avg",
                "token_pos_attention_avg"):
        row[key] = noise()
    for key in ("first_fully_connected", "first_attention") if signal_key == "first" else \
            ("token_pos_fully_connected_avg", "token_pos_attention_avg"):
        row[key][1] += 3.0 if label else -3.0
    return row


def write_features(path, plan, signal_key="first", seed=0):
    """plan: {relation: labels}. Returns {uid: label}."""
    rng = np.random.default_rng(seed)
    labels = {}
    for relation, relation_labels in plan.items():
        for i, label in enumerate(relation_labels):
            uid = f"d-en-{relation}-{i:03d}"
            shared.append_feature(path, feature(rng, uid, relation, label, signal_key))
            labels[uid] = label
    return labels


def alternating(n):
    return [i % 2 for i in range(n)]


def rows(out_dir, seed):
    return shared.read_results(out_dir / f"mam_preds_seed{seed}.jsonl")


def test_each_seed_gets_a_file_of_valid_rows_for_both_artifacts(tmp_path):
    labels = write_features(tmp_path / "features.pkl", {"P1": alternating(50)})
    by_first_token.run_schema(tmp_path / "features.pkl", tmp_path / "out", seeds=SEEDS)
    for seed in SEEDS:
        got = rows(tmp_path / "out", seed)
        assert {r["artifact"] for r in got} == {"fully_connected", "attention"}
        assert all(r["split"] == "test" and 0 <= r["p_hallucinated"] <= 1 and r["uid"] in labels for r in got)
        assert len(got) == 2 * 10 and all(r["relation"] == "P1" for r in got)
    assert {r["uid"] for r in rows(tmp_path / "out", 42)} != {r["uid"] for r in rows(tmp_path / "out", 43)}


def test_predictions_separate_the_classes_using_the_last_layer(tmp_path):
    labels = write_features(tmp_path / "features.pkl", {"P1": alternating(60)})
    by_first_token.run_schema(tmp_path / "features.pkl", tmp_path / "out", seeds=(42,))
    for artifact in ("fully_connected", "attention"):
        got = [r for r in rows(tmp_path / "out", 42) if r["artifact"] == artifact]
        assert roc_auc_score([labels[r["uid"]] for r in got], [r["p_hallucinated"] for r in got]) > 0.9


def test_same_seed_gives_identical_predictions_on_a_rerun(tmp_path):
    write_features(tmp_path / "features.pkl", {"P1": alternating(40)})
    by_first_token.run_schema(tmp_path / "features.pkl", tmp_path / "a", seeds=(42,))
    by_first_token.run_schema(tmp_path / "features.pkl", tmp_path / "b", seeds=(42,))
    assert rows(tmp_path / "a", 42) == rows(tmp_path / "b", 42)


def test_by_relation_trains_each_relation_on_its_own_items(tmp_path):
    write_features(tmp_path / "features.pkl", {"A": alternating(40), "B": alternating(30)})
    by_first_token.run_schema(tmp_path / "features.pkl", tmp_path / "out", seeds=(42,), by_relation=True)
    got = rows(tmp_path / "out", 42)
    for relation, n_test in (("A", 8), ("B", 6)):
        mine = [r for r in got if r["relation"] == relation]
        assert len(mine) == 2 * n_test and all(f"-{relation}-" in r["uid"] for r in mine)


def test_one_class_test_split_still_saves_predictions(tmp_path):
    train_idx, test_idx = train_test_split(range(10), test_size=0.2, random_state=42)
    labels = [0] * 10
    labels[train_idx[0]] = 1  # the only positive is in training, so the 2 test items are one class
    write_features(tmp_path / "features.pkl", {"P1": labels})
    by_first_token.run_schema(tmp_path / "features.pkl", tmp_path / "out", seeds=(42,))
    assert len(rows(tmp_path / "out", 42)) == 2 * len(test_idx)
    assert not (tmp_path / "out" / "errors.jsonl").exists()


def test_one_class_training_group_is_logged_and_the_others_continue(tmp_path):
    write_features(tmp_path / "features.pkl", {"A": alternating(30), "B": [0] * 30})
    by_first_token.run_schema(tmp_path / "features.pkl", tmp_path / "out", seeds=(42,), by_relation=True)
    assert {r["relation"] for r in rows(tmp_path / "out", 42)} == {"A"}
    errors = shared.read_results(tmp_path / "out" / "errors.jsonl")
    assert [(e["uid"], e["stage"]) for e in errors] == [("B", "classifier")]


def test_group_too_small_to_split_is_logged_and_the_others_continue(tmp_path):
    write_features(tmp_path / "features.pkl", {"A": alternating(30), "B": [1]})
    by_first_token.run_schema(tmp_path / "features.pkl", tmp_path / "out", seeds=(42,), by_relation=True)
    assert {r["relation"] for r in rows(tmp_path / "out", 42)} == {"A"}
    assert [e["uid"] for e in shared.read_results(tmp_path / "out" / "errors.jsonl")] == ["B"]


def test_language_with_one_class_logs_under_the_folder_name_and_writes_no_file(tmp_path):
    write_features(tmp_path / "features.pkl", {"P1": [0] * 20})
    out = tmp_path / "xfactr" / "sw"
    by_first_token.run_schema(tmp_path / "features.pkl", out, seeds=(42,))
    assert [e["uid"] for e in shared.read_results(out / "errors.jsonl")] == ["sw"]
    assert rows(out, 42) == []


def test_existing_seed_file_is_skipped_and_others_are_written(tmp_path):
    write_features(tmp_path / "features.pkl", {"P1": alternating(40)})
    out = tmp_path / "out"
    out.mkdir()
    (out / "mam_preds_seed42.jsonl").write_text("done\n", encoding="utf-8")
    by_first_token.run_schema(tmp_path / "features.pkl", out, seeds=(42, 43))
    assert (out / "mam_preds_seed42.jsonl").read_text(encoding="utf-8") == "done\n"
    assert rows(out, 43) and not list(out.glob("*.tmp"))


class Stopped(BaseException):
    """Stands in for a kill or a pause: `except Exception` must not catch it."""


def stop_after(monkeypatch, blocks):
    """Make predict_group_g6 stop the run once `blocks` trainings have finished; returns the list of trainings."""
    real, trained = by_first_token.predict_group_g6, []

    def counted(items, key, artifact, seed, device, train):
        if len(trained) == blocks:
            raise Stopped
        trained.append((items[0]["relation"], artifact, seed))
        return real(items, key, artifact, seed, device, train)

    monkeypatch.setattr(by_first_token, "predict_group_g6", counted)
    return trained


def test_a_stopped_classifier_resumes_with_only_the_missing_trainings_and_the_same_output(tmp_path, monkeypatch):
    plan = {"A": alternating(40), "B": alternating(30)}
    write_features(tmp_path / "features.pkl", plan)
    by_first_token.run_schema(tmp_path / "features.pkl", tmp_path / "whole", seeds=(42,), by_relation=True)

    stop_after(monkeypatch, 3)
    with pytest.raises(Stopped):
        by_first_token.run_schema(tmp_path / "features.pkl", tmp_path / "out", seeds=(42,), by_relation=True)
    assert not (tmp_path / "out" / "mam_preds_seed42.jsonl").exists()

    monkeypatch.undo()
    trained = stop_after(monkeypatch, 99)
    by_first_token.run_schema(tmp_path / "features.pkl", tmp_path / "out", seeds=(42,), by_relation=True)
    assert trained == [("B", "attention", 42)]
    assert (tmp_path / "out" / "mam_preds_seed42.jsonl").read_bytes() == \
        (tmp_path / "whole" / "mam_preds_seed42.jsonl").read_bytes()


def test_checkpoints_are_removed_once_every_seed_is_written(tmp_path):
    write_features(tmp_path / "features.pkl", {"P1": alternating(40)})
    by_first_token.run_schema(tmp_path / "features.pkl", tmp_path / "out", seeds=(42, 43))
    assert not (tmp_path / "classifier_parts").exists()


def test_checkpoints_from_a_smaller_feature_set_are_not_reused(tmp_path, monkeypatch):
    write_features(tmp_path / "features.pkl", {"P1": alternating(40)})
    stop_after(monkeypatch, 1)
    with pytest.raises(Stopped):
        by_first_token.run_schema(tmp_path / "features.pkl", tmp_path / "out", seeds=(42,))
    assert list((tmp_path / "classifier_parts").iterdir())

    monkeypatch.undo()
    shared.append_feature(tmp_path / "features.pkl", feature(np.random.default_rng(1), "d-en-P1-new", "P1", 1, "first"))
    trained = stop_after(monkeypatch, 99)
    by_first_token.run_schema(tmp_path / "features.pkl", tmp_path / "out", seeds=(42,))
    assert len(trained) == 2


def test_a_pause_stops_before_the_next_training_and_keeps_the_finished_ones(tmp_path, monkeypatch):
    write_features(tmp_path / "features.pkl", {"P1": alternating(40)})
    monkeypatch.setattr(shared, "PAUSE_FILE", tmp_path / "PAUSE")
    real = by_first_token.predict_group_g6

    def pause_after_training(*args):
        rows = real(*args)
        (tmp_path / "PAUSE").touch()
        return rows

    monkeypatch.setattr(by_first_token, "predict_group_g6", pause_after_training)
    with pytest.raises(shared.Paused):
        by_first_token.run_schema(tmp_path / "features.pkl", tmp_path / "out", seeds=(42,))
    assert len(list((tmp_path / "classifier_parts").iterdir())) == 1


def test_a_finished_language_does_not_load_its_features_again(tmp_path, monkeypatch):
    write_features(tmp_path / "features.pkl", {"P1": alternating(40)})
    by_first_token.run_schema(tmp_path / "features.pkl", tmp_path / "out", seeds=(42,))

    def must_not_load(_):
        raise AssertionError("features were loaded")

    monkeypatch.setattr(shared, "read_features", must_not_load)
    by_first_token.run_schema(tmp_path / "features.pkl", tmp_path / "out", seeds=(42,))


def test_missing_feature_file_fails_with_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="features.pkl"):
        by_first_token.run_schema(tmp_path / "features.pkl", tmp_path / "out")


def test_answer_token_variant_reads_the_averaged_token_features(tmp_path):
    labels = write_features(tmp_path / "features.pkl", {"P1": alternating(60)}, signal_key="token")
    by_answer_tokens.run_schema(tmp_path / "features.pkl", tmp_path / "out", seeds=(42,))
    for artifact in ("fully_connected", "attention"):
        got = [r for r in rows(tmp_path / "out", 42) if r["artifact"] == artifact]
        assert len(got) == 12
        assert roc_auc_score([labels[r["uid"]] for r in got], [r["p_hallucinated"] for r in got]) > 0.9


def test_first_token_variant_ignores_the_averaged_token_features(tmp_path):
    labels = write_features(tmp_path / "features.pkl", {"P1": alternating(60)}, signal_key="token")
    by_first_token.run_schema(tmp_path / "features.pkl", tmp_path / "out", seeds=(42,))
    got = [r for r in rows(tmp_path / "out", 42) if r["artifact"] == "attention"]
    assert roc_auc_score([labels[r["uid"]] for r in got], [r["p_hallucinated"] for r in got]) < 0.9


def test_rows_are_valid_json_lines(tmp_path):
    write_features(tmp_path / "features.pkl", {"P1": alternating(40)})
    by_first_token.run_schema(tmp_path / "features.pkl", tmp_path / "out", seeds=(42,))
    lines = (tmp_path / "out" / "mam_preds_seed42.jsonl").read_text(encoding="utf-8").splitlines()
    assert all(set(json.loads(line)) == {"uid", "relation", "split", "p_hallucinated", "artifact"}
               for line in lines)
