"""Unit tests for the MAM classifier training step and for the real-results checker."""
import json
import math

import numpy as np
import pytest
import torch

from src.common.schema import Record, write_jsonl
from src.detectors import ModelArtifacts_Classifier_g6 as classifier
from src.detectors import shared
from tests.test_results import check_results

HIDDEN = 8


def separable(n=60, seed=0):
    """Two clusters far apart: label 1 around +3, label 0 around -3."""
    rng = np.random.default_rng(seed)
    labels = np.array([i % 2 for i in range(n)])
    return (rng.normal(size=(n, HIDDEN)) + np.where(labels[:, None] == 1, 3.0, -3.0)).astype(np.float32), labels


def train(tmp_path, inputs, labels, seed):
    preds = []
    auroc, _ = classifier.gen_classifier_roc(
        inputs=inputs, correct=labels, artifact_name="toy", device=torch.device("cpu"),
        cls_results_file_path=str(tmp_path / "r.txt"), cls_results_dir=str(tmp_path), num_hidden_nodes=256,
        dropout_mlp=0.5, learning_rate=1e-4, weight_decay=1e-2, batch_size=128, seed_val=seed, preds_out=preds)
    return auroc, preds


def test_same_seed_gives_same_test_indices_and_different_seed_differs(tmp_path):
    inputs, labels = separable()
    first = [p["index"] for p in train(tmp_path, inputs, labels, 42)[1]]
    again = [p["index"] for p in train(tmp_path, inputs, labels, 42)[1]]
    other = [p["index"] for p in train(tmp_path, inputs, labels, 43)[1]]
    assert first == again and first != other and len(first) == 12


def test_probabilities_learn_a_separable_problem(tmp_path):
    inputs, labels = separable()
    auroc, preds = train(tmp_path, inputs, labels, 42)
    assert auroc > 0.9 and all(0 <= p["prediction"] <= 1 for p in preds)


def test_one_class_test_split_returns_nan_auroc_but_keeps_predictions(tmp_path):
    inputs, _ = separable(n=10)
    labels = np.zeros(10, dtype=int)
    auroc, preds = train(tmp_path, inputs, labels, 42)
    assert math.isnan(auroc) and len(preds) == 2


# ---- check_results ----------------------------------------------------------------------------

N_ITEMS = 40


class World:
    """A small valid results tree: xfactr/en with 40 processed items, 20 of them in one MAM test split."""

    def __init__(self, tmp_path):
        self.processed = tmp_path / "processed"
        self.processed.mkdir()
        self.root = tmp_path / "results"
        self.lang = self.root / "xfactr" / "en"
        self.lang.mkdir(parents=True)
        self.cfg = {"mam_limit": None, "sampling_limit": {"xfactr": 10}}
        self.uids = [f"xfactr-en-P1-{i:06d}" for i in range(N_ITEMS)]
        write_jsonl(self.processed / "xfactr.jsonl",
                    [Record(uid=u, dataset="xfactr", language="en", prompt="p", relation="P1", gold=["a"])
                     for u in self.uids])
        for i, u in enumerate(self.uids):
            shared.append_jsonl(self.lang / "answers.jsonl", {"uid": u, "answer": "a", "label": i % 2,
                                                              "label_rule": "matches", "first_token_entropy": 0.5})
        for u in self.uids[:10]:
            shared.append_jsonl(self.lang / "sgm.jsonl", {"uid": u, "score": 0.3})
            shared.append_jsonl(self.lang / "sem.jsonl", {"uid": u, "uids": [u], "entropies": {"a": 0.2},
                                                          "level": "item"})
        for u in self.uids[:20]:
            shared.append_jsonl(self.lang / "mam_preds_seed42.jsonl", {
                "uid": u, "relation": "P1", "split": "test", "p_hallucinated": 0.5, "artifact": "attention"})

    def check(self):
        return check_results(self.root, self.processed, self.cfg)

    def edit(self, name, change):
        path = self.lang / name
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        change(rows)  # in place
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def test_a_clean_tree_has_no_flags(world):
    assert world.check() == []


@pytest.mark.parametrize("name", ["answers.jsonl", "sgm.jsonl", "sem.jsonl", "mam_preds_seed42.jsonl"])
def test_uid_missing_from_processed_file_fails(world, name):
    world.edit(name, lambda rows: rows[0].update(uid="xfactr-en-P1-999999"))
    with pytest.raises(AssertionError, match="999999"):
        world.check()


def test_uid_listed_inside_a_sem_row_must_exist(world):
    world.edit("sem.jsonl", lambda rows: rows[0].update(uids=["xfactr-en-P1-999999"]))
    with pytest.raises(AssertionError, match="999999"):
        world.check()


def test_label_other_than_zero_or_one_fails(world):
    world.edit("answers.jsonl", lambda rows: rows[0].update(label=2))
    with pytest.raises(AssertionError, match="label"):
        world.check()


@pytest.mark.parametrize("name,patch", [
    ("sgm.jsonl", {"score": float("nan")}),
    ("mam_preds_seed42.jsonl", {"p_hallucinated": float("inf")}),
    ("sem.jsonl", {"entropies": {"a": float("nan")}}),
    ("answers.jsonl", {"first_token_entropy": float("nan")}),
])
def test_non_finite_scores_fail(world, name, patch):
    world.edit(name, lambda rows: rows[0].update(patch))
    with pytest.raises(AssertionError, match="finite"):
        world.check()


def test_preds_probability_outside_unit_interval_fails(world):
    world.edit("mam_preds_seed42.jsonl", lambda rows: rows[0].update(p_hallucinated=1.5))
    with pytest.raises(AssertionError, match="finite"):
        world.check()


@pytest.mark.parametrize("name", ["answers.jsonl", "sgm.jsonl", "sem.jsonl", "mam_preds_seed42.jsonl"])
def test_duplicate_uid_fails(world, name):
    world.edit(name, lambda rows: rows.append(dict(rows[0])))
    with pytest.raises(AssertionError, match="duplicate"):
        world.check()


def test_same_uid_under_two_artifacts_is_not_a_duplicate(world):
    world.edit("mam_preds_seed42.jsonl", lambda rows: rows.append(dict(rows[0], artifact="fully_connected")))
    assert world.check() == []


def test_answers_plus_mam_errors_must_equal_the_item_count(world):
    world.edit("answers.jsonl", lambda rows: rows.pop())
    with pytest.raises(AssertionError, match="answers"):
        world.check()
    shared.log_error(world.lang, world.uids[-1], "mam", ValueError("boom"))
    assert any("error" in f for f in world.check())


def test_errors_of_another_stage_do_not_count_as_mam(world):
    world.edit("answers.jsonl", lambda rows: rows.pop())
    shared.log_error(world.lang, world.uids[-1], "sgm", ValueError("boom"))
    with pytest.raises(AssertionError, match="answers"):
        world.check()


def test_mam_limit_sets_the_expected_count_and_is_capped_at_the_processed_count(world):
    world.cfg["mam_limit"] = 100
    assert world.check() == []
    world.cfg["mam_limit"] = 30
    with pytest.raises(AssertionError, match="answers"):
        world.check()


def test_sampling_count_must_match_the_limit(world):
    world.cfg["sampling_limit"]["xfactr"] = 12
    with pytest.raises(AssertionError, match="sgm"):
        world.check()
    shared.log_error(world.lang, world.uids[10], "sgm", ValueError("boom"))
    shared.log_error(world.lang, world.uids[11], "sgm", ValueError("boom"))
    assert any("error" in f for f in world.check())


def test_null_sampling_limit_means_all_items(world):
    world.cfg["sampling_limit"]["xfactr"] = None
    with pytest.raises(AssertionError, match="sgm"):
        world.check()


def test_sampling_count_is_not_checked_without_sgm_file(world):
    (world.lang / "sgm.jsonl").unlink()
    world.cfg["sampling_limit"]["xfactr"] = 12
    assert any("sgm.jsonl" in f and "missing" in f for f in world.check())


def test_thin_test_split_is_flagged(world):
    world.edit("answers.jsonl", lambda rows: [r.update(label=0) for r in rows])
    assert any("fewer than 10" in f for f in world.check())


def test_error_rows_are_flagged(world):
    shared.log_error(world.lang, "x", "sem", ValueError("boom"))
    assert any("xfactr/en" in f and "error" in f for f in world.check())


def test_language_without_a_results_folder_is_flagged_not_failed(world):
    extra = [Record(uid="xfactr-sw-P1-000000", dataset="xfactr", language="sw", prompt="p", relation="P1",
                    gold=["a"])]
    rows = (world.processed / "xfactr.jsonl").read_text(encoding="utf-8")
    (world.processed / "xfactr.jsonl").write_text(rows + extra[0].to_json() + "\n", encoding="utf-8")
    assert any("xfactr/sw" in f and "not run" in f for f in world.check())


@pytest.mark.parametrize("name", ["answers.jsonl", "sem.jsonl", "mam_preds_seed42.jsonl"])
def test_missing_file_is_flagged_not_failed(world, name):
    (world.lang / name).unlink()
    assert any(name.split("_seed")[0].split(".")[0] in f and "missing" in f for f in world.check())
