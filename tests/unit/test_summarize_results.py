"""Unit test for scripts/summarize_results.py on a tiny fake results folder."""
import json

from scripts.summarize_results import summarize


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_summarize_gives_accuracy_and_auroc(tmp_path):
    folder = tmp_path / "mtrex" / "en"
    folder.mkdir(parents=True)
    labels = {"u1": 0, "u2": 0, "u3": 1, "u4": 1}
    _write(folder / "answers.jsonl", [{"uid": u, "label": v} for u, v in labels.items()])
    perfect = {"u1": 0.1, "u2": 0.2, "u3": 0.8, "u4": 0.9}
    for seed in (42, 43, 44):
        _write(folder / f"mam_preds_seed{seed}.jsonl",
               [{"uid": u, "split": "test", "artifact": "fully_connected", "p_hallucinated": p}
                for u, p in perfect.items()])
    _write(folder / "sgm.jsonl", [{"uid": u, "score": 1 - p} for u, p in perfect.items()])  # inverted: AUROC 0
    _write(folder / "sem.jsonl", [{"uid": u, "uids": [u], "entropies": {"semantic_entropy": p}}
                                  for u, p in perfect.items()])

    (row,) = summarize(tmp_path)

    assert row["accuracy"] == 0.5
    assert row["mam_fc_auroc"] == 1.0
    assert row["mam_attn_auroc"] is None
    assert row["sgm_auroc"] == 0.0
    assert row["sem_auroc"] == 1.0 and row["sem_hallucinated"] == 2
