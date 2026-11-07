"""Phase 3 summary: accuracy and AUROC per dataset and language, from the saved results only.

Writes results/summary_phase3.csv and results/phase3_acc_auroc.png.
Labels: 1 = hallucinated. SelfCheckGPT score and semantic entropy: higher = more hallucinated.
MAM AUROC is the mean over seeds 42, 43, 44 on each seed's test split.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io import ROOT

SEEDS = (42, 43, 44)
ORDER = {
    "mtrex": ["en", "de", "hi", "bn", "ur"],
    "xfactr": ["en", "zh", "ja", "vi", "tl", "sw", "yo", "ilo"],
    "psiloqa": ["en", "hi", "zh", "fi", "ar"],
}
COLUMNS = ["dataset", "language", "n", "accuracy", "mam_fc_auroc", "mam_attn_auroc",
           "sgm_n", "sgm_auroc", "sem_n", "sem_auroc", "sem_hallucinated"]


def _rows(path) -> list:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _auroc(labels, scores):
    """None when the test split has one class only."""
    return roc_auc_score(labels, scores) if len(set(labels)) == 2 else None


def _mean(values):
    return round(sum(values) / len(values), 3) if values else None


def _mam_auroc(folder, labels) -> dict:
    per_artifact = defaultdict(list)
    for seed in SEEDS:
        groups = defaultdict(lambda: ([], []))
        for r in _rows(folder / f"mam_preds_seed{seed}.jsonl"):
            if r.get("split", "test") == "test":
                groups[r["artifact"]][0].append(labels[r["uid"]])
                groups[r["artifact"]][1].append(r["p_hallucinated"])
        for artifact, (y, p) in groups.items():
            value = _auroc(y, p)
            if value is not None:
                per_artifact[artifact].append(value)
    return {a: _mean(v) for a, v in per_artifact.items()}


def _summarize_folder(dataset, language, folder) -> dict:
    labels = {r["uid"]: r["label"] for r in _rows(folder / "answers.jsonl")}
    mam = _mam_auroc(folder, labels)
    sgm = _rows(folder / "sgm.jsonl")
    sem = _rows(folder / "sem.jsonl")
    sem_labels = [max(labels[u] for u in r["uids"]) for r in sem]
    sgm_auroc = _auroc([labels[r["uid"]] for r in sgm], [r["score"] for r in sgm])
    sem_auroc = _auroc(sem_labels, [r["entropies"]["semantic_entropy"] for r in sem])
    return {
        "dataset": dataset, "language": language, "n": len(labels),
        "accuracy": round(1 - sum(labels.values()) / len(labels), 3),
        "mam_fc_auroc": mam.get("fully_connected"), "mam_attn_auroc": mam.get("attention"),
        "sgm_n": len(sgm), "sgm_auroc": None if sgm_auroc is None else round(sgm_auroc, 3),
        "sem_n": len(sem), "sem_auroc": None if sem_auroc is None else round(sem_auroc, 3),
        "sem_hallucinated": sum(sem_labels),
    }


def summarize(results_dir) -> list:
    """One row per dataset and language folder found, in the order of ORDER (unknown ones last)."""
    results_dir = Path(results_dir)
    rows = []
    for dataset, languages in ORDER.items():
        found = sorted(p.name for p in (results_dir / dataset).glob("*") if (p / "answers.jsonl").exists())
        for language in sorted(found, key=lambda x: languages.index(x) if x in languages else len(languages)):
            rows.append(_summarize_folder(dataset, language, results_dir / dataset / language))
    return rows


def plot(rows, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {"mtrex": "mTREx", "xfactr": "X-FACTR", "psiloqa": "PsiloQA"}
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.6), dpi=200, sharey=True,
                             gridspec_kw={"width_ratios": [len(v) for v in ORDER.values()]})
    for ax, (dataset, _) in zip(axes, ORDER.items()):
        part = [r for r in rows if r["dataset"] == dataset]
        x = range(len(part))
        ax.bar([i - 0.2 for i in x], [r["accuracy"] for r in part], 0.4, color="#9a9a9a", label="Accuracy")
        ax.bar([i + 0.2 for i in x], [r["mam_fc_auroc"] or 0 for r in part], 0.4, color="#3b6e8f",
               label="MAM AUROC")
        ax.set_xticks(list(x))
        ax.set_xticklabels([r["language"] for r in part], fontsize=7)
        ax.set_xlabel(names[dataset], fontsize=8)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    rows = summarize(ROOT / "results")
    out = ROOT / "results" / "summary_phase3.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, ROOT / "results" / "phase3_acc_auroc.png")
    print(",".join(COLUMNS))
    for r in rows:
        print(",".join("" if r[c] is None else str(r[c]) for c in COLUMNS))


if __name__ == "__main__":
    main()
