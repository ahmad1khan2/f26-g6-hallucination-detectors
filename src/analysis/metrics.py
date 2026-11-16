"""Metric helpers for Phase 4, written to src/analysis/ANALYSIS_SPEC.md.

Everything here is pure: labels are 1 = hallucinated, scores higher = more
likely hallucinated, accuracy and AUROC are in points (x100).
"""
import numpy as np
from scipy.stats import rankdata, spearmanr

SEED = 42
RESAMPLES = 1000
MIN_ACC_DROP = 5.0        # below this the accuracy drop is too small for a ratio (proposal rule)
UNSTABLE_SHARE = 0.10     # interval is "unstable" when more than this share of resamples is undefined


def accuracy(labels) -> float:
    """Share of correct answers (label 0), in points."""
    return 100.0 * (1.0 - float(np.mean(labels)))


def auroc(labels, scores):
    """Rank-based AUROC in points (ties get average ranks); None if one class only.

    Rank formula instead of sklearn so the bootstrap can be vectorised; the
    tests check it against sklearn.
    """
    y = np.asarray(labels)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores)
    return 100.0 * (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def deltas(english, other):
    """Drop relative to English (points); positive = the other language is worse."""
    return english - other


def tphr(d_acc, d_auroc):
    """(log10(|dAcc| / |dAUROC|) or None, marker).

    Markers: 'UN' (dAcc = 0), 'NA' (dAUROC = 0), 'undefined' (dAcc under 5 points or
    AUROC does not fall), None when the proposal's rule defines it. The raw value is
    still returned when it can be computed so tables can show it.
    """
    value = None if d_acc == 0 or d_auroc == 0 else float(np.log10(abs(d_acc) / abs(d_auroc)))
    if d_acc == 0:
        return value, "UN"
    if d_auroc == 0:
        return value, "NA"
    if d_acc < MIN_ACC_DROP or d_auroc <= 0:
        return value, "undefined"
    return value, None


def tphr_array(d_acc, d_auroc) -> np.ndarray:
    """Vectorised TPHR for resamples: nan wherever the rule leaves it undefined."""
    d_acc, d_auroc = np.asarray(d_acc, float), np.asarray(d_auroc, float)
    defined = (d_acc >= MIN_ACC_DROP) & (d_auroc > 0)  # nan inputs compare False
    out = np.full(d_acc.shape, np.nan)
    out[defined] = np.log10(d_acc[defined] / d_auroc[defined])
    return out


def boot_auroc(labels, scores, rng, resamples=RESAMPLES) -> np.ndarray:
    """Bootstrap AUROC values (points); nan for a resample with one class."""
    y, s = np.asarray(labels), np.asarray(scores)
    idx = rng.integers(0, len(y), (resamples, len(y)))
    yb = y[idx]
    ranks = rankdata(s[idx], axis=1)
    n_pos = yb.sum(axis=1)
    n_neg = len(y) - n_pos
    denominator = n_pos * n_neg
    out = np.full(resamples, np.nan)
    ok = denominator > 0
    out[ok] = 100.0 * ((ranks * yb).sum(axis=1)[ok] - n_pos[ok] * (n_pos[ok] + 1) / 2) / denominator[ok]
    return out


def boot_accuracy(labels_en, labels_other, rng, paired, resamples=RESAMPLES):
    """Bootstrap accuracies (points) for English and the other language.

    paired=True resamples facts together (item i of both arrays), as for X-FACTR.
    """
    y_en, y_other = np.asarray(labels_en), np.asarray(labels_other)
    if paired:
        idx = rng.integers(0, len(y_en), (resamples, len(y_en)))
        return 100.0 * (1 - y_en[idx].mean(axis=1)), 100.0 * (1 - y_other[idx].mean(axis=1))
    idx_en = rng.integers(0, len(y_en), (resamples, len(y_en)))
    idx_other = rng.integers(0, len(y_other), (resamples, len(y_other)))
    return 100.0 * (1 - y_en[idx_en].mean(axis=1)), 100.0 * (1 - y_other[idx_other].mean(axis=1))


def interval(values):
    """(low, high, undefined_share, unstable): 2.5th to 97.5th percentile over the defined values."""
    values = np.asarray(values, float)
    defined = values[~np.isnan(values)]
    share = 1.0 - len(defined) / len(values)
    if len(defined) == 0:
        return float("nan"), float("nan"), 1.0, True
    low, high = np.percentile(defined, [2.5, 97.5])
    return float(low), float(high), share, share > UNSTABLE_SHARE


def decision(point, marker, ci, mtrex_range) -> str:
    """Proposal decision rule, spec item 11.

    undefined: no decision; point <= 0: closes; interval overlapping or above the mTREx
    range: holds; interval entirely below it with lower bound above 0: narrows; else inconclusive.
    """
    if marker is not None or point is None:
        return "undefined"
    if point <= 0:
        return "closes"
    low, high = ci
    if mtrex_range is None or np.isnan(low) or np.isnan(high):
        return "inconclusive"
    if high >= mtrex_range[0]:
        return "holds"
    return "narrows" if low > 0 else "inconclusive"


def spearman(x, y):
    """(rho, p, n) over the pairs where both values exist."""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None
             and not np.isnan(a) and not np.isnan(b)]
    if len(pairs) < 3:
        return float("nan"), float("nan"), len(pairs)
    rho, p = spearmanr(*zip(*pairs))
    return float(rho), float(p), len(pairs)
