import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.analysis import metrics

RANGE = (0.4, 0.9)


def test_accuracy_is_share_of_label_zero_in_points():
    assert metrics.accuracy([0, 0, 0, 1]) == pytest.approx(75.0)


def test_auroc_perfect_and_reversed_ranking():
    assert metrics.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(100.0)
    assert metrics.auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)


def test_auroc_is_none_when_one_class():
    assert metrics.auroc([1, 1, 1], [0.1, 0.2, 0.3]) is None


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.random(200), 1)  # many ties
    assert metrics.auroc(y, s) == pytest.approx(100 * roc_auc_score(y, s))


def test_deltas_english_minus_other():
    assert metrics.deltas(80.0, 55.0) == 25.0


def test_tphr_defined_value():
    value, marker = metrics.tphr(20.0, 2.0)
    assert marker is None and value == pytest.approx(1.0)


def test_tphr_marker_un_when_no_accuracy_drop():
    assert metrics.tphr(0, 3.0) == (None, "UN")


def test_tphr_marker_na_when_no_auroc_drop():
    assert metrics.tphr(10.0, 0) == (None, "NA")


def test_tphr_undefined_when_accuracy_drop_under_five_keeps_raw_value():
    value, marker = metrics.tphr(4.0, 2.0)
    assert marker == "undefined" and value == pytest.approx(np.log10(2.0))


def test_tphr_undefined_when_auroc_rises():
    value, marker = metrics.tphr(10.0, -2.0)
    assert marker == "undefined" and value == pytest.approx(np.log10(5.0))


def test_tphr_array_is_nan_where_undefined():
    out = metrics.tphr_array([20.0, 4.0, 10.0], [2.0, 2.0, -1.0])
    assert out[0] == pytest.approx(1.0) and np.isnan(out[1:]).all()


def test_bootstrap_is_reproducible_with_the_seed():
    y, s = np.array([0, 1, 0, 1, 1, 0, 1, 0]), np.linspace(0, 1, 8)
    a = metrics.boot_auroc(y, s, np.random.default_rng(metrics.SEED), 50)
    b = metrics.boot_auroc(y, s, np.random.default_rng(metrics.SEED), 50)
    assert np.array_equal(a, b, equal_nan=True)


def test_paired_accuracy_bootstrap_resamples_facts_together():
    y = np.array([0, 1, 0, 1])
    en, other = metrics.boot_accuracy(y, y, np.random.default_rng(1), paired=True, resamples=20)
    assert np.array_equal(en, other)


def test_interval_reports_undefined_share_and_unstable_flag():
    low, high, share, unstable = metrics.interval([1.0, 2.0, 3.0, np.nan])
    assert (low, high) == pytest.approx((1.05, 2.95)) and share == 0.25 and unstable


def test_interval_stable_when_few_undefined():
    values = np.concatenate([np.linspace(0, 1, 95), np.full(5, np.nan)])
    assert metrics.interval(values)[3] is False


def test_decision_undefined_when_marker_set():
    assert metrics.decision(0.5, "undefined", (0.4, 0.6), RANGE) == "undefined"


def test_decision_closes_when_point_not_positive():
    assert metrics.decision(-0.1, None, (-0.3, 0.2), RANGE) == "closes"


def test_decision_holds_when_interval_overlaps_range():
    assert metrics.decision(0.5, None, (0.3, 0.6), RANGE) == "holds"


def test_decision_holds_when_interval_above_range():
    assert metrics.decision(1.2, None, (1.0, 1.4), RANGE) == "holds"


def test_decision_narrows_when_interval_below_range_and_above_zero():
    assert metrics.decision(0.2, None, (0.1, 0.3), RANGE) == "narrows"


def test_decision_inconclusive_when_interval_below_range_and_crosses_zero():
    assert metrics.decision(0.2, None, (-0.1, 0.3), RANGE) == "inconclusive"


def test_decision_inconclusive_without_interval():
    assert metrics.decision(0.5, None, (np.nan, np.nan), RANGE) == "inconclusive"


def test_spearman_drops_missing_pairs():
    rho, p, n = metrics.spearman([1, 2, 3, 4, 5], [1, 2, None, 4, 5])
    assert n == 4 and rho == pytest.approx(1.0)
