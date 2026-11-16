"""Unit tests for the median rule and the round-robin cap in scripts/error_sample.py."""
from scripts.error_sample import find_failures, select


def _item(uid, label, score):
    return {"uid": uid, "label": label, "score": score}


def test_failures_follow_the_median_rule():
    items = [_item("c1", 0, 0.1), _item("c2", 0, 0.3), _item("c3", 0, 0.9),
             _item("h1", 1, 0.2), _item("h2", 1, 0.5), _item("h3", 1, 0.8)]
    found = {f["uid"]: f["failure_type"] for f in find_failures(items)}
    # correct median 0.3, hallucinated median 0.5
    assert found == {"h1": "missed_hallucination", "c3": "false_flag"}


def test_single_class_gives_no_failures():
    assert find_failures([_item("h1", 1, 0.2), _item("h2", 1, 0.4)]) == []


def _language(prefix, n):
    """n missed hallucinations (score 0.1 < correct median) and n false flags (0.9 > hallucinated median)."""
    items = [_item(f"{prefix}m{i}", 1, 0.1) for i in range(n)]
    items += [_item(f"{prefix}f{i}", 0, 0.9) for i in range(n)]
    return items


def test_round_robin_covers_every_language_and_caps_at_15():
    chosen = select({"a": _language("a", 30), "b": _language("b", 2)}, ["a", "b"], cap=15)
    for failure_type in ("missed_hallucination", "false_flag"):
        first_letters = [f["uid"][0] for f in chosen if f["failure_type"] == failure_type]
        assert len(first_letters) == 15
        assert first_letters.count("b") == 2


def test_fewer_than_cap_returns_what_exists():
    assert len(select({"a": _language("a", 3)}, ["a"], cap=15)) == 6
