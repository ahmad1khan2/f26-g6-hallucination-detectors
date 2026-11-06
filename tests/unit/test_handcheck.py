"""Unit tests for stem_match and scripts/handcheck.py."""
import csv

import pytest

from scripts import handcheck
from src.common.text import stem_match

COLUMNS = ["uid", "language", "prompt", "gold", "answer", "matcher_label", "checked_label", "checked_by",
           "confidence", "note"]


def test_stem_match_uses_a_snowball_stemmer_where_one_exists():
    assert stem_match("The books", ["the book"], "en")
    assert stem_match("Talossa", ["talo"], "fi")
    assert not stem_match("the cars", ["the book"], "en")


def test_stem_match_falls_back_to_cutting_up_to_three_final_characters():
    assert stem_match("kitabuni", ["kitabuna"], "sw")     # both words cut to the same 5-letter stem
    assert stem_match("Kitabuni", ["kitabuna"], "sw")     # normalized first
    assert not stem_match("gari", ["kitabu"], "sw")
    assert not stem_match("abcd", ["abce"], "sw")         # words of 5 or fewer letters are left alone


def test_stem_match_needs_equality_not_a_substring():
    assert not stem_match("the book is old", ["the book"], "en")


def write_fixture(root, per_language=60):
    """results/xfactr/<language>/answers.jsonl and a processed file for two languages."""
    import json
    processed = []
    for language in ("en", "sw"):
        folder = root / "xfactr" / language
        folder.mkdir(parents=True)
        lines = []
        for i in range(per_language):
            uid = f"xfactr-{language}-P1-{i:06d}"
            processed.append({"uid": uid, "language": language, "prompt": f"p{i} ___", "gold": ["gold"]})
            lines.append(json.dumps({"uid": uid, "answer": "gold here" if i % 2 else "nope"}))
        (folder / "answers.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    processed_path = root / "xfactr.jsonl"
    processed_path.write_text("\n".join(json.dumps(r) for r in processed) + "\n", encoding="utf-8")
    return root / "xfactr", processed_path


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_draw_is_deterministic_capped_and_labels_with_the_matcher(tmp_path):
    results, processed = write_fixture(tmp_path)
    handcheck.draw(results, processed, tmp_path / "a.csv")
    handcheck.draw(results, processed, tmp_path / "b.csv")
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()
    assert (tmp_path / "a.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    rows = read_csv(tmp_path / "a.csv")
    assert list(rows[0]) == COLUMNS
    assert [sum(r["language"] == lang for r in rows) for lang in ("en", "sw")] == [50, 50]
    assert all(r["checked_label"] == "" for r in rows)
    answers = {r["answer"]: r["matcher_label"] for r in rows}
    assert answers == {"gold here": "0", "nope": "1"}      # label 1 = hallucinated


def test_draw_does_not_overwrite_a_filled_file(tmp_path):
    results, processed = write_fixture(tmp_path)
    handcheck.draw(results, processed, tmp_path / "c.csv")
    rows = read_csv(tmp_path / "c.csv")
    rows[0]["checked_label"] = "1"
    fill(tmp_path / "c.csv", rows)
    with pytest.raises(FileExistsError):
        handcheck.draw(results, processed, tmp_path / "c.csv")
    assert read_csv(tmp_path / "c.csv")[0]["checked_label"] == "1"


def fill(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def make_rows(language, matcher, checked):
    return [{"uid": f"{language}{i}", "language": language, "prompt": "", "gold": "", "answer": "",
             "matcher_label": m, "checked_label": c, "checked_by": "x", "confidence": "", "note": ""}
            for i, (m, c) in enumerate(zip(matcher, checked))]


def test_score_computes_agreement_per_language_and_lists_those_under_ninety(tmp_path):
    rows = make_rows("en", "0101010101", "0101010101") + make_rows("sw", "0101010101", "0101010111")
    fill(tmp_path / "h.csv", rows)
    agreement, low = handcheck.score(tmp_path / "h.csv")
    assert agreement == {"en": 1.0, "sw": 0.9} and low == []
    fill(tmp_path / "h.csv", make_rows("sw", "0101010101", "0000000000"))
    agreement, low = handcheck.score(tmp_path / "h.csv")
    assert agreement == {"sw": 0.5} and low == ["sw"]


def test_score_refuses_while_any_row_is_blank(tmp_path):
    rows = make_rows("en", "01", "01") + make_rows("sw", "01", "01")
    rows[1]["checked_label"] = ""
    fill(tmp_path / "h.csv", rows)
    with pytest.raises(ValueError, match="en"):
        handcheck.score(tmp_path / "h.csv")
