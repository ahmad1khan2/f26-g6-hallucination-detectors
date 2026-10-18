import pytest

from src.common.schema import Record, read_jsonl, write_jsonl

DUMMIES = [
    Record(uid="mtrex-de-capitals-000001", dataset="mtrex", language="de",
           relation="capitals", prompt="Was ist die Hauptstadt von Frankreich?", gold=["paris"]),
    Record(uid="xfactr-sw-P36-000123", dataset="xfactr", language="sw", relation="P36",
           prompt="Mji mkuu wa Ufaransa ni ___.", gold=["paris"],
           pair_uid="xfactr-en-P36-000123", meta={"subj": "Q142", "obj": "Q90"}),
    Record(uid="psiloqa-hi-000001", dataset="psiloqa", language="hi", prompt="प्रश्न",
           answer="उत्तर", spans=[{"start": 0, "end": 2}], label=1, meta={"model": "gpt"}),
]


@pytest.mark.parametrize("record", DUMMIES, ids=lambda r: r.dataset)
def test_roundtrip(record, tmp_path):
    path = tmp_path / "x.jsonl"
    write_jsonl(path, [record])
    assert read_jsonl(path) == [record]


@pytest.mark.parametrize("bad", [
    dict(dataset="other"),
    dict(label=2),
    dict(gold=[]),
    dict(dataset="psiloqa", answer="a", spans=[{"start": 0, "end": 1}], label=0),
])
def test_invalid_records_rejected(bad):
    base = dict(uid="u", dataset="mtrex", language="en", prompt="p", gold=["g"])
    with pytest.raises(ValueError):
        Record(**{**base, **bad})
