"""Shared record layout for all three datasets (one JSON object per line)."""
import json
from dataclasses import asdict, dataclass, field

DATASETS = ("mtrex", "xfactr", "psiloqa")


@dataclass
class Record:
    uid: str
    dataset: str
    language: str
    prompt: str
    relation: str = ""          # category or relation id; empty for PsiloQA
    gold: list = field(default_factory=list)   # accepted answers (mTREx, X-FACTR)
    answer: str = ""            # PsiloQA only: the given answer
    spans: list = field(default_factory=list)  # PsiloQA only: [{"start", "end"}]
    label: int | None = None    # 1 = hallucinated; PsiloQA now, others in Phase 3
    pair_uid: str = ""          # X-FACTR only: the same fact in English
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.dataset not in DATASETS:
            raise ValueError(f"unknown dataset {self.dataset!r}")
        if not (self.uid and self.language and self.prompt):
            raise ValueError("uid, language and prompt are required")
        if self.label not in (None, 0, 1):
            raise ValueError(f"label must be 0, 1 or None, got {self.label!r}")
        if self.dataset == "psiloqa":
            if not self.answer or self.label != int(bool(self.spans)):
                raise ValueError("psiloqa needs an answer and label == (has spans)")
        elif not self.gold:
            raise ValueError(f"{self.dataset} needs a non-empty gold list")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "Record":
        return cls(**json.loads(line))


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(r.to_json() + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [Record.from_json(line) for line in f if line.strip()]
