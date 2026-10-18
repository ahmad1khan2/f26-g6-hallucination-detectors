"""PsiloQA -> shared schema. Single-turn long answers; label = 1 iff the answer has a span."""
from collections import Counter

import pandas as pd
from datasets import concatenate_datasets, load_from_disk

from src.common.io import ROOT, hash_key
from src.common.schema import Record


def _quotas(sizes: Counter, n: int) -> dict:
    """Split n across generator models in proportion to their size (largest remainder)."""
    total = sum(sizes.values())
    quota = {m: n * c // total for m, c in sizes.items()}
    by_remainder = sorted(sizes, key=lambda m: -(n * sizes[m] % total))
    for m in by_remainder[: n - sum(quota.values())]:
        quota[m] += 1
    return quota


def build(cfg, log=print):
    seed, n = cfg["seed"], cfg["psiloqa"]["items_per_language"]
    ds = load_from_disk(str(ROOT / "data" / "raw" / "psiloqa"))
    df = concatenate_datasets([ds[s] for s in ds]).to_pandas()
    df = df[df["lang"].isin(cfg["psiloqa"]["languages"])]
    log(f"psiloqa: {len(df)} rows in chosen languages (train+validation+test pooled)")
    df = df[df["llm_answer"].str.strip() != ""]
    log(f"psiloqa: {len(df)} after dropping empty answers")
    df = df.drop_duplicates(["lang", "question", "llm_answer"])
    log(f"psiloqa: {len(df)} after dropping duplicate question-answer pairs")
    records = []
    for lang in cfg["psiloqa"]["languages"]:
        d = df[df["lang"] == lang]
        quota = _quotas(Counter(d["llm_checkpoint"]), n)
        if len(d) < n:
            raise ValueError(f"psiloqa {lang}: only {len(d)} items, need {n}")
        d = d.assign(key=[hash_key(seed, i) for i in d["id"]]).sort_values("key")
        d = pd.concat([g.head(quota[m]) for m, g in d.groupby("llm_checkpoint")]).sort_values("key")
        for i, row in enumerate(d.itertuples()):
            spans = [{"start": int(s), "end": int(e)} for s, e in row.labels]
            records.append(Record(
                uid=f"psiloqa-{lang}-{i:06d}", dataset="psiloqa", language=lang, prompt=row.question,
                answer=row.llm_answer, spans=spans, label=int(bool(spans)),
                meta={"model": row.llm_checkpoint, "source_id": row.id, "passage": row.wiki_passage,
                      "golden_answer": row.golden_answer, "complexity": row.complexity}))
        rate = sum(r.label for r in records if r.language == lang) / n
        log(f"psiloqa {lang}: {n} items, hallucination rate {rate:.2f}, models {dict(quota)}")
    return records
