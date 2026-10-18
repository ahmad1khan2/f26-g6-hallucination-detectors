"""mTREx -> shared schema. Same number of rows per language and category."""
import json

import pandas as pd

from src.common.io import ROOT, hash_key
from src.common.schema import Record
from src.common.text import normalize

RAW = ROOT / "data" / "raw" / "mtrex_repo"


def _load_cell(folder, category, log):
    df = pd.read_csv(RAW / "data" / "mTREx" / folder / f"{category}.csv", dtype=str, keep_default_na=False)
    n_raw = len(df)
    df["source"] = df["source"].str.strip()
    df["gold"] = df["target"].map(lambda t: [g for g in (normalize(x) for x in t.split("<OR>")) if g])
    df = df[(df["source"] != "") & (df["gold"].map(len) > 0)]
    n_nonempty = len(df)
    df = df.drop_duplicates("source")
    log(f"mtrex {folder}/{category}: raw {n_raw}, non-empty {n_nonempty}, unique source {len(df)}")
    return df


def build(cfg, log=print):
    seed = cfg["seed"]
    prompts = json.loads((RAW / "prompts" / "mTREx_prompts.json").read_text(encoding="utf-8"))
    cells = {(folder, cat): _load_cell(folder, cat, log)
             for folder in cfg["mtrex"]["languages"] for cat in cfg["mtrex"]["categories"]}
    n = min(len(df) for df in cells.values())
    log(f"mtrex: rows per language and category = {n} (smallest cell)")
    records = []
    for (folder, cat), df in cells.items():
        lang = cfg["mtrex"]["languages"][folder]
        user = prompts[folder][cat]["user"]
        df = df.assign(key=[hash_key(seed, f"{lang}:{cat}:{s}") for s in df["source"]]).sort_values("key").head(n)
        for i, row in enumerate(df.itertuples()):
            records.append(Record(
                uid=f"mtrex-{lang}-{cat}-{i:06d}", dataset="mtrex", language=lang, relation=cat,
                prompt=user.format(source=row.source).strip(), gold=row.gold,
                meta={"source": row.source}))
    log(f"mtrex: {len(records)} records")
    return records
