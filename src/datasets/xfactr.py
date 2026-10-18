"""X-FACTR -> shared schema. Fill-in-the-blank prompts, English paired fact by fact."""
import json
import re
from collections import Counter

import pandas as pd

from src.common.io import ROOT, hash_key, load_config
from src.common.schema import Record
from src.common.text import normalize

DATA = ROOT / "data" / "raw" / "xfactr" / "data"
INFLECTION_TAG = re.compile(r"\[([XY])\.[^\]]*\]")  # [Y.Loc] -> [Y]


def _load_aliases(lang):
    """entity id -> (label, aliases), tab-separated lines."""
    out = {}
    with open(DATA / "alias" / "mTRExf" / f"{lang}.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 2 and parts[1].strip():
                out[parts[0]] = (parts[1].strip(), [p.strip() for p in parts[2:] if p.strip()])
    return out


def _load_facts():
    facts = []
    for path in sorted((DATA / "mTRExf" / "sub").glob("P*.jsonl")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            d = json.loads(line)
            facts.append({"pid": d["predicate_id"], "n": lineno, "sub": d["sub_uri"], "obj": d["obj_uri"]})
    return facts


def _prompt(template, subject, blank):
    return INFLECTION_TAG.sub(r"[\1]", template).replace("[X]", subject).replace("[Y]", blank).strip()


def _gold(label_aliases):
    label, aliases = label_aliases
    return list(dict.fromkeys(g for g in (normalize(x) for x in [label, *aliases]) if g))


def _record(lang, fact, template, alias, blank, pair_uid=""):
    subject = alias[fact["sub"]][0]
    return Record(
        uid=f"xfactr-{lang}-{fact['pid']}-{fact['n']:06d}", dataset="xfactr", language=lang,
        relation=fact["pid"], prompt=_prompt(template, subject, blank),
        gold=_gold(alias[fact["obj"]]), pair_uid=pair_uid,
        meta={"subject": subject, "sub_uri": fact["sub"], "obj_uri": fact["obj"]})


def eligible_facts(cfg, log=print):
    """Return (chosen langs, templates, aliases, kept relations, eligible fact ids per language)."""
    chosen = [l["code"] for l in load_config("languages")["languages"] if l["chosen"]]
    t = pd.read_csv(DATA / "TREx_prompts.csv", dtype=str, keep_default_na=False).set_index("pid")
    langs = ["en", *chosen]
    gendered = {pid for pid, row in t.iterrows() if any("Gender" in row[l] or not row[l] for l in langs)}
    kept = [pid for pid in t.index if pid not in gendered]
    log(f"xfactr: {len(t)} relations, dropped {len(gendered)} gender-dependent or untemplated, kept {len(kept)}")
    aliases = {l: _load_aliases(l) for l in langs}
    facts = [f for f in _load_facts() if f["pid"] in kept]
    log(f"xfactr: {len(facts)} facts in kept relations")
    eligible = {}
    for l in chosen:
        eligible[l] = {(f["pid"], f["n"]) for f in facts
                       if all(f[k] in aliases[x] for k in ("sub", "obj") for x in (l, "en"))}
        log(f"xfactr {l}: {len(eligible[l])} eligible facts")
    templates = {l: t[l].to_dict() for l in langs}
    return chosen, templates, aliases, facts, eligible


def build(cfg, log=print):
    seed, cap, blank = cfg["seed"], cfg["xfactr"]["cap_per_relation"], cfg["xfactr"]["blank"]
    chosen, templates, aliases, facts, eligible = eligible_facts(cfg, log)
    by_id = {(f["pid"], f["n"]): f for f in facts}
    coverage = Counter(fid for ids in eligible.values() for fid in ids)
    order = sorted(coverage, key=lambda fid: (-coverage[fid], hash_key(seed, f"{fid[0]}-{fid[1]}")))
    records, english_ids = [], {}
    for l in chosen:
        taken = Counter()
        for fid in order:
            if fid in eligible[l] and taken[fid[0]] < cap:
                taken[fid[0]] += 1
                english_ids[fid] = None
                f = by_id[fid]
                records.append(_record(l, f, templates[l][f["pid"]], aliases[l], blank,
                                       pair_uid=f"xfactr-en-{f['pid']}-{f['n']:06d}"))
        log(f"xfactr {l}: {sum(taken.values())} facts over {len(taken)} relations")
    for fid in sorted(english_ids):
        f = by_id[fid]
        records.append(_record("en", f, templates["en"][f["pid"]], aliases["en"], blank))
    log(f"xfactr: {len(english_ids)} distinct English facts, {len(records)} records")
    return records
