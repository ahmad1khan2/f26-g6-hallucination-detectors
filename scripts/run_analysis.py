"""Phase 4 analysis: reads results/ and data/processed/, writes results/tables/*.csv.

Follows src/analysis/ANALYSIS_SPEC.md. Labels: 1 = hallucinated. A "run" is one scored set
(uids, relations, labels, scores): three per MAM artifact (seeds 42, 43, 44), one for
SelfCheckGPT and one for Semantic Entropy.
"""
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import metrics as m
from src.common.io import ROOT, load_config
from src.common.text import matches, stem_match

RESULTS, PROCESSED, TABLES = ROOT / "results", ROOT / "data" / "processed", ROOT / "results" / "tables"
SEEDS = (42, 43, 44)
ORDER = {
    "mtrex": ["en", "de", "hi", "bn", "ur"],
    "xfactr": ["en", "zh", "ja", "vi", "tl", "sw", "yo", "ilo"],
    "psiloqa": ["en", "hi", "zh", "fi", "ar"],
}
MAM_ARTIFACT = {"mam_fc": "fully_connected", "mam_attn": "attention"}
DETECTORS = ["mam_fc", "mam_attn", "selfcheck", "sem"]
PRIMARY = "mam_fc"
MIN_CLASS = 30  # spec item 6 and the class-balance flag
MTREX_SCRIPTS = {"en": "Latin", "de": "Latin", "hi": "Devanagari", "bn": "Bengali", "ur": "Arabic"}
PSILOQA_SCRIPTS = {"en": "Latin", "fi": "Latin", "hi": "Devanagari", "zh": "Han", "ar": "Arabic"}
SCRIPT_CHARS = {"Devanagari": ["DEVANAGARI"], "Bengali": ["BENGALI"], "Arabic": ["ARABIC"],
                "Han": ["CJK UNIFIED"], "Japanese": ["HIRAGANA", "KATAKANA", "CJK UNIFIED"]}
PAPER_CSV = ROOT / "src" / "analysis" / "paper_numbers.csv"

REL, PAIR, GOLD, MODEL = {}, {}, {}, {}  # uid -> relation, English pair, gold list, generator model
DATA = {}                                # (setting, language) -> loaded results


def read(path) -> list:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_all() -> dict:
    """Fill the uid maps and DATA; returns language info (group, script, label_count) per setting."""
    for setting in ORDER:
        for r in read(PROCESSED / f"{setting}.jsonl"):
            REL[r["uid"]], PAIR[r["uid"]], GOLD[r["uid"]] = r["relation"], r["pair_uid"], r["gold"]
            MODEL[r["uid"]] = r["meta"].get("model")
    for setting, languages in ORDER.items():
        for lang in languages:
            folder = RESULTS / setting / lang
            answers = read(folder / "answers.jsonl")
            mam = {}
            for seed in SEEDS:
                for r in read(folder / f"mam_preds_seed{seed}.jsonl"):
                    if r["split"] == "test":
                        mam.setdefault((seed, r["artifact"]), []).append((r["uid"], r["p_hallucinated"]))
            DATA[(setting, lang)] = {
                "labels": {r["uid"]: r["label"] for r in answers},
                "answers": {r["uid"]: r["answer"] for r in answers},
                "mam": mam,
                "sgm": [(r["uid"], r["score"]) for r in read(folder / "sgm.jsonl")],
                "sem": [(r["uids"], r["entropies"]["semantic_entropy"]) for r in read(folder / "sem.jsonl")],
            }
    cfg = load_config("languages")
    info = {"xfactr": {l["code"]: l for l in cfg["languages"] if l["chosen"]}}
    info["xfactr"]["en"] = {"group": "english", "script": "Latin", "label_count": cfg["english"]["label_count"]}
    info["mtrex"] = {c: {"script": s} for c, s in MTREX_SCRIPTS.items()}
    info["psiloqa"] = {c: {"script": s} for c, s in PSILOQA_SCRIPTS.items()}
    return info


def stem_labels(lang) -> dict:
    """X-FACTR second label (spec Inputs): 0 if matches() or stem_match() accepts the answer."""
    d = DATA[("xfactr", lang)]
    return {u: 0 if matches(a, GOLD[u]) or stem_match(a, GOLD[u], lang) else 1 for u, a in d["answers"].items()}


def pack(uids, labels, scores):
    uids = np.array(uids)
    return uids, np.array([REL[u] for u in uids]), np.array(labels), np.array(scores, float)


def runs(d, detector, labels) -> list:
    if detector in MAM_ARTIFACT:
        out = []
        for seed in SEEDS:
            rows = d["mam"][(seed, MAM_ARTIFACT[detector])]
            out.append(pack([u for u, _ in rows], [labels[u] for u, _ in rows], [p for _, p in rows]))
        return out
    if detector == "selfcheck":
        return [pack([u for u, _ in d["sgm"]], [labels[u] for u, _ in d["sgm"]], [s for _, s in d["sgm"]])]
    # Semantic Entropy: per question, hallucinated if any of its answers is
    return [pack([g[0] for g, _ in d["sem"]], [max(labels[u] for u in g) for g, _ in d["sem"]],
                 [s for _, s in d["sem"]])]


def filter_run(run, keep):
    mask = np.array([keep(u) for u in run[0]], dtype=bool)
    return tuple(a[mask] for a in run)


def mean_auroc(rs):
    values = [m.auroc(y, s) for _, _, y, s in rs]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def en_reference(en_runs, pair_set):
    """Spec item 6: English scored items paired with L, or all of them if a class has under 30."""
    paired = [filter_run(r, lambda u: u in pair_set) for r in en_runs]
    ok = all(min(r[2].sum(), len(r[2]) - r[2].sum()) >= MIN_CLASS for r in paired)
    return (paired, "paired") if ok else (en_runs, "full")


def inputs(setting, lang, detector, lab_l=None, lab_en=None, keep_l=None, keep_en=None):
    """Everything one TPHR cell needs; keep_l / keep_en filter items (relation, generator, answer language)."""
    d, en = DATA[(setting, lang)], DATA[(setting, "en")]
    lab_l, lab_en = lab_l or d["labels"], lab_en or en["labels"]
    uids = [u for u in d["labels"] if keep_l is None or keep_l(u)]
    l_runs = runs(d, detector, lab_l)
    en_runs = runs(en, detector, lab_en)
    if keep_l:
        l_runs = [filter_run(r, keep_l) for r in l_runs]
    if keep_en:
        en_runs = [filter_run(r, keep_en) for r in en_runs]
    y_l = np.array([lab_l[u] for u in uids])
    if setting == "xfactr":
        pairs = [PAIR[u] for u in uids]
        y_en = np.array([lab_en[p] for p in pairs])
        en_runs, ref = en_reference(en_runs, set(pairs))
        return y_en, y_l, en_runs, l_runs, True, ref
    y_en = np.array([lab_en[u] for u in en["labels"] if keep_en is None or keep_en(u)])
    return y_en, y_l, en_runs, l_runs, False, "own"


def point_cell(y_en, y_l, en_runs, l_runs) -> dict:
    acc_en, acc_l = m.accuracy(y_en), m.accuracy(y_l)
    au_en, au_l = mean_auroc(en_runs), mean_auroc(l_runs)
    d_acc = m.deltas(acc_en, acc_l)
    d_au = None if au_en is None or au_l is None else m.deltas(au_en, au_l)
    value, marker = (None, "undefined") if d_au is None else m.tphr(d_acc, d_au)
    return {"acc_en": acc_en, "acc_l": acc_l, "d_acc": d_acc, "auroc_en": au_en, "auroc_l": au_l,
            "d_auroc": d_au, "tphr": value, "marker": marker}


def compare(y_en, y_l, en_runs, l_runs, paired, rng) -> dict:
    """Point cell plus the bootstrap interval of TPHR and the pooled dAUROC resamples."""
    cell = point_cell(y_en, y_l, en_runs, l_runs)
    acc_en_b, acc_l_b = m.boot_accuracy(y_en, y_l, rng, paired)
    d_acc_b = np.tile(m.deltas(acc_en_b, acc_l_b), len(en_runs))
    d_au_b = np.concatenate([m.boot_auroc(e[2], e[3], rng) - m.boot_auroc(l[2], l[3], rng)
                             for e, l in zip(en_runs, l_runs)])
    low, high, share, unstable = m.interval(m.tphr_array(d_acc_b, d_au_b))
    cell.update(ci_low=low, ci_high=high, undefined_share=share, unstable=unstable, d_auroc_b=d_au_b)
    return cell


def auroc_row(setting, lang, detector, rs, rng) -> dict:
    boots = np.concatenate([m.boot_auroc(r[2], r[3], rng) for r in rs])
    boots = boots[~np.isnan(boots)]
    low, high = np.percentile(boots, [2.5, 97.5]) if len(boots) else (np.nan, np.nan)
    return {"setting": setting, "language": lang, "detector": detector, "n_scored": len(rs[0][2]),
            "n_pos": int(rs[0][2].sum()), "auroc": mean_auroc(rs), "ci_low": low, "ci_high": high}


def tphr_class(value, marker, range_low) -> str:
    """Coarse position of a point TPHR, to see whether a robustness check could flip the decision."""
    if marker:
        return "undefined"
    if value <= 0:
        return "closes"
    return "below_range" if range_low is not None and value < range_low else "in_or_above_range"


def has_script(text, script) -> bool:
    prefixes = tuple(SCRIPT_CHARS[script])
    return any(unicodedata.name(ch, "").startswith(prefixes) for ch in text)


def verdict(changes, because) -> str:
    return f"{'yes' if changes else 'no'}: {because}"


def consensus(decisions) -> str:
    counts = Counter(d for d in decisions if d != "undefined")
    return counts.most_common(1)[0][0] if counts else "undefined"


def build_main(info):
    """accuracy, auroc, tphr and decision tables; returns them with the per-cell dAUROC resamples."""
    stem = {lang: stem_labels(lang) for lang in ORDER["xfactr"]}
    acc_rows, au_rows, tphr_rows, boots = [], [], [], {}
    for setting, languages in ORDER.items():
        for lang in languages:
            d = DATA[(setting, lang)]
            row = {"setting": setting, "language": lang, "n": len(d["labels"]),
                   "accuracy": m.accuracy(list(d["labels"].values()))}
            if setting == "xfactr":
                row["accuracy_stem"] = m.accuracy(list(stem[lang].values()))
                if lang != "en":
                    row["en_paired_accuracy"] = m.accuracy([DATA[(setting, "en")]["labels"][PAIR[u]] for u in d["labels"]])
            acc_rows.append(row)
            for det in DETECTORS:
                rng = np.random.default_rng(m.SEED)
                au_rows.append(auroc_row(setting, lang, det, runs(d, det, d["labels"]), rng))
                if lang == "en":
                    continue
                y_en, y_l, en_runs, l_runs, paired, ref = inputs(setting, lang, det)
                cell = compare(y_en, y_l, en_runs, l_runs, paired, np.random.default_rng(m.SEED))
                boots[(setting, lang, det)] = cell.pop("d_auroc_b")
                tphr_rows.append({"setting": setting, "language": lang, "detector": det, **cell, "en_ref": ref})
    tphr = pd.DataFrame(tphr_rows)
    ref_rows = tphr[(tphr.setting == "mtrex") & tphr.marker.isna()]
    ranges = {}
    for det in DETECTORS:
        v = ref_rows[ref_rows.detector == det].tphr
        ranges[det] = (v.min(), v.max()) if len(v) else None
    decisions = []
    for r in tphr.itertuples():
        rg = ranges[r.detector]
        marker = None if pd.isna(r.marker) else r.marker
        point = None if pd.isna(r.tphr) else r.tphr
        decisions.append("reference" if r.setting == "mtrex"
                         else m.decision(point, marker, (r.ci_low, r.ci_high), rg))
    decision = tphr.assign(
        mtrex_range_low=[ranges[d][0] if ranges[d] else np.nan for d in tphr.detector],
        mtrex_range_high=[ranges[d][1] if ranges[d] else np.nan for d in tphr.detector],
        decision=decisions)
    return pd.DataFrame(acc_rows), pd.DataFrame(au_rows), tphr, decision, ranges, boots


def build_spearman(acc, au, info):
    langs = [c for c in ORDER["xfactr"] if c != "en"]
    x = [info["xfactr"][c]["label_count"] for c in langs]
    rows = []
    a = acc[acc.setting == "xfactr"].set_index("language").accuracy
    rho, p, n = m.spearman(x, [a[c] for c in langs])
    rows.append({"target": "accuracy", "rho": rho, "p": p, "n": n})
    for det in DETECTORS:
        v = au[(au.setting == "xfactr") & (au.detector == det)].set_index("language").auroc
        rho, p, n = m.spearman(x, [v[c] for c in langs])
        rows.append({"target": f"auroc_{det}", "rho": rho, "p": p, "n": n})
    return pd.DataFrame(rows)


def build_groups(boots, tphr, info):
    rows = []
    for det in DETECTORS:
        for group in ("high", "medium", "low"):
            langs = [c for c in ORDER["xfactr"][1:] if info["xfactr"][c]["group"] == group]
            point = tphr[(tphr.setting == "xfactr") & (tphr.detector == det) & tphr.language.isin(langs)].d_auroc
            per_resample = np.nanmean([boots[("xfactr", c, det)] for c in langs], axis=0)
            low, high = np.percentile(per_resample, [2.5, 97.5])
            rows.append({"detector": det, "group": group, "languages": " ".join(langs),
                         "mean_d_auroc": point.mean(), "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def means_row(part):
    """TPHR of the mean drops over a set of languages (mam_fc rows)."""
    d_acc, d_au = part.d_acc.mean(), part.d_auroc.mean()
    value, marker = m.tphr(d_acc, d_au)
    return {"n_languages": len(part), "mean_d_acc": d_acc, "mean_d_auroc": d_au, "tphr_of_means": value,
            "marker": marker}


def build_bias(info, acc, au, tphr, decision, ranges):
    out, verdicts = {}, {}
    low = ranges[PRIMARY][0] if ranges[PRIMARY] else None
    prim = tphr[tphr.detector == PRIMARY]
    dprim = decision[decision.detector == PRIMARY]
    xf = dprim[dprim.setting == "xfactr"].copy()
    xf["group"] = [info["xfactr"][c]["group"] for c in xf.language]

    rows = []
    for g in ("high", "medium", "low"):
        part = prim[(prim.setting == "xfactr") & prim.language.isin(xf[xf.group == g].language)]
        rows.append({"group": g, **means_row(part), "decisions": " ".join(xf[xf.group == g].decision)})
    out["bias_resource_group"] = pd.DataFrame(rows)
    cons = {g: consensus(xf[xf.group == g].decision) for g in ("high", "medium", "low")}
    verdicts["resource_group"] = verdict(len(set(cons.values())) > 1, "decision per group " + str(cons))

    rows = []
    for setting in ORDER:
        for cls in ("Latin", "non-Latin"):
            langs = [c for c in ORDER[setting][1:] if (info[setting][c]["script"] == "Latin") == (cls == "Latin")]
            part = prim[(prim.setting == setting) & prim.language.isin(langs)]
            dec = dprim[(dprim.setting == setting) & dprim.language.isin(langs)].decision
            rows.append({"setting": setting, "script_class": cls, "languages": " ".join(langs),
                         **means_row(part), "decisions": " ".join(dec)})
    script = pd.DataFrame(rows)
    out["bias_script"] = script
    cons = {}
    for setting in ("xfactr", "psiloqa"):
        cons[setting] = {r.script_class: consensus(r.decisions.split()) for r in script[script.setting == setting].itertuples()}
    flipped = [s for s, c in cons.items() if len(set(c.values())) > 1]
    verdicts["script"] = verdict(bool(flipped), f"Latin vs non-Latin decisions {cons}")

    # leave-one-relation-out, X-FACTR, primary detector
    rows = []
    for lang in ORDER["xfactr"][1:]:
        base = prim[(prim.setting == "xfactr") & (prim.language == lang)].iloc[0]
        base_class = tphr_class(None if pd.isna(base.tphr) else base.tphr,
                                None if pd.isna(base.marker) else base.marker, low)
        values, classes = [], []
        relations = sorted({REL[u] for u in DATA[("xfactr", lang)]["labels"]})
        for rel in relations:
            keep = lambda u, rel=rel: REL[u] != rel
            c = point_cell(*inputs("xfactr", lang, PRIMARY, keep_l=keep, keep_en=keep)[:4])
            values.append(c["tphr"])
            classes.append(tphr_class(c["tphr"], c["marker"], low))
        defined = [v for v, c in zip(values, classes) if c != "undefined"]
        rows.append({"language": lang, "tphr_full": base.tphr, "class_full": base_class,
                     "tphr_min": min(defined) if defined else np.nan, "tphr_max": max(defined) if defined else np.nan,
                     "n_relations": len(relations), "n_class_changes": sum(c != base_class for c in classes),
                     "flips_decision": any(c != base_class for c in classes)})
    loro = pd.DataFrame(rows)
    out["bias_relation_xfactr_loro"] = loro
    verdicts["relation_xfactr"] = verdict(
        bool(loro.flips_decision.any()),
        f"{int(loro.n_class_changes.sum())} of {int(loro.n_relations.sum())} dropped-relation cases change a TPHR class "
        f"(languages: {' '.join(loro[loro.flips_decision].language) or 'none'})")

    # mTREx per category (also feeds the paper comparison)
    rows, cat_en = [], {}
    for cat in load_config("data")["mtrex"]["categories"]:
        keep = lambda u, cat=cat: REL[u] == cat
        for lang in ORDER["mtrex"]:
            y_en, y_l, en_runs, l_runs, *_ = inputs("mtrex", lang, PRIMARY, keep_l=keep, keep_en=keep)
            if lang == "en":
                rows.append({"category": cat, "language": "en", "acc_l": m.accuracy(y_l), "auroc_l": mean_auroc(l_runs)})
            else:
                rows.append({"category": cat, "language": lang, **point_cell(y_en, y_l, en_runs, l_runs)})
    mt = pd.DataFrame(rows)
    out["bias_relation_mtrex"] = mt
    sub = mt[mt.language != "en"]
    verdicts["relation_mtrex"] = verdict(
        bool((sub.tphr.dropna() <= 0).any()),
        f"mTREx TPHR by category ranges {sub.tphr.min():.2f} to {sub.tphr.max():.2f}")

    # PsiloQA generator
    rows = []
    for lang in ORDER["psiloqa"]:
        d = DATA[("psiloqa", lang)]
        for model in sorted({MODEL[u] for u in d["labels"]}):
            keep = lambda u, model=model: MODEL[u] == model
            rs = [filter_run(r, keep) for r in runs(d, PRIMARY, d["labels"])]
            labels = [d["labels"][u] for u in d["labels"] if keep(u)]
            rows.append({"language": lang, "model": model, "n": len(labels), "hallucination_rate": np.mean(labels),
                         "n_test": len(rs[0][2]), "n_test_pos": int(rs[0][2].sum()), "auroc_mam_fc": mean_auroc(rs)})
    gen = pd.DataFrame(rows)
    out["bias_generator_psiloqa"] = gen
    big = gen[(gen.n_test >= MIN_CLASS) & (gen.n_test_pos >= MIN_CLASS)].dropna(subset=["auroc_mam_fc"])
    spread = big.auroc_mam_fc.max() - big.auroc_mam_fc.min() if len(big) else np.nan
    verdicts["generator_psiloqa"] = verdict(
        bool(spread > 10), f"MAM AUROC spans {spread:.1f} points over {len(big)} generator cells with 30+ test items and 30+ hallucinated ones")

    # label rule, X-FACTR
    rows = []
    for lang in ORDER["xfactr"][1:]:
        lab_stem = stem_labels(lang)
        lab_en_stem = stem_labels("en")
        base = point_cell(*inputs("xfactr", lang, PRIMARY)[:4])
        alt = point_cell(*inputs("xfactr", lang, PRIMARY, lab_l=lab_stem, lab_en=lab_en_stem)[:4])
        rows.append({"language": lang, "d_acc_matches": base["d_acc"], "d_acc_stem": alt["d_acc"],
                     "auroc_l_matches": base["auroc_l"], "auroc_l_stem": alt["auroc_l"],
                     "tphr_matches": base["tphr"], "marker_matches": base["marker"],
                     "tphr_stem": alt["tphr"], "marker_stem": alt["marker"],
                     "class_changes": tphr_class(base["tphr"], base["marker"], low) != tphr_class(alt["tphr"], alt["marker"], low)})
    rule = pd.DataFrame(rows)
    out["bias_label_rule"] = rule
    verdicts["label_rule"] = verdict(bool(rule.class_changes.any()), f"{int(rule.class_changes.sum())} of {len(rule)} languages change TPHR class under the stem label")

    # answer language, non-Latin targets only
    rows = []
    for setting in ORDER:
        for lang in ORDER[setting][1:]:
            script = info[setting][lang]["script"]
            if script == "Latin":
                continue
            d = DATA[(setting, lang)]
            ok = lambda u, d=d, script=script: has_script(d["answers"][u], script)
            kept = [u for u in d["labels"] if ok(u)]
            base = point_cell(*inputs(setting, lang, PRIMARY)[:4])
            alt = point_cell(*inputs(setting, lang, PRIMARY, keep_l=ok)[:4]) if len(kept) >= 3 * MIN_CLASS else None
            rows.append({"setting": setting, "language": lang, "script": script, "n": len(d["labels"]),
                         "share_wrong_script": 1 - len(kept) / len(d["labels"]), "acc_all": base["acc_l"],
                         "acc_target_script_only": alt["acc_l"] if alt else np.nan,
                         "tphr_all": base["tphr"], "tphr_target_script_only": alt["tphr"] if alt else np.nan,
                         "class_changes": bool(alt) and tphr_class(base["tphr"], base["marker"], low) != tphr_class(alt["tphr"], alt["marker"], low)})
    lang_tab = pd.DataFrame(rows)
    out["bias_answer_language"] = lang_tab
    verdicts["answer_language"] = verdict(
        bool(lang_tab.class_changes.any()),
        f"share of answers outside the target script is {lang_tab.share_wrong_script.min():.2f} to {lang_tab.share_wrong_script.max():.2f}; "
        f"{int(lang_tab.class_changes.sum())} of {len(lang_tab)} languages change TPHR class without them")

    # class balance
    rows = []
    for setting, languages in ORDER.items():
        for lang in languages:
            d = DATA[(setting, lang)]
            labels = list(d["labels"].values())
            test = runs(d, PRIMARY, d["labels"])[0][2]
            minority = int(min(test.sum(), len(test) - test.sum()))
            rows.append({"setting": setting, "language": lang, "n": len(labels), "n_pos": int(sum(labels)),
                         "hallucination_rate": np.mean(labels), "n_test_seed42": len(test),
                         "test_minority": minority, "thin": minority < MIN_CLASS})
    bal = pd.DataFrame(rows)
    out["bias_class_balance"] = bal
    thin = bal[bal.thin]
    verdicts["class_balance"] = verdict(
        bool(len(thin)), f"{len(thin)} cells have under {MIN_CLASS} minority-class test items: "
        + (" ".join(f"{r.setting}/{r.language}" for r in thin.itertuples()) or "none"))
    return out, verdicts


def build_paper(tphr, acc, au, mt):
    """Our numbers against the base paper's (src/analysis/paper_numbers.csv); None if that file is absent."""
    if not PAPER_CSV.exists():
        print(f"paper_comparison skipped: {PAPER_CSV} not found (re-run when it exists)")
        return None
    paper = pd.read_csv(PAPER_CSV, keep_default_na=False)

    def number(value):
        try:
            return float(value)
        except ValueError:
            return value  # 'UN' / 'NA'
    lookup = {(r.category.lower(), r.language, r.metric): number(r.value) for r in paper.itertuples()}

    def paper_value(category, lang, metric):
        names = [category] + (["all", "average", "avg", "mean", "overall"] if category == "all" else [])
        return next((lookup[(n.lower(), lang, metric)] for n in names if (n.lower(), lang, metric) in lookup), None)

    def paper_tphr(category, lang):
        """The paper prints no TPHR table, so it is computed from its own accuracy and AUROC tables."""
        acc_en, acc_l = paper_value(category, "en", "accuracy"), paper_value(category, lang, "accuracy")
        au_en, au_l = paper_value(category, "en", "auroc_mam_fc"), paper_value(category, lang, "auroc_mam_fc")
        if lang == "en" or not all(isinstance(v, float) for v in (acc_en, acc_l, au_en, au_l)):
            return None
        value, marker = m.tphr(acc_en - acc_l, au_en - au_l)
        return marker or value

    rows = []
    for r in mt.itertuples():
        marker = None if pd.isna(getattr(r, "marker", np.nan)) else r.marker
        ours = None if r.language == "en" else (r.marker if marker else r.tphr)
        rows.append({"category": r.category, "language": r.language, "our_accuracy": r.acc_l,
                     "paper_accuracy": paper_value(r.category, r.language, "accuracy"),
                     "our_auroc_mam_fc": r.auroc_l, "paper_auroc_mam_fc": paper_value(r.category, r.language, "auroc_mam_fc"),
                     "our_tphr": ours, "paper_tphr": paper_tphr(r.category, r.language)})
    pooled = tphr[(tphr.setting == "mtrex") & (tphr.detector == PRIMARY)].set_index("language")
    for lang in ORDER["mtrex"]:
        a = acc[(acc.setting == "mtrex") & (acc.language == lang)].accuracy.iloc[0]
        u = au[(au.setting == "mtrex") & (au.language == lang) & (au.detector == PRIMARY)].auroc.iloc[0]
        ours = None if lang == "en" else (pooled.loc[lang, "marker"] if pd.notna(pooled.loc[lang, "marker"]) else pooled.loc[lang, "tphr"])
        rows.append({"category": "all", "language": lang, "our_accuracy": a,
                     "paper_accuracy": paper_value("all", lang, "accuracy"), "our_auroc_mam_fc": u,
                     "paper_auroc_mam_fc": paper_value("all", lang, "auroc_mam_fc"), "our_tphr": ours,
                     "paper_tphr": paper_value("all", lang, "tphr")})
    return pd.DataFrame(rows)


def write(df, name):
    df.round(4).to_csv(TABLES / f"{name}.csv", index=False)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    info = load_all()
    acc, au, tphr, decision, ranges, boots = build_main(info)
    spearman, groups = build_spearman(acc, au, info), build_groups(boots, tphr, info)
    bias, verdicts = build_bias(info, acc, au, tphr, decision, ranges)
    paper = build_paper(tphr, acc, au, bias["bias_relation_mtrex"])
    for name, df in {"accuracy": acc, "auroc": au, "tphr": tphr, "decision": decision, "spearman": spearman,
                     "detectors_by_group": groups, **bias}.items():
        write(df, name)
    if paper is not None:
        write(paper, "paper_comparison")
    write(pd.DataFrame({"slice": list(verdicts), "verdict": list(verdicts.values())}), "bias_verdicts")

    print("decisions (setting, detector):")
    print(decision.groupby(["setting", "detector"]).decision.value_counts().unstack(fill_value=0).to_string())
    print("mTREx TPHR range per detector:", {k: None if v is None else tuple(round(x, 3) for x in v) for k, v in ranges.items()})
    print(spearman.round(3).to_string(index=False))
    for k, v in verdicts.items():
        print(f"bias {k}: {v}")


if __name__ == "__main__":
    main()
