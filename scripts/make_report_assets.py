"""Rebuilds every EDA table, figure and the findings list in data/eda/ from the raw and processed data.

With --phase4 it instead draws the Phase 4 figures in results/figures/ from results/tables/ (run scripts/run_analysis.py first).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.io import ROOT, load_config
from src.common.plots import ACCENT, DARK, GREY, GROUP_COLOURS, new_axes, save
from src.common.schema import read_jsonl
from src.datasets import xfactr

EDA = ROOT / "data" / "eda"
RAW_MTREX = ROOT / "data" / "raw" / "mtrex_repo" / "data" / "mTREx"


def frame(name):
    rows = [{**r.__dict__, "n_gold": len(r.gold), "n_spans": len(r.spans),
             "prompt_chars": len(r.prompt), "answer_chars": len(r.answer),
             "span_chars": sum(s["end"] - s["start"] for s in r.spans)}
            for r in read_jsonl(ROOT / "data" / "processed" / f"{name}.jsonl")]
    return pd.DataFrame(rows)


def mtrex(cfg, findings):
    df = frame("mtrex")
    raw = pd.DataFrame([
        {"language": code, "category": cat,
         "raw_rows": len(pd.read_csv(RAW_MTREX / folder / f"{cat}.csv", dtype=str, keep_default_na=False))}
        for folder, code in cfg["mtrex"]["languages"].items() for cat in cfg["mtrex"]["categories"]])
    kept = (df.groupby(["language", "relation"]).size().rename("sampled_rows").reset_index()
            .rename(columns={"relation": "category"}))
    counts = raw.merge(kept)
    counts.to_csv(EDA / "mtrex_rows.csv", index=False)
    stats = df.groupby("language").agg(
        mean_accepted_answers=("n_gold", "mean"), multi_answer_share=("n_gold", lambda s: (s > 1).mean()),
        mean_question_chars=("prompt_chars", "mean")).round(2)
    stats.to_csv(EDA / "mtrex_answers.csv")
    fig, ax = new_axes(width=6.5, height=2.7)
    counts.pivot(index="language", columns="category", values="raw_rows").plot.bar(
        ax=ax, color=[DARK, GREY, ACCENT], width=0.8)
    ax.axhline(counts.sampled_rows.iloc[0], color="black", lw=0.8, ls="--")
    ax.set(ylabel="raw rows", xlabel="", title="mTREx: raw rows per language and category (dashed = sample size)")
    save(fig, EDA / "mtrex_rows.png")
    smallest = counts.loc[counts.raw_rows.idxmin()]
    findings.append(
        f"mTREx: raw files range from {counts.raw_rows.min()} rows ({smallest.language}/{smallest.category}) to "
        f"{counts.raw_rows.max()}; the smallest cell after cleaning fixes the sample at "
        f"{counts.sampled_rows.iloc[0]} rows per language and category.")
    findings.append("mTREx: share of questions with more than one accepted answer by language: "
                    + ", ".join(f"{l} {v:.0%}" for l, v in stats.multi_answer_share.items()) + ".")


def xfactr_eda(cfg, findings):
    chosen = {l["code"]: l for l in load_config("languages")["languages"] if l["chosen"]}
    df = frame("xfactr")
    non_en = df[df.language != "en"]
    _, _, _, facts, eligible = xfactr.eligible_facts(cfg, log=lambda *_: None)
    per_relation = pd.Series([f["pid"] for f in facts]).value_counts()
    per_relation.rename("facts").to_csv(EDA / "xfactr_facts_per_relation.csv")
    rows = []
    for code, info in chosen.items():
        d = non_en[non_en.language == code]
        rows.append({"language": code, "group": info["group"], "script": info["script"],
                     "label_count": info["label_count"], "eligible_facts": len(eligible[code]),
                     "sampled_facts": len(d), "mean_accepted_answers": round(d.n_gold.mean(), 2),
                     "median_gold_chars": d.gold.map(lambda g: len(g[0])).median(), "relations": d.relation.nunique()})
    table = pd.DataFrame(rows)
    table.to_csv(EDA / "xfactr_languages.csv", index=False)
    fig, axes = new_axes(2, width=4.2)
    colours = [GROUP_COLOURS[g] for g in table.group]
    axes[0].bar(table.language, table.label_count, color=colours)
    axes[0].set(yscale="log", ylabel="entity labels (log)", title="X-FACTR: label coverage")
    axes[1].bar(table.language, table.mean_accepted_answers, color=colours)
    axes[1].set(ylabel="accepted answers per fact", title="Aliases per gold answer")
    save(fig, EDA / "xfactr_coverage.png")
    hi, lo = table.loc[table.label_count.idxmax()], table.loc[table.label_count.idxmin()]
    low_counts = pd.Series([pid for pid, _ in eligible[lo.language]]).value_counts()
    findings.append(
        f"X-FACTR: every relation starts with {per_relation.iloc[0]} facts, but after label filtering {lo.language} keeps "
        f"{low_counts.min()} to {low_counts.max()} per relation (median {int(low_counts.median())}) and "
        f"{len(per_relation) - len(low_counts)} relations disappear, hence the cap of {cfg['xfactr']['cap_per_relation']}.")
    findings.append(
        f"X-FACTR: entity labels fall from {hi.label_count} ({hi.language}) to {lo.label_count} ({lo.language}); "
        "accepted answers per fact: " + ", ".join(f"{r.language} {r.mean_accepted_answers}" for r in table.itertuples()) + ".")
    findings.append(
        f"X-FACTR: {(df.language == 'en').sum()} distinct English facts back {len(non_en)} non-English records, "
        "so the English side is small and shared.")


def psiloqa(cfg, findings):
    df = frame("psiloqa")
    df["model"] = df.meta.map(lambda m: m["model"])
    df["span_share"] = df.span_chars / df.answer_chars
    per_lang = df.groupby("language").agg(
        items=("uid", "size"), hallucination_rate=("label", "mean"), mean_answer_chars=("answer_chars", "mean"),
        mean_answer_words=("answer", lambda s: s.str.split().str.len().mean()),
        mean_spans=("n_spans", "mean"), mean_span_share=("span_share", "mean")).round(3)
    per_lang["correct_answers"] = (per_lang["items"] * (1 - per_lang.hallucination_rate)).round().astype(int)
    per_lang.to_csv(EDA / "psiloqa_languages.csv")
    df.groupby(["language", "model"]).size().rename("items").to_csv(EDA / "psiloqa_models.csv")
    hallucinated = df[df.label == 1]
    fig, axes = new_axes(2, width=4.2)
    axes[0].bar(per_lang.index, per_lang.hallucination_rate, color=DARK)
    axes[0].set(ylim=(0, 1.1), ylabel="hallucination rate", title="PsiloQA: share of answers with a span")
    for i, (rate, n) in enumerate(zip(per_lang.hallucination_rate, per_lang.correct_answers)):
        axes[0].text(i, rate + 0.02, f"{n} ok", ha="center", fontsize=8)
    axes[1].hist(hallucinated.span_share, bins=20, color=GREY)
    axes[1].axvline(hallucinated.span_share.median(), color=ACCENT)
    axes[1].set(xlabel="share of answer covered by spans", ylabel="answers", title="Span coverage (line = median)")
    save(fig, EDA / "psiloqa_spans.png")
    findings.append(
        "PsiloQA: hallucination rate per language: "
        + ", ".join(f"{l} {r:.2f}" for l, r in per_lang.hallucination_rate.items())
        + "; correct answers out of 1,000: " + ", ".join(f"{l} {n}" for l, n in per_lang.correct_answers.items()) + ".")
    findings.append(
        f"PsiloQA: spans are long: the median hallucinated answer is {hallucinated.span_share.median():.0%} covered by "
        f"spans, with {hallucinated.n_spans.mean():.1f} spans per hallucinated answer on average.")
    findings.append("PsiloQA: generator models differ by language (psiloqa_models.csv), so language and generator effects are confounded.")


TABLES, FIGURES = ROOT / "results" / "tables", ROOT / "results" / "figures"
SETTING_NAMES = {"mtrex": "mTREx", "xfactr": "X-FACTR", "psiloqa": "PsiloQA"}
DETECTOR_STYLE = {"mam_fc": (ACCENT, "o"), "mam_attn": (DARK, "s"), "selfcheck": (GREY, "^"), "sem": ("#cfcfcf", "D")}
GROUP_SHADE = {"high": "#ececec", "medium": "#f6f6f6", "low": "#e0e0e0"}


def table(name):
    # keep_default_na=False so the paper marker "NA" is not read as missing
    return pd.read_csv(TABLES / f"{name}.csv", keep_default_na=False, na_values=[""])


def fig_acc_vs_auroc(acc, au):
    fig, axes = new_axes(3, width=2.1, height=2.4)
    for ax, (setting, name) in zip(axes, SETTING_NAMES.items()):
        a = acc[acc.setting == setting].reset_index(drop=True)
        u = au[(au.setting == setting) & (au.detector == "mam_fc")].set_index("language").loc[a.language]
        x = range(len(a))
        ax.bar([i - 0.2 for i in x], a.accuracy, 0.4, color=GREY, label="Accuracy")
        ax.bar([i + 0.2 for i in x], u.auroc, 0.4, color=DARK, label="MAM AUROC",
               yerr=[u.auroc - u.ci_low, u.ci_high - u.auroc], error_kw={"lw": 0.7, "capsize": 1.5})
        ax.set_xticks(list(x), a.language, fontsize=7)
        ax.set(ylim=(0, 100), xlabel=name)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("points")
    axes[2].legend(fontsize=6.5, frameon=False, loc="upper right")
    save(fig, FIGURES / "acc_vs_auroc.png")


def fig_tphr_ci(tphr):
    fig, axes = new_axes(3, width=2.1, height=2.9)
    for ax, (setting, name) in zip(axes, SETTING_NAMES.items()):
        t = tphr[tphr.setting == setting]
        languages = list(dict.fromkeys(t.language))
        groups = {l["code"]: l["group"] for l in load_config("languages")["languages"]}
        for row, lang in enumerate(languages):
            if setting == "xfactr":
                ax.axhspan(row - 0.5, row + 0.5, color=GROUP_SHADE[groups[lang]], lw=0)
        for k, (det, (colour, marker)) in enumerate(DETECTOR_STYLE.items()):
            for row, lang in enumerate(languages):
                r = t[(t.language == lang) & (t.detector == det)].iloc[0]
                y = row + (k - 1.5) * 0.15
                defined = pd.isna(r.marker)
                if pd.notna(r.tphr):
                    ax.plot(r.tphr, y, marker, ms=3.5, color=colour, mfc=colour if defined else "white", mew=0.8,
                            label=det if row == 0 else None)
                if defined and pd.notna(r.ci_low):
                    ax.plot([r.ci_low, r.ci_high], [y, y], color=colour, lw=0.7)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_yticks(range(len(languages)), languages, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(f"{name}: TPHR", fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
    axes[0].legend(fontsize=5.5, frameon=False, loc="lower right")
    save(fig, FIGURES / "tphr_ci.png")


def fig_labels_scatter(acc, au, spearman):
    fig, axes = new_axes(2, width=3.1, height=2.5)
    langs = {l["code"]: l for l in load_config("languages")["languages"] if l["chosen"]}
    a = acc[(acc.setting == "xfactr") & (acc.language != "en")].set_index("language")
    u = au[(au.setting == "xfactr") & (au.detector == "mam_fc") & (au.language != "en")].set_index("language")
    for ax, values, target, label in ((axes[0], a.accuracy, "accuracy", "Accuracy (points)"),
                                      (axes[1], u.auroc, "auroc_mam_fc", "MAM AUROC (points)")):
        for lang, v in values.items():
            ax.scatter(langs[lang]["label_count"], v, color=GROUP_COLOURS[langs[lang]["group"]], s=18)
            ax.annotate(lang, (langs[lang]["label_count"], v), textcoords="offset points", xytext=(3, 3), fontsize=7)
        rho = spearman.set_index("target").loc[target]
        ax.set(xscale="log", xlabel="entity labels (log)", ylabel=label)
        ax.set_title(f"Spearman rho {rho.rho:.2f}, p {rho.p:.2f}, n {int(rho.n)}", fontsize=7.5)
    save(fig, FIGURES / "labels_scatter.png")


def fig_bias_summary(loro, rule, gen, balance):
    fig, axes = new_axes(3, width=2.1, height=2.5)
    ax = axes[0]
    rule = rule.set_index("language")
    for i, r in enumerate(loro.itertuples()):
        if pd.notna(r.tphr_min):
            ax.plot([r.tphr_min, r.tphr_max], [i, i], color=GREY, lw=2)
        if pd.notna(r.tphr_full):
            ax.plot(r.tphr_full, i, "o", color=ACCENT, ms=3.5)
        if pd.notna(rule.loc[r.language, "tphr_stem"]):
            ax.plot(rule.loc[r.language, "tphr_stem"], i, "x", color=DARK, ms=3.5)
    ax.set_yticks(range(len(loro)), loro.language, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("X-FACTR TPHR: full (dot),\nleave-one-relation-out (bar),\nstem label (x)", fontsize=6.5)
    ax.tick_params(axis="x", labelsize=7)
    ax = axes[1]
    g = gen.dropna(subset=["auroc_mam_fc"])
    languages = list(dict.fromkeys(gen.language))
    ax.scatter([languages.index(l) for l in g.language], g.auroc_mam_fc, s=g.n_test.clip(upper=100) / 4 + 4, color=DARK)
    ax.axhline(50, color="black", lw=0.6)
    ax.set_xticks(range(len(languages)), languages, fontsize=7)
    ax.set(ylabel="MAM AUROC per generator", ylim=(0, 100))
    ax.set_xlabel("PsiloQA (dot size = test items)", fontsize=6.5)
    ax.tick_params(axis="y", labelsize=7)
    ax = axes[2]
    labels = [f"{s[0].upper()}-{l}" for s, l in zip(balance.setting, balance.language)]
    ax.bar(range(len(balance)), balance.test_minority.clip(lower=1), color=[GREY if not t else ACCENT for t in balance.thin])
    ax.axhline(30, color="black", lw=0.6, ls="--")
    ax.set(yscale="log", ylabel="minority test items")
    ax.set_xticks(range(len(balance)), labels, rotation=90, fontsize=5.5)
    ax.tick_params(axis="y", labelsize=7)
    save(fig, FIGURES / "bias_summary.png")


def phase4_figures():
    FIGURES.mkdir(parents=True, exist_ok=True)
    acc, au, tphr = table("accuracy"), table("auroc"), table("tphr")
    fig_acc_vs_auroc(acc, au)
    fig_tphr_ci(tphr)
    fig_labels_scatter(acc, au, table("spearman"))
    fig_bias_summary(table("bias_relation_xfactr_loro"), table("bias_label_rule"),
                     table("bias_generator_psiloqa"), table("bias_class_balance"))
    print(f"wrote 4 figures to {FIGURES}")


def main():
    if "--phase4" in sys.argv:
        return phase4_figures()
    cfg = load_config("data")
    EDA.mkdir(exist_ok=True)
    findings = []
    mtrex(cfg, findings)
    xfactr_eda(cfg, findings)
    psiloqa(cfg, findings)
    text = "# EDA findings (generated by scripts/make_report_assets.py)\n\n" + "\n".join(f"- {f}" for f in findings) + "\n"
    (EDA / "findings.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
