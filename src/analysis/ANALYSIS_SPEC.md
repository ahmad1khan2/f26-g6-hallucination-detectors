# Analysis specification (Phase 4)

Written and committed before any Phase 4 metric was computed. Analysis code
reads `results/<dataset>/<lang>/` and `data/processed/` only and writes
`results/tables/` and `results/figures/`. If a choice here changes later, the
change and its reason are logged at the bottom and both versions are reported.

## Inputs

- Labels: `answers.jsonl` (`label`, 1 = hallucinated). X-FACTR also gets a
  second label, `label_stem` = 0 if `matches()` or `stem_match()` accepts the
  answer (`src/common/text.py`), computed from the processed gold.
- MAM: `mam_preds_seed{42,43,44}.jsonl`, test split only (`split == "test"`).
- SelfCheckGPT: `sgm.jsonl` `score`. Semantic Entropy: `sem.jsonl`
  `entropies.semantic_entropy`; for PsiloQA it is per question, labelled 1 if
  any of its answers (`uids`) is hallucinated.
- Resource group, script and entity-label count: `configs/languages.yaml`.

## Choices

1. **Pooling.** mTREx is pooled over its three categories per language for the
   decision rule, and reported per category for the paper comparison. X-FACTR
   is pooled over relations. PsiloQA is pooled.
2. **Accuracy** = share of label 0 over all sampled items of a language, in
   points (x100). The same accuracy is used with every detector.
3. **AUROC** = `sklearn.metrics.roc_auc_score`, hallucinated = positive class,
   higher score = more likely hallucinated (as in the authors' code), x100.
   A split with one class gives no AUROC.
4. **MAM artifact.** Primary: last-layer fully connected activations
   (`fully_connected`). Secondary: last-layer attention outputs (`attention`).
5. **MAM point estimate** = mean over seeds 42, 43, 44 of the per-seed test
   AUROC.
6. **English reference.** mTREx and PsiloQA use their own English items.
   X-FACTR: Acc(EN) uses the English items paired with language L's items
   (`pair_uid`); AUROC(EN) uses the English scored items (test split for MAM)
   paired with L. If either class has fewer than 30 such items, the full
   English scored set (test split for MAM) is used and the row says so
   (`en_ref = full`).
7. **Deltas.** dAcc = Acc(EN) - Acc(L), dAUROC = AUROC(EN) - AUROC(L), points.
8. **TPHR** = log10(|dAcc| / |dAUROC|), as in the base paper. Paper markers:
   `UN` when dAcc = 0, `NA` when dAUROC = 0. Our proposal's marker
   `undefined` applies when dAcc < 5 or dAUROC <= 0; both drops are still
   reported, and undefined rows get no decision.
9. **Bootstrap.** 1,000 resamples, seed 42, items resampled with replacement
   within English and within L. X-FACTR accuracy resamples facts as pairs
   (an L item together with its English pair). For MAM each seed's test split
   is resampled and the 3 x 1,000 values are pooled. Intervals are the 2.5th
   to 97.5th percentiles; for TPHR only over resamples where it is defined.
   The share of undefined resamples is reported, and an interval is called
   `unstable` when that share is over 10%.
10. **mTREx range** per detector = [min, max] of the point TPHR over German,
    Hindi, Bengali and Urdu (defined values only).
11. **Decision rule** (proposal rule, unchanged; two clarifications written
    here before any result): undefined gives no decision (`undefined`);
    point TPHR <= 0 gives `closes`; an interval that overlaps the mTREx range
    or lies above it gives `holds`; an interval entirely below the range with
    its lower bound above 0 gives `narrows`; anything else is `inconclusive`.
    Applied to X-FACTR and PsiloQA rows; mTREx rows are the `reference`.
12. **Spearman.** Across the seven non-English X-FACTR languages, entity-label
    count against accuracy, and against AUROC for each detector. Report rho,
    p and n, and note that n is small.
13. **Detector comparison.** dAUROC per detector averaged over the languages
    of each X-FACTR resource group (high, medium, low), with a bootstrap
    interval from the per-resample group means.
14. **Sampling detectors** were run on 91 items per language (cut order step
    1). Their AUROC intervals are wide and are reported as they are.

## Bias check slices

Each slice gets a table and a one-line verdict "does the conclusion change:
yes or no, because ...".

- resource group (X-FACTR)
- script (Latin against non-Latin targets, all settings)
- relation: leave-one-relation-out TPHR for X-FACTR (primary MAM), per
  category for mTREx
- PsiloQA generator model (`meta.model`): hallucination rate and MAM AUROC
- label rule: `matches` against `label_stem` (X-FACTR)
- answer language: share of answers with no character in the target script
  (non-Latin targets only; for Latin targets the script cannot tell the
  language apart), and accuracy with those answers removed
- class balance per language

## Paper comparison

Our mTREx per-category accuracy and MAM fully connected AUROC against the base
paper's LLaMA-8B numbers (accuracy: Table 8; AUROC: appendix tables).
Pattern check: accuracy falls, AUROC holds, TPHR positive.

## Change log

- (none)
