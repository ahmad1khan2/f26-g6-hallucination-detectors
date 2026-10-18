# Data card

Sources, licences, fields, decisions and known issues for the three datasets.
Raw data is rebuilt by `bash scripts/download_data.sh` (git-ignored under `data/raw/`).

## Raw sources (Task 6, downloaded 2026-10-18)

| Dataset | Source | Version pinned | Licence |
|---|---|---|---|
| mTREx | https://github.com/aisoc-lab/low-resource-hallucination-detection, `data/mTREx/` | commit 644d655 (same as `third_party/` submodule) | MIT |
| X-FACTR | https://github.com/jzbjyb/x-factr, `data/` | commit c1938ec | no licence file in the repository; data derived from Wikidata and Wikipedia |
| PsiloQA | https://huggingface.co/datasets/s-nlp/PsiloQA | `load_dataset("s-nlp/PsiloQA")` on 2026-10-18, hub revision 375c332 | CC BY 4.0 (dataset card) |

Commit hashes: `data/raw/mtrex_repo` = 644d6551e16c090a1ae3290c1d0e9635fcc157c7, `data/raw/xfactr` = c1938ec7262ebc34ac0b39daff3f446e103e8a87.

### mTREx structure
- Folders `English, Deutsch, Hindi, Bengali, Urdu`, each with `capitals.csv`, `country.csv`, `official_language.csv`; columns `source,target`; `target` may hold several answers separated by ` <OR> `.
- Rows per file (raw):

| Language | capitals | country | official_language |
|---|---|---|---|
| English | 10000 | 10000 | 2375 |
| Deutsch | 10000 | 10000 | 2375 |
| Hindi | 2500 | 2500 | 2374 |
| Bengali | 2500 | 2500 | 2375 |
| Urdu | 2500 | 2500 | 2375 |

- Smallest file is Hindi `official_language` (2374 rows): this bounds the equal-rows-per-language-and-category sample (Task 7).
- Authors' question prompts: `data/raw/mtrex_repo/prompts/mTREx_prompts.json` (path to confirm in Task 7).

### X-FACTR structure
- `data/TREx_prompts.csv`: 46 relations (`pid`, `relation`) x 24 languages plus English (columns `en, zh, fr, el, nl, ru, ko, es, mr, vi, he, tr, yo, ja, hu, bn, war, tl, sw, mg, pa, ilo, ceb`). Templates use `[X]` (subject) and `[Y]` (object), with inflection tags such as `[Y.Loc]` and gender alternatives such as `[né;X-Gender=MASC|née;X-Gender=FEM]`.
- `data/mTRExf/sub/P*.jsonl`: facts, 46 files (one per relation), 46,000 lines in total; fields `uuid, predicate_id, sub_uri, sub_label, obj_uri, obj_label, count` (English labels, Wikidata ids `Q...`).
- `data/alias/mTRExf/<lang>.txt`: tab-separated `entity id, label, aliases...` per language, 22 files (`en, ceb, el, es, fr, he, hu, ilo, ja, ko, mg, mr, nl, pa, ru, sw, tl, tr, vi, war, yo, zh`); `en` has 57,837 entities, `sw` 6,362. Link to facts through the entity id (`sub_uri`, `obj_uri`).
- `data/TREx_gender.txt` (and `mTRExf_gender.txt`): gender reference file, 35,131 lines.
- Languages in the template file with no alias file here: `bn` (a `bengali/` folder exists) — to check in Task 7.

### PsiloQA structure
- `DatasetDict` with splits `train` 63,792, `validation` 3,355, `test` 2,897 rows.
- Columns: `id, lang, wiki_title, wiki_url, llm_checkpoint, wiki_passage, question, golden_answer, llm_answer, annotated_span, complexity, labels`.
- Mapping to our schema: question -> prompt (with `wiki_passage` in `meta`), `llm_answer` -> answer, `labels` (list of `[start, end]` character offsets) -> spans, `llm_checkpoint` -> `meta.model`, `lang` -> language. `annotated_span` is the answer with `[HAL]...[/HAL]` markers.
- 14 languages in train: en 16115, fi 6557, hi 6201, ca 5843, sv 5234, zh 5187, cs 4665, fa 3008, es 2138, it 1999, eu 1935, ar 1857, fr 1824, de 1229. Includes English and Hindi, as required.
- 26 generator models (`llm_checkpoint`); each language is dominated by one to three of them, so the "same generator mix per language" rule needs checking in Task 7.

## Processed files (Task 7)

`python scripts/build_processed.py` writes `data/processed/{mtrex,xfactr,psiloqa}.jsonl` (shared schema, `src/common/schema.py`) and `build_log.txt` with every step count. Settings are in `configs/data.yaml` and `configs/languages.yaml`. Tests: `tests/test_processed.py`.

| File | Records | Languages |
|---|---|---|
| mtrex.jsonl | 32,100 (2,140 per language and category) | en, de, hi, bn, ur |
| xfactr.jsonl | 9,977 (8,181 non-English + 1,796 distinct English) | en + zh, ja, vi (high); tl, sw (medium); yo, ilo (low) |
| psiloqa.jsonl | 5,000 (1,000 per language) | en, hi, zh, fi, ar |

## Decision log

1. **mTREx sample size.** After dropping empty rows and duplicate sources, the smallest cell is Hindi/official_language with 2,140 rows, so every language and category keeps 2,140 rows, chosen by SHA-1 hash order (seed 42). Duplicate sources removed: e.g. German capitals 10,000 -> 9,624.
2. **mTREx prompt.** `prompt` holds the authors' user prompt for the category with the source filled in; the system prompt and answer prefix are added in Phase 3 from `configs/prompts.yaml`.
3. **X-FACTR languages.** Resource groups from alias-file label counts, boundaries at the gaps in the counts: high >= 15,000, medium 4,000-14,999, low < 4,000. Chosen from templates with no gender markers and no inflection tags that need a morphology engine: zh, ja, vi (high), tl, sw (medium), yo, ilo (low). Not in X-FACTR's alias files: bn, ur, hi, de (so X-FACTR does not overlap with mTREx languages).
4. **X-FACTR relations.** No chosen language's template contains `Gender`, so no relation is dropped: all 46 are kept (46,000 facts). The gender rule stays in the code.
5. **Punjabi dropped.** Its alias labels are in Shahmukhi (Arabic script) while its templates are Gurmukhi, so prompts mix scripts and answers could not match. The low group is yo and ilo only.
6. **X-FACTR cap.** Eligible facts per relation are very uneven (median 14-561 depending on the language, maximum 1,000), so a cap of 30 facts per relation is used. This keeps 834-1,380 facts per language, and no relation dominates.
7. **X-FACTR prompt.** Inflection tags `[Y.Loc]`, `[X.Nom]` are reduced to `[X]` / `[Y]`; `[X]` becomes the subject label and `[Y]` the blank `___`. English facts are written once and shared: each non-English record's `pair_uid` points to its English record.
8. **X-FACTR gold.** Object label plus aliases in that language, NFKC-normalised and lower-cased. Short aliases (one or two characters) can cause false matches under the substring rule; to be reviewed in the Phase 3 hand check.
9. **PsiloQA splits.** train, validation and test are pooled (no training happens on PsiloQA); 13 empty-or-duplicate rows removed in total.
10. **PsiloQA languages and size.** en, hi, zh, fi, ar, 1,000 items each (ar has only 2,072 usable rows).
11. **PsiloQA generator mix: plan deviation.** Each language's answers come from its own models (only a few models are shared: EuroLLM en/sv, Qwen2 zh/fa, gemma hi/eu), so an identical generator mix across languages is impossible. We take equal item counts and keep each language's own generator proportions (largest-remainder quotas); ar has a single generator. The bias check (Phase 4) reports results by generator model.
12. **PsiloQA class imbalance (known issue).** Hallucination rate is 0.84-0.96 per language (en 0.96), so English has about 40 correct answers in 1,000. AUROC will be unstable there; to be discussed before Phase 3.
13. **PsiloQA fields.** `prompt` is the question; the Wikipedia passage is in `meta.passage` and is added by the Phase 3 prompt. Spans are character offsets into `answer`.
