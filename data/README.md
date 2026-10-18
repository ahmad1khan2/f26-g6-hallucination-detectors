# Data card

Sources, licences, fields, decisions and known issues for the three datasets.
Raw data is rebuilt by `bash scripts/download_data.sh` (git-ignored under `data/raw/`).

## Raw sources (Task 6, downloaded 2026-10-18)

| Dataset | Source | Version pinned | Licence |
|---|---|---|---|
| mTREx | https://github.com/aisoc-lab/low-resource-hallucination-detection, `data/mTREx/` | commit 644d655 (same as `third_party/` submodule) | MIT |
| X-FACTR | https://github.com/jzbjyb/x-factr, `data/` | commit c1938ec | check repo (data derived from Wikidata/Wikipedia) |
| PsiloQA | https://huggingface.co/datasets/s-nlp/PsiloQA | `load_dataset("s-nlp/PsiloQA")` on 2026-10-18 | check dataset card |

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

## Decision log

(none yet)
