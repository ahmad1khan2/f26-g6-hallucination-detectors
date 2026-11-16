# f26-g6-hallucination-detectors

AI2002 Artificial Intelligence, Fall 2026, Track B (Research & Development), Group 6.

## Purpose

Datta et al. (EACL 2026) report that hallucination detectors lose far less performance than task accuracy when a model moves from English to low-resource languages. We ask whether this holds on a broader set of languages (X-FACTR) and on long answers with span-level hallucinations (PsiloQA). One generator, Llama-3.1-8B-Instruct in 4-bit, is used everywhere. Three detectors from the authors' released code (MAM, SelfCheckGPT, Semantic Entropy) are compared with accuracy, AUROC and TPHR, with bootstrap intervals and a decision rule fixed in advance.

Base paper: Datta, Chilukuri, Kumar, Ghosh & Zafar (2026). Do LLM hallucination detectors suffer from low-resource effect? EACL 2026. https://aclanthology.org/2026.eacl-long.136/

## Team

- Ahmad Ali Khan (Lead, 24L-3049)
- Talha Pasha (24L-3005)
- Moeed Amir (24L-3060)
- Bilal Kashif (24L-3036)

## Repository layout

| Path | Content |
|---|---|
| `configs/` | seed and sizes (`data.yaml`), languages, prompts, run settings (`run.yaml`) |
| `data/` | `README.md` (data card and decision log), `processed/` (shared-schema JSONL), `eda/` (figures, tables, findings); `data/raw/` is rebuilt, not stored |
| `src/common/` | shared schema, `matches()` labelling rule, JSONL IO, plots |
| `src/datasets/` | builders for mTREx, X-FACTR, PsiloQA |
| `src/detectors/` | our marked copies of the authors' scripts (`*_g6.py`), `AUTHORS_CODE.md`, `patches/`, `LICENSE.authors` |
| `src/analysis/` | Phase 4 metrics and `ANALYSIS_SPEC.md` |
| `scripts/` | one script per stage (below) |
| `results/` | run outputs per dataset and language, `tables/`, `figures/`, `errors/`, `RUNS.md`; `results/raw/` stays on the run machine |
| `tests/` | `unit/`, `integration/`, `regression/`, `system/` plus top-level checks |
| `third_party/low-resource-hallucination-detection` | the authors' repository as a submodule, read-only |
| `Deliverables/N/` | the submitted `F26-6.pdf` and `F26-6.ipynb` of each phase |
| `report/final/` | LaTeX source of the final report and its diagrams |
| `F26-6.ipynb` | working notebook, runs from the saved results |

## Setup

CPU part (data, analysis, tests). Python 3.11, Windows shown; on Linux use `source .venv/bin/activate`.

```
git clone --recurse-submodules https://github.com/ahmad1khan2/f26-g6-hallucination-detectors
cd f26-g6-hallucination-detectors
python -m venv %USERPROFILE%\venvs\f26g6
%USERPROFILE%\venvs\f26g6\Scripts\activate
pip install -r requirements.lock.txt
python scripts/check_env.py
```

`requirements.txt` lists the direct dependencies and `requirements.lock.txt` the exact versions. The stages above run with that lock alone. The test suite also needs the model libraries (torch, transformers, spaCy and the rest), because its fixtures build a tiny model; on a machine without a GPU the CPU build of torch is enough:

```
pip install -r requirements-gpu.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

GPU part (Windows, NVIDIA GPU with 16 GB, Python 3.11, git). The Llama model is gated: request access on Hugging Face, then provide a token in the `HF_TOKEN` environment variable (bootstrap also offers a login). `scripts/bootstrap.py` creates `.venv`, installs the CUDA build of torch and the pins from `requirements-gpu.txt` / `requirements-gpu.lock.txt`, and runs the GPU check:

```powershell
py -3.11 scripts\bootstrap.py
```

Full steps and troubleshooting: `GPU_SETUP.md`. Every command for pausing, resuming and backing up a run: `RUN_COMMANDS.md`.

## Running the stages in order

1. Download the raw data (git bash; needs `datasets`): `bash scripts/download_data.sh`
2. Build the processed files (`data/processed/*.jsonl`, `build_log.txt`): `python scripts/build_processed.py`
3. EDA figures and tables (`data/eda/`): `python scripts/make_report_assets.py`
4. GPU run (on the GPU machine; each stage resumes where it stopped, so rerun the same command after any stop):
   ```powershell
   .venv\Scripts\python scripts\run_phase3.py smoke
   .venv\Scripts\python scripts\run_phase3.py timing --hours 30
   .venv\Scripts\python scripts\run_phase3.py full
   .venv\Scripts\python scripts\run_phase3.py verify          # add --partial for a run stopped early
   ```
   Or all stages unattended: `powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Hours 24` (see `GPU_SETUP.md` and `RUN_COMMANDS.md`). Commit and push `results/` afterwards; `results/raw/` is git-ignored.
5. Phase 3 summary (`results/summary_phase3.csv`, `phase3_acc_auroc.png`): `python scripts/summarize_results.py`
6. Hand check of X-FACTR answers: `python scripts/handcheck.py draw`, fill `checked_label` in `results/xfactr/handcheck.csv`, then `python scripts/handcheck.py score`
7. Analysis tables (`results/tables/*.csv`, spec in `src/analysis/ANALYSIS_SPEC.md`): `python scripts/run_analysis.py`
8. Phase 4 figures (`results/figures/`): `python scripts/make_report_assets.py --phase4`
9. Error-analysis sample (`results/errors/sample.csv`): `python scripts/error_sample.py`
10. Generator entropy table (`results/tables/entropy.csv`): `python scripts/entropy_table.py`
11. Notebook: open `F26-6.ipynb` and run all cells; it reads the saved results and needs no GPU.
12. Final report (needs a LaTeX install with LuaLaTeX and the TeX Gyre fonts): in `report/final/`, `latexmk -lualatex main.tex` (without Perl: `lualatex main`, `bibtex main`, then `lualatex main` twice). Final PDF: `Deliverables/6/`.

## Tests

```
pytest                # everything except the GPU tests
pytest tests/unit     # one level: unit, integration, regression or system
pytest -m gpu         # real-model tests, GPU machine only
```

## Model files

We use the public `meta-llama/Llama-3.1-8B-Instruct` unchanged: it is loaded in 4-bit (NF4) at load time and is not fine-tuned, so there are no model weights of ours to ship. The MAM classifiers are small and are retrained inside the run (seeds 42, 43, 44). Their weights were not kept; only their test-split predictions are saved, in `results/<dataset>/<language>/mam_preds_seed*.jsonl`. Rebuilding the classifiers needs the GPU run, because the activation features (about 7 GB) are not in git.

## Deliverables and tags

Each phase's submitted files are in `Deliverables/N/` (`F26-6.pdf`, and `F26-6.ipynb` from phase 2 on). Each phase ends with a git tag: `phase2`, `phase3`, `phase4`, `phase5`, `final`.

## Licences

Our code is MIT (`LICENSE`). The authors' released code that we adapt is MIT licensed too; its notice is in `src/detectors/LICENSE.authors`. The datasets keep their own licences, recorded in `data/README.md`: mTREx MIT, PsiloQA CC BY 4.0, X-FACTR has no licence file (data derived from Wikidata and Wikipedia). Llama 3.1 is used under Meta's licence.
