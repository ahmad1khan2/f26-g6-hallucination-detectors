#!/usr/bin/env bash
# Fetches the three raw datasets into data/raw/ (git-ignored). Usage: PYTHON=python bash scripts/download_data.sh
set -euo pipefail
PYTHON="${PYTHON:-python}"
mkdir -p data/raw
[ -d data/raw/mtrex_repo ] || git clone --depth 1 https://github.com/aisoc-lab/low-resource-hallucination-detection data/raw/mtrex_repo
[ -d data/raw/xfactr ] || git clone --depth 1 https://github.com/jzbjyb/x-factr data/raw/xfactr
[ -d data/raw/psiloqa ] || "$PYTHON" - <<'PY'
from datasets import load_dataset
ds = load_dataset("s-nlp/PsiloQA")
ds.save_to_disk("data/raw/psiloqa")
print(ds)
PY
for d in mtrex_repo xfactr; do echo "$d $(git -C data/raw/$d rev-parse HEAD)"; done
