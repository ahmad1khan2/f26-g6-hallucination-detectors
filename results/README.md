# Results

Outputs of the Phase 3 run. One folder per dataset and language: `results/<dataset>/<language>/`.

- `answers.jsonl`: uid, answer, label, label_rule, first_token_entropy (one greedy answer, shared by all detectors)
- `sgm.jsonl`: SelfCheckGPT score per uid
- `sem.jsonl`: Semantic Entropy values per uid (PsiloQA: per question, with all its uids)
- `mam_preds_seed42.jsonl`, `mam_preds_seed43.jsonl`, `mam_preds_seed44.jsonl`: MAM test-split probabilities
- `errors.jsonl`: items that failed, with the stage and the error
- `run.json`: model, quantization, submodule commit, seeds, item count, times, GPU, generation config

`results/raw/` (not tracked) holds the activations (`features.pkl`), the samples (`samples.jsonl`) and the smoke and timing runs.
`RUNS.md` lists every stage that was started. Section Flags is written by `python scripts/run_phase3.py verify`.

## Flags

See `results/flags.txt` after `verify`: notes such as "not run yet" or a thin test split. Hard problems fail the check.

## Machine versions

`scripts/check_gpu.py` appends one block per machine below.
