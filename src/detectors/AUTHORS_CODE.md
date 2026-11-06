# Authors' code: what each script does

Source: `third_party/low-resource-hallucination-detection` at commit 644d655 (read-only). Paper: Datta et al., EACL 2026 (`2026.eacl-long.136`). All paths below are relative to `scripts/` in that repository. Every statement was checked against the code at that commit; page numbers refer to the paper's printed page numbers (2959 to 2985).

Items common to all six scripts:

- Model: `meta-llama/<model_name>` loaded in bfloat16 with `device_map="auto"`, `trust_remote_code=True`, `cache_dir="/data/hf_cache/"` (hard-coded), `max_memory` of 135 GiB on the chosen GPU and 0 GiB elsewhere. The tokenizer uses `padding_side="left"` and `pad_token_id = eos_token_id`.
- Prompt: system, user and assistant turns from `prompts/mTREx_prompts.json` (keyed by the language name, then by category), built with `apply_chat_template(..., tokenize=False, add_generation_prompt=True)`. The assistant turn (for example `Answer (in English): `) is part of the prompt, and generation continues after it.
- Data: `{mTREx_data_dir}/{language}/{category}.csv`, columns `source` and `target`; the first `max_num_samples` rows. Categories `capitals`, `country`, `official_language`. Languages `English`, `Hindi`, `Bengali`, `Deutsch`, `Urdu`. The G-MMLU branch is not used by this project.
- Label (`post_process_mtrex_response`): quotes and parentheses are removed from `target`, the result is split on `<OR>`, each phrase is stripped and lower-cased, and the answer is correct iff any phrase is a substring of the lower-cased response. The response has `\n` replaced by a space and is stripped.
- Greedy decoding: `do_sample=False, num_beams=1, temperature=0`, `max_new_tokens` 50.
- Every per-item `try/except` logs the error at INFO level and moves on, so a failed item is dropped without a trace in the outputs.
- Default paths point to `/data/debtanu/Research_work/...` (the authors' server). All are overridable except the cache directory.

## MAM_HD_methods/ModelArtifacts.py

Generates the greedy answer and stores the first-token activations (MAM, "model artifacts").

- Arguments (default): `--gpu_index` (`"0"`), `--model_name` (required; 4 choices), `--layer_number` (`-1`, meaning all layers), `--language` (required), `--max_num_samples` (2500), `--max_new_tokens` (50), `--min_new_tokens` (5), `--mTREx_data_dir`, `--base_results_dir`, `--dataset_name` (`mTREx` or `GMMLU`, required), `--prompts_dir`, `--seed_val` (42; unused on mTREx).
- Reads: the mTREx CSV and the prompts file.
- Writes, under `{base_results_dir}/ModelArti_{dataset}_results_v1/{model}_ModelArti_{category}_{language}_results/`: `pickle_files/{category}_chunk_{i}.pickle` (one pickled dict per item, `i` = row number + 1), `{dataset}_{model}_{category}_{language}.json` (all generations), `.txt` (counts and accuracy, with and without a `?` in the answer), and a `_results_log.txt`.
- Decoding: greedy with `min_new_tokens=5`, `max_new_tokens=50`, `output_scores=True`. Then one teacher-forced forward pass `model(sequences)` over prompt plus answer; the hooks and `first_logits` come from this pass.
- Hooks: forward hooks on `model.layers.<n>.mlp.up_proj` (fully connected, width 14,336 for Llama-3.1-8B) and `model.layers.<n>.self_attn.o_proj` (attention output, width 4,096) for every layer, with the output stored for every call. Only the middle layer (index 15 of 32, `total // 2 - 1`) and the last layer (31) are used, stacked in that order.
- Stored per item: `source`, `target`, `str_response`, `correct`, `first_logits` (128,256 floats), `start_pos`, `first_fully_connected` (2 x 14,336), `first_attention` (2 x 4,096). The `final_*` arrays are computed and not stored.
- "First" is the position `start_pos` in the teacher-forced sequence, which is the first generated token itself (the prompt occupies positions 0 to `start_pos - 1`). `first_logits` is therefore the next-token distribution after the first generated token.
- Hard-coded: hook layer names, chunk size 1, cache path, memory limits, layer counts per model (32 for Llama-3.1-8B).
- What we change: see "Deviations from the released code".

## MAM_HD_methods/ModelArtifacts_Classifier.py

Trains and tests the MAM classifiers on the stored pickles.

- Arguments (default): `--gpu_index` (`"0"`), `--model_name`, `--language`, `--max_num_samples` (2500), `--num_hidden_nodes` (256), `--batch_size` (128), `--dropout_mlp` (0.5), `--dropout_gru` (0.25; unused), `--learning_rate` (1e-4), `--weight_decay` (1e-2), `--base_results_dir`, `--dataset_name`, `--seed_val` (42).
- Reads: `pickle_files/{category}_chunk_{i}.pickle` for `i = 1, 2, ...`; the loop stops at the first missing file. A failed item in the generation script leaves a gap, so every item after it is silently ignored here.
- Writes `classifier_results_NEW/`: `classifier_results_{artifact}.json` (test indices and predicted probabilities), `classifier_results.txt`, a log. Artifacts: `first_logits`, `first_fully_connected_{0,1}`, `first_attention_{0,1}` (0 = middle layer, 1 = last layer).
- Model: `Linear(input, 256)`, ReLU, `Dropout(0.5)`, `Linear(256, 2)`. Training: AdamW, 1001 steps (`range(1001)`), each step a random batch of 128 drawn with `torch.randperm` (no replacement within a step), cross-entropy. No `torch.manual_seed` call, so the weights, dropout and batches are unseeded; only the split is seeded.
- Split: `train_test_split(test_size=0.2, random_state=seed_val)`, shuffled, within one category and language.
- Label: hallucination = 1 - correct, so the positive class is hallucination. AUROC is `roc_auc_score(y_test, softmax(logits)[:, 1])` on the 20% split; accuracy at threshold 0.5.
- `first_logits` input is the softmax of the stored logits (128,256 inputs).
- What we change: see "Deviations from the released code".

## MAM_HD_methods/ModelArtifacts_multiple_gen_tokens.py

Same as `ModelArtifacts.py` plus the multi-token variant (paper Appendix F.2, Tables 16 and 17).

- Extra argument: `--avg_num_tokens` (10). Default `--base_results_dir` is `/data/debtanu/Research_work/Results_ModelArti_Rebuttal`. All other arguments, reads, decoding and hooks are as above.
- Extra stored fields: `token_pos_fully_connected_avg` and `token_pos_attention_avg` (each 2 layers x width): the mean over positions `start_pos` to `start_pos + 9` of the teacher-forced sequence, stopping early when the sequence is shorter.
- Writes the same files as `ModelArtifacts.py` under the new base directory.

## MAM_HD_methods/ModelArtifacts_Classifier_multiple_gen_tokens.py

Same as `ModelArtifacts_Classifier.py` with two more artifacts, `token_pos_fully_connected_avg_{0,1}` and `token_pos_attention_avg_{0,1}`.

- Output directory is `classifier_results_latest/` (not `_NEW`); the default `--base_results_dir` is the `Results_ModelArti_Rebuttal` path above. Arguments, model, training and label are identical to the single-token classifier script.

## Sampling_HD_methods/selfcheckGPT.py

SelfCheckGPT with the unigram score.

- Arguments (default): `--gpu_index` (`"0"`), `--model_name`, `--language`, `--num_sample_generations` (20), `--max_num_samples` (2500), `--max_new_tokens` (50), `--mTREx_data_dir`, `--base_results_dir`, `--dataset_name`, `--prompts_dir`, `--seed_val` (42; unused on mTREx).
- Reads: the mTREx CSV and the prompts file.
- Writes `{base}/SelfCheckGPT_{dataset}_results_v1/{model}_SelfCheckGPT_{category}_{language}_results/`: a `.json` (question, target, base response, `correct`, `sampled_responses`, score), a `.txt` (the arrays of correctness and scores, the AUROC, counts and accuracy) and a log. Nothing is written until the whole category has finished, so a killed run loses everything.
- Decoding: base answer greedy with no `min_new_tokens`. Samples: `do_sample=True, temperature=1`, `num_return_sequences=20`, `max_new_tokens=50`, no top-k or top-p argument, so the model's own `generation_config` applies (Llama-3.1-8B-Instruct ships `top_p` 0.9; confirm on the run machine). Each sample is cut after its first end token.
- Score: a unigram model is built on the base answer plus the 20 samples (`spacy.blank(lang_code)` tokens, `k=0`, unseen token mapped to `<unk>` with count 0), the base answer's average negative log-probability is taken and the score is `1 - exp(-avg_neg_logprob)`. High score means low consistency, and it is scored against `uncertain = 1 - correct` (hallucination positive).
- Language codes: English `en`, Hindi `hi`, Bengali `bn`, Deutsch `de`, Urdu `ur`.
- AUROC: one number per category and language over all items.
- What we change: see "Deviations from the released code".

## Sampling_HD_methods/semanticEntropy.py

Semantic Entropy with LaBSE clustering.

- Arguments (default): as `selfcheckGPT.py` plus `--cutoff_score` (0.75).
- Reads: the mTREx CSV, the prompts file and `sentence-transformers/LaBSE` (loaded on the same device).
- Writes `{base}/SE_{dataset}_results_v1/{model}_SE_{category}_{language}_results/`: a `.json` (including `log_likelihoods`, `semantic_ids`, `entropies` per item), a `.txt` (AUROC per key, counts) and a log. Again nothing is written until the category ends.
- Decoding: as in `selfcheckGPT.py` (greedy base answer without `min_new_tokens`; 20 samples at temperature 1). Per-sample log-likelihood list comes from `compute_transition_scores(normalize_logits=True)` up to and including the first end token.
- Clustering: greedy. Each unassigned sample starts a cluster, and later samples join it if the LaBSE cosine similarity to the first member is strictly greater than 0.75.
- Keys computed: `cluster_assignment_entropy`, and for the mean (no suffix) and sum (`_sum`) of token log-likelihoods, `semantic_entropy`, `semantic_entropy_sum-normalized`, `semantic_entropy_sum-normalized-rao`, `semantic_entropy_cmean` (each also with the `_sum` prefix variant, for example `semantic_entropy_sum_cmean`). `regular_entropy` and `regular_entropy_sum` are computed and excluded from the AUROC step. 9 keys are scored.
- AUROC: `roc_auc_score(uncertain, value)` per key, higher entropy meaning hallucination, over all items of a category.
- Hard-coded: the plain `semantic_entropy` aggregation (`agg='sum'`) subtracts `5.0` from each cluster's log-probability (`np.log(np.sum(np.exp(...))) - 5.0`). Because `predictive_entropy` is `-mean(log_probs)`, the offset adds the same constant 5 to every item's score, so it cannot change an AUROC.
- Which key is "SEM" in the paper: the paper does not name a key. Its text (Section 4.3, page 2963, and Appendix E.3, pages 2974 and 2975) says the sequence log-likelihood is the average of the token log-probabilities, the probability of a cluster is the sum of the probabilities of its members (`P(C_k) = sum P(response_i)`), and the entropy is taken over the cluster probabilities. That is the combination of the mean aggregation and the `'sum'` cluster aggregation, which is `semantic_entropy`. It is the best match, not a confirmed one: `semantic_entropy_cmean` (mean of member probabilities) and the `_sum` variants do not fit that text, but the text also never mentions the `- 5.0` offset or the `-sum(log_probs) / n` form the code actually computes. We store all keys, report `semantic_entropy` as the main SEM score and treat the other keys as a sensitivity check.

## Facts table

| Item | Value in the code | Paper |
|---|---|---|
| Greedy answer | temperature 0, `do_sample=False`, 50 new tokens max | same (page 2971) |
| `min_new_tokens` | 5 in the MAM scripts, none in SelfCheckGPT and Semantic Entropy | not stated |
| Samples | 20 at temperature 1 | 20 at 1.0 (page 2971) |
| Sample decoding | no top-k or top-p argument; the model's `generation_config` applies | not stated |
| LaBSE threshold | 0.75, strictly greater | 0.75 (page 2963) |
| SelfCheckGPT score | `1 - exp(-avg_neg_logprob)`, unigram, `k=0` | same formula (page 2962); the paper's words "higher scores mean higher consistency" are the reverse of what the code scores against `1 - correct` |
| Classifier | 1 hidden layer of 256, ReLU, dropout 0.5 | 256 hidden units (page 2962); dropout not stated |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-2 | "Adam" with the same values (page 2971) |
| Steps and batch | 1001 steps, batch 128 | 1000 iterations, batch 128 (page 2971) |
| Split | 80/20, `random_state=42`, one category at a time | random 80/20 per category (page 2962) |
| Layers | middle (15) and last (31) stored; index 0 and 1 trained as separate artifacts | final layer (page 2962) |
| Token position | position `start_pos` of the teacher-forced pass (first generated token) | first generated token (page 2962) |
| mTREx sizes | rows up to `max_num_samples` 2500 | 2500, 2500, 2374 (page 2961) |
| Label rule | lower-case substring of any `<OR>` phrase | same (page 2963) |
| Precision and hardware | bfloat16, 8 x H200 | same hardware (page 2971) |
| Timing | not recorded by the scripts | measured in Task 6 |
| AUROC unit | one per category and language | per dataset and language |
| Seeds for the classifier | split only | 20 reruns with different splits (page 2975) |

## Deviations from the released code

The adapted scripts in `src/detectors/` change only the following, each marked `# G6:` in the code.

1. One shared greedy answer per item, made with the MAM settings (`min_new_tokens=5`, `max_new_tokens=50`), is used for the label and for all three detectors. The released SelfCheckGPT and Semantic Entropy scripts generate their own base answer without `min_new_tokens`; sharing one answer makes the three detectors score the same text and keeps one label per item.
2. The first-token entropy of the next-token softmax is stored instead of the 128,256-float `first_logits` per item. This is the only quantity the entropy analysis needs and it avoids storing 0.5 MB per item. The `first_logits` classifier artifact is not trained.
3. The 20 samples are generated once per item and shared by SelfCheckGPT and Semantic Entropy. The released scripts sample twice, independently.
4. The classifier is seeded with `torch.manual_seed` (the released script seeds only the split), so a run can be repeated exactly. Seeds 42, 43 and 44 give three splits.
5. Only the middle and last layers are hooked, not all 32. The stored arrays are identical; the saved memory lets the run fit in 16 GB.
6. Failures are logged to `errors.jsonl` with the item's uid and the stage, instead of being dropped silently, and a run resumes from the items already stored.
7. spaCy tokenizers for SelfCheckGPT: `sw` and `ilo` use the language-neutral `xx` pipeline (spaCy has no blank pipeline for them), Vietnamese uses `use_pyvi=False`, Japanese uses sudachipy through `spacy.blank("ja")`. The released scripts cover only `en`, `hi`, `bn`, `de`, `ur`.
8. Model loading: 4-bit NF4 with bitsandbytes on one GPU instead of bfloat16 spread by `device_map="auto"`; paths, language lists and our schema loaders replace the CSV reading; the label comes from `matches()` in `src/common/text.py`.

Detection logic, hyperparameters, clustering rule and score formulas are unchanged.
