# Error analysis sample

`sample.csv` is drawn by `scripts/error_sample.py`: primary MAM (fully connected),
test split, seed 42. Missed hallucinations are hallucinated items scoring below
the median score of that language's correct items; false flags are correct items
scoring above the median score of its hallucinated items. Per setting 15 of each
type, in hash order, round-robin over languages (90 items in all).

## Coding

The code list was fixed before reading: matcher_error, wrong_language,
partial_answer, refusal_or_non_answer, plausible_wrong_entity, unrelated_entity,
prompt_or_template_problem, gold_data_error, span_problem (PsiloQA), other.

Each item was coded twice, independently: `code_a` (Ahmad Ali Khan) and
`code_b` (Bilal Kashif). Agreement: 76 of 90 items (84.4%). The 14
disagreements were resolved by discussion into `final_code`, with the reason in
`note`. Items in languages neither coder reads were checked against the gold
aliases and say "tool-checked" in the note.

Most `other` items are false flags whose answer is correct (note "Correct
answer, detector uncertain"): the list has no code for a detector error on a
correct answer, and the list was not changed after coding started.
