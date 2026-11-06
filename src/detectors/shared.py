"""Group 6 helpers shared by the adapted detector scripts: model loading, prompts, result files."""
import json
import os
import pickle
from functools import lru_cache
from itertools import zip_longest
from pathlib import Path

from src.common.io import ROOT, hash_key, load_config
from src.common.schema import Record

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
RESULTS = ROOT / "results"
AUTHORS_PROMPTS = ROOT / "third_party" / "low-resource-hallucination-detection" / "prompts" / "mTREx_prompts.json"

# spaCy has no Swahili or Ilocano pipeline, so they get its language-neutral tokenizer.
SPACY_CODES = {code: code for code in "en de hi bn ur zh ja vi tl yo fi ar".split()} | {"sw": "xx", "ilo": "xx"}
# pyvi would add a dependency only for Vietnamese word segmentation; whitespace syllables are enough for unigrams.
SPACY_CONFIG = {"vi": {"nlp": {"tokenizer": {"use_pyvi": False}}}}


def load_generator(model_name: str = MODEL_NAME):
    """Llama in 4-bit NF4 on the GPU, with the authors' left padding and pad = eos."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=compute_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization,
                                                 device_map={"": 0})
    return model.eval(), tokenizer


def load_prompts() -> dict:
    """The authors' mTREx prompts unchanged, plus ours for X-FACTR and PsiloQA."""
    with open(AUTHORS_PROMPTS, encoding="utf-8") as f:
        return {"mtrex": json.load(f), **load_config("prompts")}


def _mtrex_language_name(code: str) -> str:
    names = {c: name for name, c in load_config("data")["mtrex"]["languages"].items()}
    return names[code]


def build_messages(record: Record, prompts: dict) -> list:
    """Chat messages for a record. A PsiloQA given answer is never included: callers append it themselves."""
    if record.dataset == "mtrex":
        language = prompts["mtrex"][_mtrex_language_name(record.language)]
        category = language[record.relation]
        user = category["user"].format(source=record.meta["source"])
        return _messages(language["system"], user, category["asst"])
    language = prompts[record.dataset][record.language]
    if record.dataset == "xfactr":
        return _messages(language["system"], language["user"].format(sentence=record.prompt), language["asst"])
    user = language["user"].format(passage=record.meta["passage"], question=record.prompt)
    return _messages(language["system"], user)


def _messages(system: str, user: str, assistant: str | None = None) -> list:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return messages + [{"role": "assistant", "content": assistant}] if assistant is not None else messages


def chat_text(record: Record, tokenizer, prompts: dict) -> str:
    """The chat-template string the model reads, built as the authors do (with add_generation_prompt)."""
    return tokenizer.apply_chat_template(build_messages(record, prompts), tokenize=False,
                                         add_generation_prompt=True)


@lru_cache(maxsize=None)
def blank_spacy(language: str):
    """Tokenizer-only spaCy pipeline. Cached: building one per item is slow (Japanese loads a dictionary)."""
    import spacy

    return spacy.blank(SPACY_CODES[language], config=SPACY_CONFIG.get(language, {}))


def results_dir(dataset: str, language: str, raw: bool = False) -> Path:
    """results/<dataset>/<language>/ (tracked) or results/raw/<dataset>/<language>/ (git-ignored)."""
    path = RESULTS / "raw" / dataset / language if raw else RESULTS / dataset / language
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sync(f) -> None:
    """Force a file's bytes to the disk: without this a power cut loses writes the OS was still holding."""
    f.flush()
    os.fsync(f.fileno())


def write_atomic(path, text: str) -> None:
    """Replace a file in one step, so a stop at any moment leaves the old file or the new one, never half of one."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            _sync(f)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def append_jsonl(path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        _sync(f)


def read_results(path) -> list:
    """Rows of a results file. Only an unterminated last line is forgiven (a run killed mid-write)."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_bytes().split(b"\n")
    complete, tail = lines[:-1], lines[-1]
    rows = [json.loads(line.decode("utf-8")) for line in complete if line.strip()]
    return rows if not tail.strip() else rows + _parse_tail(tail)


def _parse_tail(tail: bytes) -> list:
    try:
        return [json.loads(tail.decode("utf-8"))]
    except ValueError:  # cut inside a row, inside a character, or zero bytes left by a power cut
        return []


def repair_jsonl(path) -> None:
    """Cut a half-written last line so the next append starts on a clean line. Call once before resuming."""
    path = Path(path)
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        with open(path, "r+b") as f:  # truncate in place: rewriting could lose the whole file in a second power cut
            f.truncate(data.rfind(b"\n") + 1)


def done_uids(path) -> set:
    return {row["uid"] for row in read_results(path)}


def append_feature(path, feature: dict) -> None:
    """Add one item's activations (a dict with a uid) to a pickle stream, as the authors' chunks do."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as f:
        pickle.dump(feature, f)
        _sync(f)


def _scan_features(path):
    """Yield (feature, end offset) for each whole record; stops quietly at a half-written tail."""
    with open(path, "rb") as f:
        while True:
            try:
                feature = pickle.load(f)
            except (EOFError, pickle.UnpicklingError, MemoryError):  # MemoryError: a garbled frame length
                return
            yield feature, f.tell()


def read_features(path) -> list:
    """All features by uid in first-seen order. A repeated uid (killed between two writes) keeps the later one."""
    if not Path(path).exists():
        return []
    return list({feature["uid"]: feature for feature, _ in _scan_features(path)}.values())


def repair_features(path) -> set:
    """Cut a half-written last record so the next append stays readable; returns the uids kept. Call once before resuming."""
    path = Path(path)
    if not path.exists():
        return set()
    end, uids = 0, set()
    for feature, end in _scan_features(path):
        uids.add(feature["uid"])
    with open(path, "r+b") as f:
        f.truncate(end)
    return uids


def resume_mam(answers_path, features_path) -> set:
    """Repair both MAM files and return the uids that are done. Call once before resuming.

    An item counts as done only with both its feature and its answer. Runs written before appends were synced
    can lose a feature in a power cut while its answer survives; that answer is dropped so the item runs again.
    """
    repair_jsonl(answers_path)
    feature_uids = repair_features(features_path)
    rows = read_results(answers_path)
    kept = [row for row in rows if row["uid"] in feature_uids]
    if len(kept) != len(rows):
        write_atomic(answers_path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept))
    return {row["uid"] for row in kept}


def log_error(directory, uid: str, stage: str, error: Exception) -> None:
    path = Path(directory) / "errors.jsonl"
    repair_jsonl(path)  # no stage repairs this file on resume, and a torn row would corrupt the next one
    append_jsonl(path, {"uid": uid, "stage": stage, "error": f"{type(error).__name__}: {error}"})


def write_run_json(directory, **fields) -> None:
    path = Path(directory) / "run.json"
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):  # unreadable after a power cut: only the start time is lost
        current = {}
    # default=str: a model config value json cannot store must not stop a long run on every restart.
    write_atomic(path, json.dumps(current | fields, ensure_ascii=False, indent=2, default=str))


def item_seed(uid: str) -> int:
    """A per-item seed, so sampling gives the same result after a restart."""
    return int(hash_key(load_config("data")["seed"], uid)[:8], 16)


def take_balanced(records: list, limit: int | None) -> list:
    """First `limit` records, cycling through relations (PsiloQA: generators) and keeping file order in each."""
    if limit is None:
        return list(records)
    groups = {}
    for record in records:
        groups.setdefault(record.relation or record.meta.get("model", ""), []).append(record)
    cycled = [r for row in zip_longest(*groups.values()) for r in row if r is not None]
    return cycled[:limit]
