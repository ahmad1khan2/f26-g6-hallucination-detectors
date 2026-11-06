"""Phase 3 driver: smoke, timing, full, verify. Every stage resumes by uid, so rerun the same command after a stop."""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path.home() / "hf_cache"))  # before anything imports transformers
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from src.common.io import ROOT, load_config
from src.common.schema import read_jsonl
from src.detectors import ModelArtifacts_Classifier_g6 as classifier_first
from src.detectors import ModelArtifacts_Classifier_multiple_gen_tokens_g6 as classifier_answer
from src.detectors import ModelArtifacts_g6 as mam
from src.detectors import ModelArtifacts_multiple_gen_tokens_g6 as mam_given
from src.detectors import selfcheckGPT_g6 as sgm
from src.detectors import semanticEntropy_g6 as sem
from src.detectors import shared
from tests.test_results import check_results

DATASETS = ("mtrex", "xfactr", "psiloqa")
MTREX_PRIORITY = ["en", "hi", "de", "bn", "ur"]
SMOKE_LANGUAGE = {"mtrex": "bn", "xfactr": "ilo", "psiloqa": "ar"}
PAPER_ACCURACY = {"capitals": 74, "country": 77, "official_language": 70}
ACCURACY_WARN_POINTS = 10
MAX_NEW_TOKENS = 50
PAUSED_EXIT = 75  # tells run_all.ps1 this is a pause, so it must not retry, verify or push
DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_SAMPLE_STEP = 50
SMOKE_STEPS = (10, 20)
MAM_TIMING_ITEMS = 20
SAMPLING_TIMING_ITEMS = 3
PROCESSED = ROOT / "data" / "processed"
SUBMODULE = ROOT / "third_party" / "low-resource-hallucination-detection"
DEFAULT_CONFIG = ROOT / "configs" / "run.yaml"
CONFIG_HEADER = "# Run sizes. `run_phase3.py timing --hours H` rewrites this file and adds `measured`; `full` needs it.\n"
RUNS_HEADER = "| date | machine | command | commit |\n|---|---|---|---|\n"


# ---- pure helpers ---------------------------------------------------------------------------------------------

def languages(dataset: str, cfg: dict) -> list:
    """Languages in run priority order; cfg['languages'][dataset] only restricts (the proposal's cut order)."""
    if dataset == "mtrex":
        order = MTREX_PRIORITY
    elif dataset == "xfactr":
        order = ["en"] + [x["code"] for x in load_config("languages")["languages"] if x["chosen"]]
    else:
        order = load_config("data")["psiloqa"]["languages"]
    keep = (cfg.get("languages") or {}).get(dataset)
    return [code for code in order if code in keep] if keep else list(order)


def _sampling_seconds(throughput, counts, num_languages, limits) -> float:
    return sum(throughput["sampling"][d] * num_languages[d] * (counts[d] if limits.get(d) is None else min(counts[d], limits[d]))
               for d in counts)


def _fixed_seconds(throughput, counts, num_languages) -> float:
    return sum((throughput["mam"][d] * counts[d] + throughput["clf"][d]) * num_languages[d] for d in counts)


def plan_sizes(throughput, counts, hours, num_languages) -> dict:
    """Sampling limit per dataset (None = all). MAM and classifiers are never cut; one N for every dataset."""
    budget = hours * 3600 - _fixed_seconds(throughput, counts, num_languages)
    if budget <= 0:
        print(f"WARNING: MAM and the classifiers alone need more than {hours} h, so sampling gets 0 items.")
        return dict.fromkeys(counts, 0)
    n = max(counts.values())
    while n > 0 and _sampling_seconds(throughput, counts, num_languages, dict.fromkeys(counts, n)) > budget:
        n -= 1
    return {d: None if n >= counts[d] else n for d in counts}


def round_robin(groups_by_language: dict, step: int):
    """Yield (language, records) turns; a turn takes whole groups so a repeated PsiloQA question never splits."""
    turns = max((-(-len(g) // step) for g in groups_by_language.values()), default=0)
    for turn in range(turns):
        for language, groups in groups_by_language.items():
            chunk = groups[turn * step:(turn + 1) * step]
            if chunk:
                yield language, [record for group in chunk for record in group]


def category_accuracy(answers: list, records: list) -> dict:
    relation = {r.uid: r.relation for r in records}
    counts = {}
    for row in answers:
        hit, total = counts.get(relation[row["uid"]], (0, 0))
        counts[relation[row["uid"]]] = (hit + (row["label"] == 0), total + 1)
    return {name: 100 * hit / total for name, (hit, total) in counts.items()}


def accuracy_warnings(accuracy: dict) -> list:
    return [f"{name} accuracy {value:.1f} is more than {ACCURACY_WARN_POINTS} points from the paper's {PAPER_ACCURACY[name]}"
            for name, value in accuracy.items() if abs(value - PAPER_ACCURACY[name]) > ACCURACY_WARN_POINTS]


# ---- environment ------------------------------------------------------------------------------------------------

def _git(*args, cwd=ROOT) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log_run(results: Path, command: str) -> None:
    path = results / "RUNS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = f"| {_now()} | {platform.node()} ({_gpu_name()}) | {command} | {_git('rev-parse', '--short', 'HEAD')} |\n"
    fresh = not path.exists() or not path.stat().st_size
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write((RUNS_HEADER if fresh else "") + row)


def _dirs(dataset: str, language: str):
    return shared.results_dir(dataset, language), shared.results_dir(dataset, language, raw=True)


@lru_cache(maxsize=None)
def _processed(dataset: str) -> dict:
    by_language = {}
    for record in read_jsonl(PROCESSED / f"{dataset}.jsonl"):
        by_language.setdefault(record.language, []).append(record)
    return by_language


def load_records(dataset: str, language: str, limit) -> list:
    return shared.take_balanced(_processed(dataset)[language], limit)


# ---- stages -----------------------------------------------------------------------------------------------------

def _run_mam(dataset, language, records, model, tokenizer, prompts):
    out, raw = _dirs(dataset, language)
    if dataset == "psiloqa":
        mam_given.run_schema_given_answer(records, model, tokenizer, prompts, out, raw)
    else:
        mam.run_schema(records, model, tokenizer, prompts, out, raw, max_new_tokens=MAX_NEW_TOKENS)


def _write_run_info(out, model, cfg, items, started):
    quantization = getattr(model.config, "quantization_config", None)
    fields = {"model": getattr(model.config, "_name_or_path", "") or shared.MODEL_NAME,
              "quantization": quantization.to_dict() if hasattr(quantization, "to_dict") else "none",
              "submodule_commit": _git("rev-parse", "HEAD", cwd=SUBMODULE),
              "seeds": cfg.get("seeds", DEFAULT_SEEDS), "items": items, "started": started, "ended": _now(),
              "gpu": _gpu_name(), "generation_config": model.generation_config.to_dict()}
    if (out / "run.json").exists():
        del fields["started"]  # keep the first start across resumed runs
    shared.write_run_json(out, **fields)


def _report_mtrex_accuracy(out, records):
    accuracy = category_accuracy(shared.read_results(out / "answers.jsonl"), records)
    for name, value in accuracy.items():
        print(f"mTREx en {name}: accuracy {value:.1f} (paper {PAPER_ACCURACY[name]})")
    for warning in accuracy_warnings(accuracy):
        print(f"WARNING: {warning}")


def stage_mam(cfg, model, tokenizer, prompts):
    for dataset in DATASETS:
        for language in languages(dataset, cfg):
            records = load_records(dataset, language, cfg.get("mam_limit"))
            started = _now()
            print(f"MAM {dataset}/{language}: {len(records)} items")
            _run_mam(dataset, language, records, model, tokenizer, prompts)
            _write_run_info(_dirs(dataset, language)[0], model, cfg, len(records), started)
            if dataset == "mtrex" and language == "en":
                _report_mtrex_accuracy(_dirs(dataset, language)[0], records)


def _run_classifier(dataset, language, seeds):
    out, raw = _dirs(dataset, language)
    features = raw / "features.pkl"
    if not features.exists():  # every item failed in MAM; the reason is in errors.jsonl
        shared.log_error(out, language, "classifier", FileNotFoundError(features))
    elif dataset == "psiloqa":
        classifier_answer.run_schema(features, out, seeds, _device())
    else:
        classifier_first.run_schema(features, out, seeds, by_relation=dataset == "mtrex", device=_device())


def stage_classifiers(cfg):
    for dataset in DATASETS:
        for language in languages(dataset, cfg):
            print(f"Classifiers {dataset}/{language}")
            _run_classifier(dataset, language, cfg.get("seeds", DEFAULT_SEEDS))


def _max_new_tokens(dataset, cfg):
    return cfg["psiloqa_max_new_tokens"] if dataset == "psiloqa" else MAX_NEW_TOKENS


def _sample(dataset, language, records, model, tokenizer, prompts, cfg):
    out, raw = _dirs(dataset, language)
    sgm.sample_schema(records, model, tokenizer, prompts, raw, num_samples=cfg.get("num_samples", 20),
                      max_new_tokens=_max_new_tokens(dataset, cfg), chunk=cfg.get("sample_chunk", 20), out_dir=out)


def _score(dataset, language, sbert):
    out, raw = _dirs(dataset, language)
    sgm.score_schema(raw / "samples.jsonl", out / "answers.jsonl", out, language)
    sem.score_schema(raw / "samples.jsonl", out, sbert, level="question" if dataset == "psiloqa" else "item")


def stage_sampling(cfg, model, tokenizer, prompts, sbert_loader):
    sbert = None
    for dataset in DATASETS:
        limit = (cfg.get("sampling_limit") or {}).get(dataset)
        groups = {language: sgm._question_groups(load_records(dataset, language, limit))
                  for language in languages(dataset, cfg)}
        for language, records in round_robin(groups, cfg.get("sample_step", DEFAULT_SAMPLE_STEP)):
            sbert = sbert if sbert is not None else sbert_loader(_device())
            print(f"Sampling {dataset}/{language}: {len(records)} items")
            _sample(dataset, language, records, model, tokenizer, prompts, cfg)
            _score(dataset, language, sbert)


# ---- commands ---------------------------------------------------------------------------------------------------

def run_full(args, cfg, loader, sbert_loader):
    model, tokenizer = loader()
    prompts = shared.load_prompts()
    stage_mam(cfg, model, tokenizer, prompts)
    stage_classifiers(cfg)
    stage_sampling(cfg, model, tokenizer, prompts, sbert_loader)
    print("Full run finished. Check it with: python scripts/run_phase3.py verify")
    return 0


def _smoke_labels_one_class(directory) -> bool:
    return len({row["label"] for row in shared.read_results(directory / "answers.jsonl")}) < 2


def _smoke_layout_problems(root, cfg, plan) -> tuple:
    problems, warnings = [], []
    for dataset, language in plan:
        directory = root / dataset / language
        names = ["answers.jsonl", "sgm.jsonl", "sem.jsonl", "run.json",
                 *[f"mam_preds_seed{s}.jsonl" for s in cfg.get("seeds", DEFAULT_SEEDS)]]
        one_class = _smoke_labels_one_class(directory)
        if one_class:
            warnings.append(f"{dataset}/{language}: all labels are one class, so the classifiers cannot be checked")
        # 20 items split by relation can leave every group one-class even when the language has both classes.
        skipped = any(row["stage"] == "classifier" for row in shared.read_results(directory / "errors.jsonl"))
        if skipped:
            warnings.append(f"{dataset}/{language}: some classifier groups had too few items of a class")
        for name in names:
            path = directory / name
            if not path.exists() or (not path.stat().st_size and not ((one_class or skipped) and name.startswith("mam_preds"))):
                problems.append(f"{dataset}/{language}/{name} is missing or empty")
        for name in ("features.pkl", "samples.jsonl"):
            if not (root / "raw" / dataset / language / name).exists():
                problems.append(f"raw/{dataset}/{language}/{name} is missing")
    return problems, warnings


def _write_examples(root, plan) -> Path:
    path = root / "examples.txt"
    blocks = []
    for dataset, language in plan:
        answers = {r["uid"]: r["answer"] for r in shared.read_results(root / dataset / language / "answers.jsonl")}
        shown = [r for r in load_records(dataset, language, max(SMOKE_STEPS)) if r.uid in answers][:5]
        blocks += [f"[{dataset}/{language}] {r.uid}\nPROMPT: {r.prompt}\nANSWER: {answers[r.uid]}\n" for r in shown]
    path.write_text("\n".join(blocks), encoding="utf-8", newline="\n")
    return path


def run_smoke(args, cfg, loader, sbert_loader):
    root = shared.RESULTS / "raw" / "smoke"
    shutil.rmtree(root, ignore_errors=True)
    plan = [(d, language) for d in DATASETS for language in ("en", SMOKE_LANGUAGE[d])]
    cfg = {**cfg, "languages": {d: ["en", SMOKE_LANGUAGE[d]] for d in DATASETS}}
    shared.RESULTS = root  # restored by main
    model, tokenizer = loader()
    prompts = shared.load_prompts()
    for items in SMOKE_STEPS:  # the second step finds the first step's items done: proves resume
        step = {**cfg, "mam_limit": items, "sampling_limit": dict.fromkeys(DATASETS, items)}
        stage_mam(step, model, tokenizer, prompts)
        if items == SMOKE_STEPS[-1]:
            stage_classifiers(step)
        stage_sampling(step, model, tokenizer, prompts, sbert_loader)
    problems, warnings = _smoke_layout_problems(root, step, plan)
    try:
        warnings += check_results(root, PROCESSED, step)
    except AssertionError as e:
        problems.append(f"result check: {e}")
    if not args.no_gpu_tests and subprocess.run([sys.executable, "-m", "pytest", "-m", "gpu", "-q"], cwd=ROOT).returncode:
        problems.append("pytest -m gpu failed")
    print(f"Examples to read (are the answers in the target language?): {_write_examples(root, plan)}")
    for line in warnings:
        print(f"WARN {line}")
    for line in problems:
        print(f"FAIL {line}")
    print("SMOKE FAIL" if problems else "SMOKE PASS")
    return 1 if problems else 0


def _timed(fn, *args, **kwargs):
    start = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - start


def _p95_answer_tokens(tokenizer) -> int:
    lengths = sorted(len(tokenizer(r.answer, add_special_tokens=False)["input_ids"])
                     for rows in _processed("psiloqa").values() for r in rows)
    return lengths[-(-95 * len(lengths) // 100) - 1]


def _out_of_memory(out) -> bool:
    return any("OutOfMemoryError" in row["error"] for row in shared.read_results(out / "errors.jsonl")
               if row["stage"] == "sampling")


def _time_sampling(dataset, records, model, tokenizer, prompts, sbert, cfg) -> tuple:
    """(seconds per item for sampling, for scoring, chunk used). Halves the chunk after an out-of-memory error."""
    out, raw = _dirs(dataset, "en")
    groups = len(sgm._question_groups(records))
    while True:
        (raw / "samples.jsonl").unlink(missing_ok=True)
        (out / "errors.jsonl").unlink(missing_ok=True)
        sampling = _timed(_sample, dataset, "en", records, model, tokenizer, prompts, cfg)
        if not _out_of_memory(out) or cfg["sample_chunk"] == 1:
            break
        cfg = {**cfg, "sample_chunk": cfg["sample_chunk"] // 2}
    scoring = _timed(_score, dataset, "en", sbert)
    return sampling / groups, scoring / groups, cfg["sample_chunk"]


def _measure(cfg, model, tokenizer, prompts, sbert) -> tuple:
    throughput = {"mam": {}, "clf": {}, "sampling": {}, "scoring": {}}
    seeds = cfg.get("seeds", DEFAULT_SEEDS)
    cfg = {**cfg, "psiloqa_max_new_tokens": _p95_answer_tokens(tokenizer)}
    for dataset in DATASETS:
        per_item = []
        for language in ("en", "hi") if dataset == "mtrex" else ("en",):
            records = load_records(dataset, language, MAM_TIMING_ITEMS)
            per_item.append(_timed(_run_mam, dataset, language, records, model, tokenizer, prompts) / len(records))
        throughput["mam"][dataset] = max(per_item)  # the slower language: a conservative plan
        # Trains on a few items; the 1,001 steps cost the same however many items there are.
        throughput["clf"][dataset] = _timed(_run_classifier, dataset, "en", seeds[:1]) * len(seeds)
        records = load_records(dataset, "en", SAMPLING_TIMING_ITEMS)
        sampling, scoring, chunk = _time_sampling(dataset, records, model, tokenizer, prompts, sbert, cfg)
        cfg = {**cfg, "sample_chunk": chunk}
        throughput["sampling"][dataset], throughput["scoring"][dataset] = sampling + scoring, scoring
    return throughput, cfg


def _print_plan(throughput, limits, counts, num_languages, eta_hours):
    print(f"{'dataset':9}{'MAM s/item':>12}{'classifier s/lang':>19}{'sampling s/item':>17}{'limit':>8}{'languages':>11}")
    for d in DATASETS:
        print(f"{d:9}{throughput['mam'][d]:12.2f}{throughput['clf'][d]:19.1f}{throughput['sampling'][d]:17.2f}"
              f"{'all' if limits[d] is None else limits[d]:>8}{num_languages[d]:11}")
    print(f"Total ETA: {eta_hours:.1f} h")


def run_timing(args, cfg, loader, sbert_loader):
    root = shared.RESULTS / "raw" / "timing"
    shutil.rmtree(root, ignore_errors=True)
    shared.RESULTS = root  # restored by main
    model, tokenizer = loader()
    prompts = shared.load_prompts()
    throughput, cfg = _measure(cfg, model, tokenizer, prompts, sbert_loader(_device()))
    counts = {d: max(len(rows) for rows in _processed(d).values()) for d in DATASETS}  # xfactr counts differ: use the largest
    num_languages = {d: len(languages(d, cfg)) for d in DATASETS}
    limits = plan_sizes(throughput, counts, args.hours, num_languages)
    eta = (_fixed_seconds(throughput, counts, num_languages)
           + _sampling_seconds(throughput, counts, num_languages, limits)) / 3600
    cfg = {**cfg, "sampling_limit": limits,
           "measured": {"date": _now(), "hours": args.hours, "eta_hours": round(eta, 2), "throughput": throughput}}
    shared.write_atomic(args.config, CONFIG_HEADER + yaml.safe_dump(cfg, sort_keys=False))
    _print_plan(throughput, limits, counts, num_languages, eta)
    return 0


def run_verify(args, cfg, loader, sbert_loader):
    env = {**os.environ, "G6_PARTIAL": "1" if args.partial else "0"}
    code = subprocess.run([sys.executable, "-m", "pytest", "tests/test_results.py", "-q"], cwd=ROOT, env=env).returncode
    flags = ROOT / "results" / "flags.txt"  # written by tests/test_results.py
    if flags.exists():
        print("results/flags.txt:\n" + (flags.read_text(encoding="utf-8") or "(no flags)"))
    return code


COMMANDS = {"smoke": run_smoke, "timing": run_timing, "full": run_full, "verify": run_verify}


def _parse(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--hours", type=float, help="time budget for timing")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--results", default=str(shared.RESULTS))
    parser.add_argument("--no-gpu-tests", action="store_true", help="smoke: skip pytest -m gpu")
    parser.add_argument("--partial", action="store_true", help="verify: a run stopped early, unfinished counts only warn")
    args = parser.parse_args(argv)
    if args.command == "timing" and args.hours is None:
        parser.error("timing needs --hours")
    return args


def main(argv=None, loader=shared.load_generator, sbert_loader=sem.load_sbert) -> int:
    for stream in (sys.stdout, sys.stderr):  # Hindi and Urdu answers crash a cp1252 console
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = _parse(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    if args.command == "full" and "measured" not in cfg:
        print("configs/run.yaml has no measured sizes: run timing first "
              "(python scripts/run_phase3.py timing --hours H).")
        return 2
    results, original = Path(args.results), shared.RESULTS
    shared.RESULTS = results
    try:
        if args.command != "verify":  # verify may run beside a job that has yet to see the pause, and must not take it
            shared.check_pause()
        log_run(results, " ".join(sys.argv[1:] if argv is None else argv))
        return COMMANDS[args.command](args, cfg, loader, lru_cache(maxsize=None)(sbert_loader))
    except shared.Paused:
        shared.PAUSE_FILE.unlink(missing_ok=True)
        print("Paused: everything finished so far is saved. Run the same command to continue.")
        return PAUSED_EXIT
    except KeyboardInterrupt:
        print("Interrupted: rerun the same command to resume.")
        return 130
    finally:
        shared.RESULTS = original


if __name__ == "__main__":
    sys.exit(main())
