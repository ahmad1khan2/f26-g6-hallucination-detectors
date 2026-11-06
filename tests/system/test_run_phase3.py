"""scripts/run_phase3.py end to end on the tiny random model: smoke, timing, full, a killed run, verify."""
import pytest
import yaml

from scripts import run_phase3 as rp
from src.detectors import selfcheckGPT_g6 as sgm
from src.detectors import shared
from tests.conftest import FakeEncoder
from tests.test_results import check_results

RUN_CONFIG = {
    "mam_limit": 6, "sampling_limit": dict.fromkeys(rp.DATASETS, 4), "psiloqa_max_new_tokens": 6,
    "num_samples": 3, "sample_chunk": 3, "sample_step": 2, "seeds": [42],
    "languages": {"mtrex": ["en", "hi"], "xfactr": ["en", "ja"], "psiloqa": ["en", "hi"]},
    "measured": {"eta_hours": 1},
}
FILES = ("answers.jsonl", "sgm.jsonl", "sem.jsonl")
TINY_NEW_TOKENS = 8  # the tiny model answers in noise; long answers would only cost time


@pytest.fixture(scope="module", autouse=True)
def short_answers():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rp, "MAX_NEW_TOKENS", TINY_NEW_TOKENS)
        yield


@pytest.fixture
def run(tmp_path, tiny_model, tiny_tokenizer, fake_encoder):
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(RUN_CONFIG), encoding="utf-8")

    def call(*command, results=tmp_path / "results", config_path=config):
        return rp.main([*command, "--config", str(config_path), "--results", str(results)],
                       loader=lambda: (tiny_model, tiny_tokenizer), sbert_loader=lambda device: fake_encoder)
    return call


@pytest.fixture(scope="module")
def reference(tmp_path_factory, tiny_model, tiny_tokenizer):
    """One uninterrupted full run, shared by the tests that compare against it."""
    base = tmp_path_factory.mktemp("reference")
    config = base / "run.yaml"
    config.write_text(yaml.safe_dump(RUN_CONFIG), encoding="utf-8")
    code = rp.main(["full", "--config", str(config), "--results", str(base / "results")],
                   loader=lambda: (tiny_model, tiny_tokenizer), sbert_loader=lambda device: FakeEncoder())
    assert code == 0
    return base / "results"


def uids(results):
    """{(dataset, language, file): uids in file order} for every results file."""
    return {(dataset, language, name): [r["uid"] for r in shared.read_results(results / dataset / language / name)]
            for dataset in rp.DATASETS for language in RUN_CONFIG["languages"][dataset] for name in FILES}


def test_full_produces_the_layout_and_check_results_accepts_it(reference):
    for dataset in rp.DATASETS:
        for language in RUN_CONFIG["languages"][dataset]:
            directory = reference / dataset / language
            assert all((directory / name).stat().st_size for name in (*FILES, "run.json"))
            assert (reference / "raw" / dataset / language / "features.pkl").exists()
            assert (directory / "mam_preds_seed42.jsonl").exists()
    check_results(reference, rp.PROCESSED, RUN_CONFIG)
    assert "| full" in (reference / "RUNS.md").read_text(encoding="utf-8")


def test_a_killed_full_run_resumes_to_the_same_uids_without_duplicates(tmp_path, run, reference, monkeypatch):
    real_sample, turns = sgm.sample_schema, []

    def killed_after_two_turns(*args, **kwargs):
        real_sample(*args, **kwargs)
        turns.append(1)
        if len(turns) == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(sgm, "sample_schema", killed_after_two_turns)
    assert run("full") == 130
    monkeypatch.setattr(sgm, "sample_schema", real_sample)
    assert run("full") == 0

    resumed = uids(tmp_path / "results")
    for key, found in resumed.items():
        assert len(found) == len(set(found)), key
    assert resumed == uids(reference)


def _pause_after_two_sampling_turns(tmp_path, monkeypatch):
    pause_file = tmp_path / "PAUSE"
    monkeypatch.setattr(shared, "PAUSE_FILE", pause_file)
    real_sample, turns = sgm.sample_schema, []

    def pausing(*args, **kwargs):
        real_sample(*args, **kwargs)
        turns.append(1)
        if len(turns) == 2:
            pause_file.touch()

    monkeypatch.setattr(sgm, "sample_schema", pausing)
    return pause_file, real_sample


def test_a_paused_full_run_exits_with_the_pause_code_and_resumes_to_the_same_results(tmp_path, run, reference, monkeypatch, capsys):
    pause_file, real_sample = _pause_after_two_sampling_turns(tmp_path, monkeypatch)
    assert run("full") == rp.PAUSED_EXIT
    assert not pause_file.exists()
    assert "Paused" in capsys.readouterr().out
    monkeypatch.setattr(sgm, "sample_schema", real_sample)
    assert run("full") == 0
    assert uids(tmp_path / "results") == uids(reference)
    for dataset in rp.DATASETS:
        for language in RUN_CONFIG["languages"][dataset]:
            for name in ("answers.jsonl", "mam_preds_seed42.jsonl"):
                assert (tmp_path / "results" / dataset / language / name).read_bytes() == \
                    (reference / dataset / language / name).read_bytes()


def test_a_pause_set_before_the_start_does_not_load_the_model(tmp_path, monkeypatch):
    pause_file = tmp_path / "PAUSE"
    monkeypatch.setattr(shared, "PAUSE_FILE", pause_file)
    pause_file.touch()
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(RUN_CONFIG), encoding="utf-8")

    def must_not_load():
        raise AssertionError("the model was loaded")

    code = rp.main(["full", "--config", str(config), "--results", str(tmp_path / "results")], loader=must_not_load)
    assert code == rp.PAUSED_EXIT and not pause_file.exists()


def test_verify_leaves_a_pending_pause_for_the_running_job(tmp_path, monkeypatch):
    pause_file = tmp_path / "PAUSE"
    monkeypatch.setattr(shared, "PAUSE_FILE", pause_file)
    pause_file.touch()
    monkeypatch.setattr(rp, "COMMANDS", {**rp.COMMANDS, "verify": lambda *a: 0})
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(RUN_CONFIG), encoding="utf-8")
    assert rp.main(["verify", "--config", str(config), "--results", str(tmp_path / "results")]) == 0
    assert pause_file.exists()


def test_a_power_cut_leaving_torn_files_resumes_to_the_same_results(tmp_path, run, reference, monkeypatch):
    real_sample, turns = sgm.sample_schema, []

    def killed(*args, **kwargs):
        real_sample(*args, **kwargs)
        turns.append(1)
        if len(turns) == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(sgm, "sample_schema", killed)
    assert run("full") == 130
    monkeypatch.setattr(sgm, "sample_schema", real_sample)
    results = tmp_path / "results"
    for path in [*results.glob("raw/*/*/samples.jsonl"), *results.glob("*/*/sgm.jsonl"), *results.glob("*/*/sem.jsonl"),
                 *results.glob("*/*/answers.jsonl")]:
        data = path.read_bytes()
        if data:
            path.write_bytes(data[:-3] + b"\0" * 8)  # the last row is torn, with a zero-filled tail
    for path in results.glob("raw/*/*/features.pkl"):
        with open(path, "ab") as f:
            f.write(b"\x80\x04\x95half")
    assert run("full") == 0

    resumed = uids(results)
    for key, found in resumed.items():
        assert len(found) == len(set(found)), key
    assert {k: set(v) for k, v in resumed.items()} == {k: set(v) for k, v in uids(reference).items()}
    check_results(results, rp.PROCESSED, RUN_CONFIG)


def test_timing_writes_the_measured_sizes_into_the_config(tmp_path, run, capsys):
    fresh = tmp_path / "fresh.yaml"
    fresh.write_text(yaml.safe_dump({k: v for k, v in RUN_CONFIG.items() if k != "measured"}), encoding="utf-8")

    assert run("timing", "--hours", "30", config_path=fresh) == 0

    written = yaml.safe_load(fresh.read_text(encoding="utf-8"))
    assert set(written["sampling_limit"]) == set(rp.DATASETS) and "measured" in written
    assert set(written["measured"]["throughput"]) >= {"mam", "clf", "sampling"}
    assert written["psiloqa_max_new_tokens"] > 0 and "Total ETA" in capsys.readouterr().out


def test_smoke_passes_on_the_tiny_model_and_writes_examples(tmp_path, run, monkeypatch):
    monkeypatch.setattr(rp, "SMOKE_STEPS", (3, 6))  # the real 10 and 20 items would take minutes here
    assert run("smoke", "--no-gpu-tests") == 0

    examples = (tmp_path / "results" / "raw" / "smoke" / "examples.txt").read_text(encoding="utf-8")
    assert "[mtrex/bn]" in examples and "[psiloqa/ar]" in examples


def test_verify_exits_with_the_exit_code_of_test_results(run):
    assert run("verify") == 0
