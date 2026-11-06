"""scripts/run_all.ps1 end to end, with Python and the pipeline replaced by small fakes (Windows only)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="the unattended script is for the Windows GPU machine")

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_all.ps1"
FAKE_PIPELINE = """
import pathlib, sys
stage = sys.argv[1]
root = pathlib.Path(__file__).resolve().parents[1]
with open(root / "calls.txt", "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")
calls = (root / "calls.txt").read_text().split("\\n")
fail = (root / "fail.txt").read_text().split() if (root / "fail.txt").exists() else []
startup = root.parent / "startup" / "g6_run.cmd"
(root / "startup_seen.txt").write_text(str(startup.exists()) + "\\n" + (startup.read_text() if startup.exists() else ""))
if (root / "pause_stage.txt").exists() and (root / "pause_stage.txt").read_text().strip() == stage:
    sys.exit(75)
if stage == "timing":
    (root / "configs" / "run.yaml").write_text("measured: 1\\n")
if stage == "full":
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "out.txt").write_text(str(len(calls)))
    if fail.count("full") >= sum(c.startswith("full") for c in calls):
        sys.exit(1)
elif stage in fail:
    sys.exit(1)
"""


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    remote, root = tmp_path / "remote.git", tmp_path / "repo"
    _git(tmp_path, "init", "--bare", str(remote))
    root.mkdir()
    (root / "scripts").mkdir()
    (root / "configs").mkdir()
    shutil.copy(SCRIPT, root / "scripts" / "run_all.ps1")
    (root / "scripts" / "bootstrap.py").write_text("import sys\n", encoding="utf-8")
    (root / "scripts" / "run_phase3.py").write_text(FAKE_PIPELINE, encoding="utf-8")
    (root / "configs" / "run.yaml").write_text("mam_limit: null\n", encoding="utf-8")
    venv = root / ".venv" / "Scripts"
    venv.mkdir(parents=True)
    (venv / "python.cmd").write_text(f'@"{sys.executable}" %*\n', encoding="utf-8")
    (root / ".gitignore").write_text(".venv/\ncalls.txt\nfail.txt\npause_stage.txt\nstartup_seen.txt\nresults/raw/\n", encoding="utf-8")
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "t")
    _git(root, "config", "user.email", "t@t")
    _git(root, "remote", "add", "origin", str(remote))
    _git(root, "add", ".")
    _git(root, "commit", "-m", "start")
    _git(root, "push", "-u", "origin", "main")
    return root


def _startup(root):
    return root.parent / "startup" / "g6_run.cmd"


def _run(root, *extra):
    _startup(root).parent.mkdir(exist_ok=True)
    command = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(root / "scripts" / "run_all.ps1"),
               "-Hours", "24", "-RetryWaitSeconds", "0", "-StartupDir", str(_startup(root).parent), *extra]
    return subprocess.run(command, cwd=root, capture_output=True, text=True, env=_env(), input="\n", timeout=120)


def _env():
    return {**os.environ, "HF_TOKEN": "x"}


def _calls(root):
    return (root / "calls.txt").read_text().split("\n")[:-1]


def test_runs_every_stage_in_order_and_pushes_the_results(repo):
    done = _run(repo)
    assert done.returncode == 0, done.stdout + done.stderr
    assert _calls(repo) == ["smoke", "timing --hours 24", "full", "verify"]
    pushed = subprocess.run(["git", "ls-tree", "-r", "--name-only", "origin/main"], cwd=repo, capture_output=True, text=True)
    assert "results/out.txt" in pushed.stdout


def test_restarts_full_after_a_crash(repo):
    (repo / "fail.txt").write_text("full full", encoding="utf-8")
    done = _run(repo)
    assert done.returncode == 0, done.stdout + done.stderr
    assert _calls(repo).count("full") == 3 and _calls(repo)[-1] == "verify"


def test_gives_up_on_full_after_the_retry_limit_without_pushing(repo):
    (repo / "fail.txt").write_text(" ".join(["full"] * 50), encoding="utf-8")
    done = _run(repo)
    assert done.returncode != 0
    assert _calls(repo).count("full") == 5 and "verify" not in _calls(repo)


def test_a_failed_smoke_stops_before_the_long_run(repo):
    (repo / "fail.txt").write_text("smoke", encoding="utf-8")
    done = _run(repo)
    assert done.returncode != 0 and _calls(repo) == ["smoke"]


def test_a_rerun_after_timing_goes_straight_to_full_and_keeps_the_sizes(repo):
    (repo / "configs" / "run.yaml").write_text("measured: 7\n", encoding="utf-8")
    done = _run(repo)
    assert done.returncode == 0, done.stdout + done.stderr
    assert _calls(repo) == ["full", "verify"]
    assert (repo / "configs" / "run.yaml").read_text() == "measured: 7\n"


def test_a_passed_smoke_is_not_run_again_after_a_stop_during_timing(repo):
    marker = repo / "results" / "raw" / "smoke" / "passed"
    marker.parent.mkdir(parents=True)
    marker.touch()
    done = _run(repo)
    assert done.returncode == 0, done.stdout + done.stderr
    assert _calls(repo) == ["timing --hours 24", "full", "verify"]


def test_a_pause_during_full_stops_without_retrying_verifying_or_pushing(repo):
    (repo / "pause_stage.txt").write_text("full", encoding="utf-8")
    done = _run(repo)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "=== PAUSED" in done.stdout
    assert _calls(repo)[-1] == "full" and _calls(repo).count("full") == 1


@pytest.mark.parametrize("stage", ["smoke", "timing"])
def test_a_pause_during_an_early_stage_stops_before_the_next_one(repo, stage):
    (repo / "pause_stage.txt").write_text(stage, encoding="utf-8")
    done = _run(repo)
    assert done.returncode == 0 and "=== PAUSED" in done.stdout
    assert _calls(repo)[-1].split()[0] == stage


def test_the_startup_file_exists_during_the_run_and_restarts_it_with_the_same_hours(repo):
    (repo / "fail.txt").write_text("verify", encoding="utf-8")
    _run(repo)
    seen = (repo / "startup_seen.txt").read_text()
    assert seen.startswith("True") and "-AutoResumed" in seen and "-Hours 24" in seen and str(repo) in seen
    assert not _startup(repo).exists()


def test_the_startup_file_is_removed_after_done(repo):
    assert _run(repo).returncode == 0
    assert not _startup(repo).exists()


def test_the_startup_file_is_removed_after_a_pause(repo):
    (repo / "pause_stage.txt").write_text("full", encoding="utf-8")
    _run(repo)
    assert not _startup(repo).exists()


def test_the_startup_file_is_removed_after_a_stop(repo):
    (repo / "fail.txt").write_text("smoke", encoding="utf-8")
    assert _run(repo).returncode != 0
    assert not _startup(repo).exists()


def test_a_second_start_is_refused_while_one_is_running(repo):
    lock = repo / "results" / "raw" / "run_all.lock"
    lock.parent.mkdir(parents=True)
    with open(lock, "a"):
        done = _run(repo)
    assert done.returncode != 0 and "already running" in done.stdout
    assert not (repo / "calls.txt").exists()


def test_a_manual_start_clears_a_leftover_pause(repo):
    pause = repo / "results" / "raw" / "PAUSE"
    pause.parent.mkdir(parents=True)
    pause.touch()
    assert _run(repo).returncode == 0
    assert "verify" in _calls(repo) and not pause.exists()


def test_an_automatic_restart_honours_a_pause_and_starts_nothing(repo):
    pause = repo / "results" / "raw" / "PAUSE"
    pause.parent.mkdir(parents=True)
    pause.touch()
    done = _run(repo, "-AutoResumed")
    assert done.returncode == 0 and "=== PAUSED" in done.stdout
    assert not (repo / "calls.txt").exists() and not _startup(repo).exists()


def test_a_failed_final_commit_stops_instead_of_reporting_done(repo):
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    done = _run(repo)
    assert done.returncode != 0
    assert "=== DONE" not in done.stdout and "=== STOPPED" in done.stdout


def test_a_failed_verify_does_not_push(repo):
    (repo / "fail.txt").write_text("verify", encoding="utf-8")
    done = _run(repo)
    assert done.returncode != 0
    pushed = subprocess.run(["git", "ls-tree", "-r", "--name-only", "origin/main"], cwd=repo, capture_output=True, text=True)
    assert "results/out.txt" not in pushed.stdout
