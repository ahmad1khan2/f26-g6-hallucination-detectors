"""The processed data and the authors' code must not change after Phase 2."""
import hashlib
import subprocess

import pytest

from src.common.io import ROOT

SUBMODULE = "third_party/low-resource-hallucination-detection"
SUBMODULE_COMMIT = "644d655"
SHA256 = {  # line endings normalised: git may check the files out with CRLF
    "mtrex": "c261fcb356a10a1b806796043a376fea517e4dde774f00156931ade651f3b1be",
    "xfactr": "06bef2fbea09124bd1426e8f59b37863d83b6a08e03ed5d83e7977deedb50170",
    "psiloqa": "b993a7249555272ae8ff362cd6c0a0fb0beecb98dd1547c5f3c42fab33c92c66",
}


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


@pytest.mark.parametrize("dataset", SHA256)
def test_processed_file_is_unchanged(dataset):
    data = (ROOT / "data" / "processed" / f"{dataset}.jsonl").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == SHA256[dataset]


def test_the_submodule_is_pinned_at_the_authors_commit():
    mode, commit, *_ = git("ls-files", "-s", SUBMODULE).split()
    assert mode == "160000" and commit.startswith(SUBMODULE_COMMIT)


def test_the_authors_code_has_no_local_changes():
    assert git("status", "--porcelain", "third_party") == ""
