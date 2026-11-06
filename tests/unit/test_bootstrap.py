"""Unit tests for the pure helpers of scripts/bootstrap.py."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "bootstrap.py"
spec = importlib.util.spec_from_file_location("bootstrap", SCRIPT)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)

SMI = """
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 551.61       Driver Version: 551.61       CUDA Version: 12.4     |
+-----------------------------------------------------------------------------+
"""


def test_parse_cuda_version_reads_the_header_line():
    assert bootstrap.parse_cuda_version(SMI) == (12, 4)


def test_parse_cuda_version_fails_clearly_without_the_field():
    with pytest.raises(RuntimeError, match="CUDA Version"):
        bootstrap.parse_cuda_version("NVIDIA-SMI has failed")


@pytest.mark.parametrize("version,tag", [((13, 4), "cu134"), ((13, 3), "cu132"), ((13, 0), "cu130"),
                                         ((12, 9), "cu128"), ((12, 6), "cu126")])
def test_torch_index_picks_highest_not_above_driver(version, tag):
    assert bootstrap.torch_index(version) == f"https://download.pytorch.org/whl/{tag}"


def test_torch_index_rejects_an_old_driver():
    with pytest.raises(RuntimeError, match="driver"):
        bootstrap.torch_index((11, 8))


def test_torch_pin_returns_the_pinned_version_without_local_tag():
    assert bootstrap.torch_pin("numpy==2.0\ntorch==2.15.0+cu132\nx==1\n") == "torch==2.15.0"
    assert bootstrap.torch_pin("torch==2.15.0\n") == "torch==2.15.0"


def test_torch_pin_is_none_without_a_pin():
    assert bootstrap.torch_pin("numpy==2.0\ntorchvision==0.1\n# torch==1\n") is None


def test_install_torch_falls_back_to_the_index_newest_when_the_pin_has_no_wheel(monkeypatch, capsys):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "torch==2.15.0" in command:
            raise bootstrap.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(bootstrap, "run", fake_run)
    bootstrap.install_torch("py", "torch==2.15.0", "https://index")
    assert ["torch==2.15.0" in c for c in calls] == [True, False] and "torch" in calls[1]
    assert "torch==2.15.0" in capsys.readouterr().out


def test_install_torch_installs_once_when_the_pin_works(monkeypatch):
    calls = []
    monkeypatch.setattr(bootstrap, "run", lambda command, **kwargs: calls.append(command))
    bootstrap.install_torch("py", "torch==2.15.0", "https://index")
    assert len(calls) == 1


def test_install_torch_without_a_pin_still_fails_loudly(monkeypatch):
    def fake_run(command, **kwargs):
        raise bootstrap.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(bootstrap, "run", fake_run)
    with pytest.raises(bootstrap.subprocess.CalledProcessError):
        bootstrap.install_torch("py", None, "https://index")


def test_venv_python_matches_the_platform(tmp_path):
    path = bootstrap.venv_python(tmp_path)
    expected = "Scripts/python.exe" if bootstrap.sys.platform == "win32" else "bin/python"
    assert path == tmp_path / ".venv" / Path(expected)
