"""Unit tests for the pure parts of scripts/check_gpu.py (no GPU needed)."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_gpu.py"
spec = importlib.util.spec_from_file_location("check_gpu", SCRIPT)
check_gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_gpu)


def test_answer_one_returns_a_string_and_is_greedy(tiny_model, tiny_tokenizer, mtrex_records, fake_prompts):
    first = check_gpu.answer_one(tiny_model, tiny_tokenizer, mtrex_records[0], fake_prompts)
    again = check_gpu.answer_one(tiny_model, tiny_tokenizer, mtrex_records[0], fake_prompts)
    assert isinstance(first, str) and first == again


def test_require_cuda_exits_with_a_clear_message(monkeypatch):
    monkeypatch.setattr(check_gpu.torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as exc:
        check_gpu.require_cuda()
    assert "CUDA" in str(exc.value) and "driver" in str(exc.value)


def test_log_versions_creates_results_on_a_fresh_clone_and_appends(tmp_path):
    path = tmp_path / "results" / "README.md"
    check_gpu.log_versions(path, "PC-1", {"torch": "1"})
    check_gpu.log_versions(path, "PC-2", {"torch": "2"})
    text = path.read_text(encoding="utf-8")
    assert "## PC-1" in text and "## PC-2" in text


def test_answer_one_tokenizes_like_the_authors(tiny_tokenizer, tiny_model, mtrex_records, fake_prompts, monkeypatch):
    seen = {}
    real_call = type(tiny_tokenizer).__call__

    def spy(self, *args, **kwargs):
        seen.update(kwargs)
        return real_call(self, *args, **kwargs)

    monkeypatch.setattr(type(tiny_tokenizer), "__call__", spy)
    check_gpu.answer_one(tiny_model, tiny_tokenizer, mtrex_records[0], fake_prompts)
    assert "add_special_tokens" not in seen


def test_versions_section_names_the_machine_and_versions():
    text = check_gpu.versions_section("PC-1", {"torch": "2.15.0", "GPU": "RTX A4000"})
    assert text.startswith("## PC-1\n") and "- torch: 2.15.0" in text and "- GPU: RTX A4000" in text
