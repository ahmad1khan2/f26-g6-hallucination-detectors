"""Real-model checks for the GPU machine: run with `pytest -m gpu`. Skipped without CUDA."""
import math

import pytest
import torch

from src.common.schema import read_jsonl
from src.detectors import ModelArtifacts_g6 as mam
from src.detectors import selfcheckGPT_g6 as sgm
from src.detectors import semanticEntropy_g6 as sem
from src.detectors import shared
from src.common.io import ROOT

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")]
PEAK_VRAM_GIB = 15


@pytest.fixture(scope="module")
def generator():
    return shared.load_generator()


@pytest.fixture(scope="module")
def records():
    rows = [r for r in read_jsonl(ROOT / "data" / "processed" / "mtrex.jsonl") if r.language == "en"]
    return rows[:2]


@pytest.fixture(scope="module")
def prompts():
    return shared.load_prompts()


def test_the_model_loads_in_4_bit_on_cuda(generator):
    model, _ = generator
    assert model.device.type == "cuda"
    assert model.config.quantization_config.to_dict()["load_in_4bit"]


def test_an_mtrex_question_gets_a_non_empty_answer(generator, records, prompts):
    model, tokenizer = generator
    inputs = tokenizer(shared.chat_text(records[0], tokenizer, prompts), return_tensors="pt").to(model.device)
    out = model.generate(**inputs, do_sample=False, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id)
    assert tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def test_mam_features_have_the_llama_widths(generator, records, prompts, tmp_path):
    model, tokenizer = generator
    mam.run_schema(records, model, tokenizer, prompts, tmp_path / "out", tmp_path / "raw")
    features = shared.read_features(tmp_path / "raw" / "features.pkl")
    assert len(features) == 2
    assert {tuple(f["first_fully_connected"].shape) for f in features} == {(2, 14336)}
    assert {tuple(f["first_attention"].shape) for f in features} == {(2, 4096)}


def test_sampling_gives_finite_log_likelihoods(generator, records, prompts, tmp_path):
    model, tokenizer = generator
    sgm.sample_schema(records, model, tokenizer, prompts, tmp_path, num_samples=2, max_new_tokens=10, chunk=2)
    rows = shared.read_results(tmp_path / "samples.jsonl")
    assert len(rows) == 2
    assert all(math.isfinite(x) for row in rows for sample in row["log_liks"] for x in sample)


def test_labse_encodes_text():
    encoded = sem.load_sbert("cuda").encode(["Paris", "Parigi"], convert_to_tensor=True)
    assert encoded.shape[0] == 2 and torch.isfinite(encoded).all()


def test_peak_vram_stays_under_the_16_gb_card(generator):
    assert torch.cuda.max_memory_allocated() / 2**30 < PEAK_VRAM_GIB
