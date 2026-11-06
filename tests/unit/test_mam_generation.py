"""Unit tests for the G6 helpers added to the two MAM scripts."""
import math

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.detectors import ModelArtifacts_g6 as mam
from src.detectors import ModelArtifacts_multiple_gen_tokens_g6 as mam_multi


def test_entropy_of_uniform_logits_is_log_vocab():
    assert math.isclose(mam.token_entropy(torch.zeros(10)), math.log(10), rel_tol=1e-6)


def test_entropy_of_one_hot_logits_is_zero_and_a_python_float():
    value = mam.token_entropy(torch.tensor([100.0, 0.0, 0.0]))
    assert isinstance(value, float) and abs(value) < 1e-6


def _four_layer_model():
    config = LlamaConfig(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                         num_attention_heads=2)
    return LlamaForCausalLM(config).eval()


def _forward(model):
    with torch.no_grad():
        model(torch.tensor([[1, 2, 3]]))


def test_hooks_sit_only_on_middle_and_last_layer_of_both_scripts():
    for module in (mam, mam_multi):
        model = _four_layer_model()
        module.FULLY_CONNECTED_HIDDEN_LAYERS.clear()
        module.ATTENTION_HIDDEN_LAYERS.clear()
        handles = module.register_g6_hooks(model)
        _forward(model)
        for handle in handles:
            handle.remove()
        assert sorted(module.FULLY_CONNECTED_HIDDEN_LAYERS) == ["model.layers.1.mlp.up_proj",
                                                                "model.layers.3.mlp.up_proj"]
        assert sorted(module.ATTENTION_HIDDEN_LAYERS) == ["model.layers.1.self_attn.o_proj",
                                                          "model.layers.3.self_attn.o_proj"]


def test_removed_hooks_stop_capturing():
    model = _four_layer_model()
    handles = mam.register_g6_hooks(model)
    for handle in handles:
        handle.remove()
    mam.FULLY_CONNECTED_HIDDEN_LAYERS.clear()
    _forward(model)
    assert not mam.FULLY_CONNECTED_HIDDEN_LAYERS
