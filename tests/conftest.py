"""Shared offline fixtures: tiny random Llama, byte-level tokenizer, fake sentence encoder, small records."""
import hashlib

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from src.common.schema import Record

CHAT_TEMPLATE = (
    "{% for m in messages %}<|start|>{{ m['role'] }}<|sep|>{{ m['content'] }}<|eot_id|>{% endfor %}"
    "{% if add_generation_prompt %}<|start|>assistant<|sep|>{% endif %}"
)


def _byte_tokenizer():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|eot_id|>",
                                  additional_special_tokens=["<|start|>", "<|sep|>"], padding_side="left")
    tok.pad_token_id = tok.eos_token_id
    tok.chat_template = CHAT_TEMPLATE
    return tok


@pytest.fixture(scope="session")
def tiny_tokenizer():
    return _byte_tokenizer()


@pytest.fixture(scope="session")
def tiny_model(tiny_tokenizer):
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=len(tiny_tokenizer), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=4096,
                         eos_token_id=tiny_tokenizer.eos_token_id, pad_token_id=tiny_tokenizer.eos_token_id)
    return LlamaForCausalLM(config).eval()


class FakeEncoder:
    """Deterministic stand-in for LaBSE: same text, same vector; no download."""

    def encode(self, texts, convert_to_tensor=True):
        def vector(text):
            seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)
            return np.random.default_rng(seed).normal(size=16)
        return torch.tensor(np.stack([vector(t) for t in texts]), dtype=torch.float32)


@pytest.fixture
def fake_encoder():
    return FakeEncoder()


@pytest.fixture
def fake_prompts():
    return {
        "mtrex": {"English": {"system": "Answer in English. \n",
                              "capitals": {"user": "Question: What is the capital of {source}? \n",
                                           "asst": "Answer (in English): "}}},
        "xfactr": {"en": {"system": "Fill the blank.", "user": "Sentence: {sentence}", "asst": "Answer: "}},
        "psiloqa": {"en": {"system": "Answer from the passage.", "user": "Passage: {passage}\nQuestion: {question}"}},
    }


@pytest.fixture
def mtrex_records():
    return [Record(uid=f"mtrex-en-capitals-{i:06d}", dataset="mtrex", language="en",
                   prompt=f"Question: What is the capital of {c}?", relation="capitals",
                   gold=[g], meta={"source": c})
            for i, (c, g) in enumerate([("Somalia", "mogadishu"), ("France", "paris"), ("Peru", "lima")])]


@pytest.fixture
def xfactr_records():
    return [Record(uid=f"xfactr-en-P37-{i:06d}", dataset="xfactr", language="en", prompt=f"{s} speaks ___ .",
                   relation="P37", gold=[g], meta={"subject": s})
            for i, (s, g) in enumerate([("Spain", "spanish"), ("Italy", "italian"), ("Japan", "japanese")])]


@pytest.fixture
def psiloqa_records():
    rows = [("Who wrote Hamlet?", "Shakespeare wrote Hamlet.", []),
            ("Who wrote Hamlet?", "Marlowe wrote Hamlet.", [{"start": 0, "end": 7}]),
            ("What is 2+2?", "It is five.", [{"start": 6, "end": 10}])]
    return [Record(uid=f"psiloqa-en-{i:06d}", dataset="psiloqa", language="en", prompt=q, answer=a, spans=s,
                   label=int(bool(s)), meta={"passage": "Short passage.", "model": "m"})
            for i, (q, a, s) in enumerate(rows)]
