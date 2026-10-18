"""make_jolt_cache end-to-end on tiny random models (CPU, no downloads)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kvsafety.jolt_cache import make_jolt_cache

COMMON = dict(
    vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    max_position_embeddings=2048,
)


def _model(kind: str):
    torch.manual_seed(0)
    if kind == "qwen3":
        return Qwen3ForCausalLM(Qwen3Config(**COMMON)).eval()
    return LlamaForCausalLM(LlamaConfig(**COMMON)).eval()


def _decode(model, ids, past_key_values, steps=5):
    """Prefill ids[:-1], then last token, then greedy decode; returns logits per step."""
    out = []
    with torch.no_grad():
        model(ids[:, :-1], past_key_values=past_key_values, use_cache=True)
        logits = model(ids[:, -1:], past_key_values=past_key_values, use_cache=True).logits[:, -1]
        for _ in range(steps):
            out.append(logits)
            nxt = logits.argmax(-1, keepdim=True)
            logits = model(nxt, past_key_values=past_key_values, use_cache=True).logits[:, -1]
    return torch.stack(out)


@pytest.mark.parametrize("kind", ["qwen3", "llama"])
def test_make_jolt_cache_ratio_4_decode(kind):
    from transformers import DynamicCache

    model = _model(kind)
    ids = torch.randint(0, 128, (1, 200), generator=torch.Generator().manual_seed(1))
    dense = _decode(model, ids, DynamicCache(config=model.config))

    cache = make_jolt_cache(model, 4.0)
    try:
        jolt = _decode(model, ids, cache)
    finally:
        cache.detach()

    assert cache.last_original_bytes and cache.last_compressed_bytes
    assert cache.last_compression_ratio == pytest.approx(4.0, rel=0.35)
    assert torch.isfinite(jolt).all()
    assert not torch.equal(jolt, dense)


def test_compression_time_1024_tokens():
    model = _model("qwen3")
    ids = torch.randint(0, 128, (1, 1024), generator=torch.Generator().manual_seed(1))
    cache = make_jolt_cache(model, 4.0)
    try:
        with torch.no_grad():
            model(ids[:, :-1], past_key_values=cache, use_cache=True)
            t0 = time.perf_counter()
            model(ids[:, -1:], past_key_values=cache, use_cache=True)
            dt = time.perf_counter() - t0
    finally:
        cache.detach()
    print(f"\ncompress+1 token, 1024-token prompt, CPU tiny model: {dt:.2f}s")
    assert cache.last_compression_ratio is not None
