"""Tests for `CompressedKVCache.has_persistent_dense_recon()` (B0's minimal
assertion hook, added for the FlashJoLT fused-decode-attention integration
in `experiments/downstream/ruler_hard.py`'s `--attention-path fused`).

The hook must report False right after compression (nothing reconstructed
yet) and True once the ordinary (non-fused) decode path has actually
materialized a dense per-layer reconstruction, so a caller asserting the
"fused decode never materializes a persistent dense reconstruction"
invariant gets a real signal, not a hook that is always False.

Drives the cache manually with synthetic tensors (no model, CPU) -- mirrors
`test_compressed_cache_decode_path.py`'s harness pattern at a smaller size.

Run: .venv/bin/python -m pytest tests/test_compressed_cache_persistent_recon.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kv_compress.integration import CompressedCacheConfig, CompressedKVCache

N_LAYERS = 2
N_KV = 2
T_PREFILL = 32
D_H = 16


def _synth_cos_sin(seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


def _synth_layer(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(N_KV, T_PREFILL, 4, generator=g)
    b = torch.randn(N_KV, 4, D_H, generator=g)
    k = (a @ b + 0.05 * torch.randn(N_KV, T_PREFILL, D_H, generator=g)).unsqueeze(0)
    a2 = torch.randn(N_KV, T_PREFILL, 4, generator=g)
    b2 = torch.randn(N_KV, 4, D_H, generator=g)
    v = (a2 @ b2 + 0.05 * torch.randn(N_KV, T_PREFILL, D_H, generator=g)).unsqueeze(0)
    return k, v


def _build_compressed_cache(ratio: float = 2.0) -> CompressedKVCache:
    cfg = CompressedCacheConfig(target_compression_ratio=ratio)
    cache = CompressedKVCache(cfg, model=None, attach=False)
    for i in range(N_LAYERS):
        k, v = _synth_layer(seed=200 + i)
        cache.inject_prefill_layer(i, k, v)
    cos, sin = _synth_cos_sin(T_PREFILL, D_H)
    cache.inject_prefill_cos_sin(cos, sin)
    cache.force_compress()
    return cache


def test_no_persistent_recon_immediately_after_compression():
    cache = _build_compressed_cache()
    assert cache.has_persistent_dense_recon() is False


def test_persistent_recon_appears_after_ordinary_dense_decode_step():
    """The non-fused decode path (_handle_decode_layer -> _decode_ready_layer)
    DOES populate the layer-level recon cache -- this is the exact thing the
    fused path documents itself as never doing (see
    flashjolt_attention.py's module docstring). Confirms the hook actually
    detects the state it exists to detect, not just returning False always.
    """
    cache = _build_compressed_cache()
    g = torch.Generator().manual_seed(7)
    for layer_idx in range(N_LAYERS):
        k_new = torch.randn(1, N_KV, 1, D_H, generator=g)
        v_new = torch.randn(1, N_KV, 1, D_H, generator=g)
        cache.update(k_new, v_new, layer_idx)
    assert cache.has_persistent_dense_recon() is True


def test_reset_clears_persistent_recon():
    cache = _build_compressed_cache()
    g = torch.Generator().manual_seed(7)
    for layer_idx in range(N_LAYERS):
        cache.update(
            torch.randn(1, N_KV, 1, D_H, generator=g),
            torch.randn(1, N_KV, 1, D_H, generator=g),
            layer_idx,
        )
    assert cache.has_persistent_dense_recon() is True
    cache.reset()
    assert cache.has_persistent_dense_recon() is False
