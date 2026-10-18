"""Tests for decode-ready cache reuse and release behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kv_compress.integration.compressed_cache as cc_mod
from kv_compress.integration import CompressedCacheConfig, CompressedKVCache
from kv_compress.integration.pre_rope_capture import apply_rope_to_k

N_LAYERS = 4
N_KV = 2
T_PREFILL = 64
D_H = 16


def _synth_cos_sin(seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


def _synth_layer(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Low-rank-ish K/V so the allocator has structure to exploit."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(N_KV, T_PREFILL, 4, generator=g)
    b = torch.randn(N_KV, 4, D_H, generator=g)
    noise = 0.05 * torch.randn(N_KV, T_PREFILL, D_H, generator=g)
    k = (a @ b + noise).unsqueeze(0)
    a2 = torch.randn(N_KV, T_PREFILL, 4, generator=g)
    b2 = torch.randn(N_KV, 4, D_H, generator=g)
    noise2 = 0.05 * torch.randn(N_KV, T_PREFILL, D_H, generator=g)
    v = (a2 @ b2 + noise2).unsqueeze(0)
    return k, v


def _build_compressed_cache(ratio: float = 2.0) -> CompressedKVCache:
    cfg = CompressedCacheConfig(target_compression_ratio=ratio)
    cache = CompressedKVCache(cfg, model=None, attach=False)
    for i in range(N_LAYERS):
        k, v = _synth_layer(seed=100 + i)
        cache.inject_prefill_layer(i, k, v)
    cos, sin = _synth_cos_sin(T_PREFILL, D_H)
    cache.inject_prefill_cos_sin(cos, sin)
    cache.force_compress()
    return cache


def _reference_prefill_recon(
    cache: CompressedKVCache, layer_idx: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent decode-ready computation from the compressed state."""
    k_pre, v_recon = cache._reconstruct_layer(layer_idx)
    cos = cache._prefill_cos.to(dtype)
    sin = cache._prefill_sin.to(dtype)
    k_post = apply_rope_to_k(k_pre.to(dtype), cos, sin)
    return k_post, v_recon.to(dtype)


def _decode_step(
    cache: CompressedKVCache, step_seed: int, dtype: torch.dtype = torch.float32
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    g = torch.Generator().manual_seed(step_seed)
    outs = []
    for layer_idx in range(N_LAYERS):
        k_new = torch.randn(1, N_KV, 1, D_H, generator=g).to(dtype)
        v_new = torch.randn(1, N_KV, 1, D_H, generator=g).to(dtype)
        outs.append(cache.update(k_new, v_new, layer_idx))
    return outs


def test_decode_outputs_match_reference() -> None:
    cache = _build_compressed_cache()
    outs = _decode_step(cache, step_seed=1)
    for layer_idx, (k_full, v_full) in enumerate(outs):
        k_ref, v_ref = _reference_prefill_recon(cache, layer_idx, torch.float32)
        assert k_full.shape == (1, N_KV, T_PREFILL + 1, D_H)
        assert torch.allclose(k_full[..., :T_PREFILL, :], k_ref, rtol=0, atol=1e-6)
        assert torch.allclose(v_full[..., :T_PREFILL, :], v_ref, rtol=0, atol=1e-6)
        assert torch.equal(
            k_full[..., T_PREFILL:, :], cache._tail_k[layer_idx]
        )


def test_cache_hit_path_bitwise_stable_across_steps() -> None:
    cache = _build_compressed_cache()
    outs1 = _decode_step(cache, step_seed=1)
    outs2 = _decode_step(cache, step_seed=2)
    for (k1, v1), (k2, v2) in zip(outs1, outs2):
        assert torch.equal(k1[..., :T_PREFILL, :], k2[..., :T_PREFILL, :])
        assert torch.equal(v1[..., :T_PREFILL, :], v2[..., :T_PREFILL, :])
    assert cache.get_seq_length() == T_PREFILL + 2


def test_group_caches_released_after_full_pass() -> None:
    cache = _build_compressed_cache()
    _decode_step(cache, step_seed=1)
    assert cache._recon_cache_k == {}, "fp32 group K recon cache not released"
    assert cache._recon_cache_v == {}, "fp32 group V recon cache not released"
    assert cache._group_layers_pending == {}
    assert sorted(cache._recon_cache_layer.keys()) == list(range(N_LAYERS))
    for k_ready, v_ready in cache._recon_cache_layer.values():
        assert k_ready.dtype == torch.float32
        assert v_ready.dtype == torch.float32
        assert k_ready.shape == (N_KV, T_PREFILL, D_H)


def test_decode_ready_cache_uses_attention_dtype() -> None:
    cache = _build_compressed_cache()
    _decode_step(cache, step_seed=1, dtype=torch.bfloat16)
    for k_ready, v_ready in cache._recon_cache_layer.values():
        assert k_ready.dtype == torch.bfloat16
        assert v_ready.dtype == torch.bfloat16
    assert cache._cos_sin_cast is not None
    assert cache._cos_sin_cast[0] == torch.bfloat16
    k_full, _ = cache.update(
        torch.randn(1, N_KV, 1, D_H).to(torch.bfloat16),
        torch.randn(1, N_KV, 1, D_H).to(torch.bfloat16),
        0,
    )
    k_ref, _ = _reference_prefill_recon(cache, 0, torch.bfloat16)
    assert torch.equal(k_full[..., :T_PREFILL, :], k_ref)


def test_long_context_gate_skips_layer_cache() -> None:
    saved = cc_mod.LAYER_RECON_CACHE_MAX_T
    cc_mod.LAYER_RECON_CACHE_MAX_T = T_PREFILL // 2  # force the gate
    try:
        cache = _build_compressed_cache()
        outs = _decode_step(cache, step_seed=1)
        assert cache._recon_cache_layer == {}, "gate must skip layer caching"
        assert cache._recon_cache_k != {}, "group caches must stay resident"
        assert cache._group_layers_pending != {}
        k_ref, v_ref = _reference_prefill_recon(cache, 0, torch.float32)
        k_full, v_full = outs[0]
        assert torch.allclose(k_full[..., :T_PREFILL, :], k_ref, rtol=0, atol=1e-6)
        assert torch.allclose(v_full[..., :T_PREFILL, :], v_ref, rtol=0, atol=1e-6)
    finally:
        cc_mod.LAYER_RECON_CACHE_MAX_T = saved


def test_group_caches_released_again_after_dtype_change_rebuild() -> None:
    """Rebuilds released group caches when dtype changes."""
    cache = _build_compressed_cache()
    _decode_step(cache, step_seed=1)  # fp32 pass: build + release
    assert cache._recon_cache_k == {}
    dev = cache._prefill_cos.device
    for layer_idx in range(N_LAYERS):
        cache._decode_ready_layer(layer_idx, torch.bfloat16, dev)
    assert cache._recon_cache_k == {}, (
        "fp32 group K cache resident after dtype-change rebuild"
    )
    assert cache._recon_cache_v == {}
    for k_ready, _v in cache._recon_cache_layer.values():
        assert k_ready.dtype == torch.bfloat16




# Minimal attached model to exercise prompt-specific RoPE capture.
def apply_rotary_pos_emb(q, k, cos, sin, *args, **kwargs):
    return q, k


class _H1FakeAttn(torch.nn.Module):
    def __init__(self, hidden_size: int, n_kv: int, head_dim: int):
        super().__init__()
        self.k_proj = torch.nn.Linear(hidden_size, n_kv * head_dim, bias=False)


class _H1FakeLayer:
    def __init__(self, self_attn):
        self.self_attn = self_attn


class _H1FakeInnerModel:
    def __init__(self, layers):
        self.layers = layers


class _H1FakeConfig:
    def __init__(self, num_key_value_heads: int, head_dim: int):
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim


class _H1FakeModel:
    def __init__(self, hidden_size: int, n_kv: int, head_dim: int):
        self.model = _H1FakeInnerModel([_H1FakeLayer(_H1FakeAttn(hidden_size, n_kv, head_dim))])
        self.config = _H1FakeConfig(n_kv, head_dim)


def test_reset_captures_fresh_rope_table_not_stale_one() -> None:
    hidden_size, n_kv, head_dim = 8, 2, 4
    model = _H1FakeModel(hidden_size, n_kv, head_dim)
    cfg = CompressedCacheConfig(target_compression_ratio=2.0)
    cache = CompressedKVCache(cfg, model=model, attach=True)

    def _do_prefill(T: int, cos_val: float) -> None:
        attn = model.model.layers[0].self_attn
        hidden_states = torch.randn(1, T, hidden_size)
        attn.k_proj(hidden_states)  # fires the pre-RoPE-K capture hook
        cos = torch.full((1, T, head_dim), cos_val)
        sin = torch.zeros((1, T, head_dim))
        dummy = torch.zeros(1, T, head_dim)
        apply_rotary_pos_emb(dummy, dummy, cos, sin)
        k_post = torch.randn(1, n_kv, T, head_dim)
        v = torch.randn(1, n_kv, T, head_dim)
        cache.update(k_post, v, 0)

    try:
        _do_prefill(T=5, cos_val=1.0)
        assert cache._prefill_cos is not None
        assert torch.equal(cache._prefill_cos, torch.full((1, 5, head_dim), 1.0))

        cache.reset()
        _do_prefill(T=5, cos_val=2.0)

        assert torch.equal(cache._prefill_cos, torch.full((1, 5, head_dim), 2.0)), (
            "second prompt's prefill_cos must be the FRESH table (2.0), not the "
            "first prompt's stale table (1.0) -- reset() must reset "
            "PreRoPECapture's own cos/sin latch via begin_forward()"
        )
    finally:
        cache.detach()


def _owns_storage(t: torch.Tensor) -> bool:
    return t.untyped_storage().size() == t.numel() * t.element_size()


def test_layer_entries_own_storage_when_no_residual_fp32() -> None:
    cache = _build_compressed_cache()
    for i in range(N_LAYERS):
        cache._residual_k[i] = None
        cache._residual_v[i] = None
    _decode_step(cache, step_seed=1)  # fp32 == recon dtype -> no-op .to()
    assert cache._recon_cache_k == {} and cache._recon_cache_v == {}
    for k_ready, v_ready in cache._recon_cache_layer.values():
        assert _owns_storage(v_ready), (
            "cached V entry aliases the released fp32 group cache; its "
            "storage retains the whole group tensor"
        )
        assert _owns_storage(k_ready)


def test_pre_rope_analysis_path_survives_group_release() -> None:
    cache = _build_compressed_cache()
    _decode_step(cache, step_seed=1)
    assert cache._recon_cache_k == {}
    k_pre, v_recon = cache._reconstruct_layer(0)
    assert k_pre.shape == (1, N_KV, T_PREFILL, D_H)
    assert k_pre.dtype == torch.float32
    assert v_recon.dtype == torch.float32


def main() -> int:
    test_decode_outputs_match_reference()
    print("OK test_decode_outputs_match_reference")
    test_cache_hit_path_bitwise_stable_across_steps()
    print("OK test_cache_hit_path_bitwise_stable_across_steps")
    test_group_caches_released_after_full_pass()
    print("OK test_group_caches_released_after_full_pass")
    test_decode_ready_cache_uses_attention_dtype()
    print("OK test_decode_ready_cache_uses_attention_dtype")
    test_long_context_gate_skips_layer_cache()
    print("OK test_long_context_gate_skips_layer_cache")
    test_group_caches_released_again_after_dtype_change_rebuild()
    print("OK test_group_caches_released_again_after_dtype_change_rebuild")
    test_layer_entries_own_storage_when_no_residual_fp32()
    print("OK test_layer_entries_own_storage_when_no_residual_fp32")
    test_pre_rope_analysis_path_survives_group_release()
    print("OK test_pre_rope_analysis_path_survives_group_release")
    test_reset_captures_fresh_rope_table_not_stale_one()
    print("OK test_reset_captures_fresh_rope_table_not_stale_one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
