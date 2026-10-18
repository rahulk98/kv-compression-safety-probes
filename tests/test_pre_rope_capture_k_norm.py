"""Tests for the k_norm-aware pre-RoPE K capture in `PreRoPECapture`
(kv_compress/integration/pre_rope_capture.py).

Background: OLMoE (and QK-norm models generally) apply an RMSNorm (`k_norm`)
to the FULL k_proj output BEFORE reshape into heads and BEFORE RoPE
(`key_states = self.k_norm(self.k_proj(hidden_states))` in
transformers/models/olmoe/modeling_olmoe.py). A forward hook on `k_proj`
alone therefore used to capture the RAW pre-norm projection, which is not
the key the model's cache actually holds. `PreRoPECapture` now hooks the
attention layer's `k_norm` submodule instead (when present), so the
captured tensor is post-k_norm, pre-reshape, pre-RoPE -- the tensor the
deployed cache actually holds before RoPE is re-applied.

No GPU, no model download: everything here is a tiny synthetic attention
module (nn.Linear/nn.Parameter only), mirroring the pattern already used by
tests/test_flashjolt_attention.py's `_FakeAttention` to drive
`PreRoPECapture`'s real hooks and its `apply_rotary_pos_emb` module-level
resolution trick. This is a mechanical hook/shape test, not a compression-
quality test, so it does not fall under the project's "real caches only"
rule (which governs KV-COMPRESSION fidelity testing, not capture-plumbing
unit tests -- tests/test_moe_kv_shape_parity.py already sets this precedent
for OLMoE-shape testing with synthetic configs).

Run: .venv/bin/python -m pytest tests/test_pre_rope_capture_k_norm.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kv_compress.integration.pre_rope_capture import PreRoPECapture

# ------------------------------------------------------------ fake attention
# Module-level `apply_rotary_pos_emb` / `rotate_half` so
# `PreRoPECapture._install_rope_patch`'s `importlib.import_module(type(attn
# ).__module__)` resolution finds them here, exactly as it would find the
# real ones in modeling_olmoe.py (same trick test_flashjolt_attention.py
# uses for modeling_llama.py).


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Matches OLMoE's real signature (no position_ids) and formula."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)


class _FakeKeyNorm(nn.Module):
    """OLMoE-style RMSNorm on the FULL concatenated K width, applied BEFORE
    reshape into heads -- mirrors OlmoeRMSNorm.forward exactly (variance
    over the last dim, then a weight scale). Weight is a non-identity scale
    (1.5, not 1.0) so normalization visibly changes values -- otherwise a
    bug that skipped the norm entirely could coincidentally still pass a
    "does it match k_norm's output" assertion.
    """

    def __init__(self, width: int, eps: float = 1e-5, scale: float = 1.5):
        super().__init__()
        self.weight = nn.Parameter(torch.full((width,), scale))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


class _FakeBadKeyNorm(nn.Module):
    """A key-norm that returns the WRONG width (per-head, not the full
    n_kv*head_dim concatenated width) -- used only to prove the shape guard
    in PreRoPECapture._make_key_norm_hook raises loudly instead of silently
    mis-reshaping."""

    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., : self.head_dim]


class _FakePerHeadKeyNorm(nn.Module):
    """A Qwen3-style PER-HEAD key-norm: it is applied AFTER the reshape into
    heads, so it both receives and returns a 4-D (B, T, n_kv, head_dim)
    tensor. Used to prove the shape guard rejects that rank with the
    documented RuntimeError instead of dying on a 3-tuple unpack."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FakePerHeadKeyNormAttention(nn.Module):
    """Qwen3-shaped attention: `k_norm(k_proj(h).view(B, T, n_kv, d_h))`, i.e.
    the norm sits AFTER the reshape and sees a 4-D tensor. Out of scope for
    PreRoPECapture's k_norm hook; must fail loudly, not silently."""

    def __init__(self, hidden_size: int, n_kv: int, head_dim: int, layer_idx: int):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.k_proj = nn.Linear(hidden_size, n_kv * head_dim, bias=False)
        self.k_norm = _FakePerHeadKeyNorm()

    def forward(self, hidden_states, **kwargs):
        bsz, q_len, _ = hidden_states.shape
        n_kv = self.k_proj.out_features // self.head_dim
        return self.k_norm(
            self.k_proj(hidden_states).view(bsz, q_len, n_kv, self.head_dim)
        )


class _FakeKeyNormAttention(nn.Module):
    """OLMoE-shaped attention: `key_states = k_norm(k_proj(hidden_states))`,
    optional `clip_qkv` clamp AFTER k_norm/q_proj/v_proj but before reshape,
    then reshape + RoPE. `q_norm_scale` is a plain mutable attribute (not a
    parameter) standing in for a q_norm transform, applied to Q ONLY --
    used to confirm the captured K is independent of it.
    """

    def __init__(
        self,
        hidden_size: int,
        n_q: int,
        n_kv: int,
        head_dim: int,
        layer_idx: int,
        clip_qkv: float | None = None,
        bad_key_norm: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_key_value_groups = n_q // n_kv
        self.scaling = head_dim ** -0.5
        self.layer_idx = layer_idx
        self.clip_qkv = clip_qkv
        self.q_norm_scale = 1.0
        self.q_proj = nn.Linear(hidden_size, n_q * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, n_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, n_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(n_q * head_dim, hidden_size, bias=False)
        self.k_norm: nn.Module = (
            _FakeBadKeyNorm(head_dim) if bad_key_norm else _FakeKeyNorm(n_kv * head_dim)
        )

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
        bsz, q_len, _ = hidden_states.shape
        n_q = self.q_proj.out_features // self.head_dim
        n_kv = self.k_proj.out_features // self.head_dim

        # OLMoE ordering: k_norm on the FULL k_proj output, before reshape.
        query_flat = self.q_proj(hidden_states) * self.q_norm_scale
        key_flat = self.k_norm(self.k_proj(hidden_states))
        value_flat = self.v_proj(hidden_states)
        if self.clip_qkv is not None:
            query_flat = query_flat.clamp(min=-self.clip_qkv, max=self.clip_qkv)
            key_flat = key_flat.clamp(min=-self.clip_qkv, max=self.clip_qkv)
            value_flat = value_flat.clamp(min=-self.clip_qkv, max=self.clip_qkv)

        q = query_flat.view(bsz, q_len, n_q, self.head_dim).transpose(1, 2)
        k = key_flat.view(bsz, q_len, n_kv, self.head_dim).transpose(1, 2)
        v = value_flat.view(bsz, q_len, n_kv, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        n_rep = self.num_key_value_groups
        k_rep, v_rep = _repeat_kv(k, n_rep), _repeat_kv(v, n_rep)
        weights = torch.matmul(q, k_rep.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            weights = weights + attention_mask
        weights = torch.softmax(weights, dim=-1)
        out = torch.matmul(weights, v_rep).transpose(1, 2).contiguous()
        out = out.reshape(bsz, q_len, -1).contiguous()
        return self.o_proj(out), weights


class _FakeDenseAttention(nn.Module):
    """No k_norm at all -- the Mistral/LLaMA-2 shape. Identical to
    test_flashjolt_attention.py's `_FakeAttention`: the regression target
    for "dense models are byte-identical to before this change"."""

    def __init__(self, hidden_size: int, n_q: int, n_kv: int, head_dim: int, layer_idx: int):
        super().__init__()
        self.head_dim = head_dim
        self.num_key_value_groups = n_q // n_kv
        self.scaling = head_dim ** -0.5
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(hidden_size, n_q * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, n_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, n_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(n_q * head_dim, hidden_size, bias=False)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
        bsz, q_len, _ = hidden_states.shape
        n_q = self.q_proj.out_features // self.head_dim
        n_kv = self.k_proj.out_features // self.head_dim
        q = self.q_proj(hidden_states).view(bsz, q_len, n_q, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, n_kv, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        n_rep = self.num_key_value_groups
        k_rep, v_rep = _repeat_kv(k, n_rep), _repeat_kv(v, n_rep)
        weights = torch.matmul(q, k_rep.transpose(2, 3)) * self.scaling
        weights = torch.softmax(weights, dim=-1)
        out = torch.matmul(weights, v_rep).transpose(1, 2).contiguous()
        out = out.reshape(bsz, q_len, -1).contiguous()
        return self.o_proj(out), weights


class _FakeLayer:
    def __init__(self, self_attn):
        self.self_attn = self_attn


class _FakeInnerModel:
    def __init__(self, layers):
        self.layers = layers


class _FakeCausalLM:
    def __init__(self, layers):
        self.model = _FakeInnerModel(layers)


def _fake_key_norm_model(
    n_layers, hidden_size, n_q, n_kv, head_dim, clip_qkv=None, bad_key_norm=False
):
    attn_modules = [
        _FakeKeyNormAttention(hidden_size, n_q, n_kv, head_dim, i, clip_qkv=clip_qkv, bad_key_norm=bad_key_norm)
        for i in range(n_layers)
    ]
    model = _FakeCausalLM([_FakeLayer(a) for a in attn_modules])
    model.config = SimpleNamespace(
        num_key_value_heads=n_kv, head_dim=head_dim, clip_qkv=clip_qkv
    )
    return model, attn_modules


def _fake_dense_model(n_layers, hidden_size, n_q, n_kv, head_dim):
    attn_modules = [_FakeDenseAttention(hidden_size, n_q, n_kv, head_dim, i) for i in range(n_layers)]
    model = _FakeCausalLM([_FakeLayer(a) for a in attn_modules])
    model.config = SimpleNamespace(num_key_value_heads=n_kv, head_dim=head_dim)
    return model, attn_modules


def _rotary_table(seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


# --------------------------------------------------------------------- tests


def test_has_key_norm_flag_detects_k_norm_submodule():
    model_kn, _ = _fake_key_norm_model(1, hidden_size=8, n_q=2, n_kv=2, head_dim=4)
    model_dense, _ = _fake_dense_model(1, hidden_size=8, n_q=2, n_kv=2, head_dim=4)
    assert PreRoPECapture(model_kn).has_key_norm is True
    assert PreRoPECapture(model_dense).has_key_norm is False


def test_key_norm_capture_is_post_norm_pre_rope():
    """Core faithfulness test: captured pre-RoPE K must equal the model's
    OWN post-k_norm output reshaped (norm WAS applied), and must differ from
    both the raw pre-norm k_proj reshape and the post-RoPE key (RoPE was NOT
    applied at capture time)."""
    n_layers, n_kv, n_q, head_dim, T, hidden_size = 2, 4, 4, 8, 5, 16
    model, attn_modules = _fake_key_norm_model(n_layers, hidden_size, n_q, n_kv, head_dim)
    capture = PreRoPECapture(model)
    assert capture.has_key_norm is True
    capture.attach()
    try:
        hidden_states = torch.randn(1, T, hidden_size)
        cos, sin = _rotary_table(T, head_dim)
        with torch.no_grad():
            for layer_idx in range(n_layers):
                attn_modules[layer_idx](
                    hidden_states, position_embeddings=(cos, sin),
                    attention_mask=None, past_key_values=None,
                )

        for layer_idx in range(n_layers):
            captured = capture.pop_pre_rope_k(layer_idx)
            attn = attn_modules[layer_idx]
            assert captured.shape == (1, n_kv, T, head_dim)

            with torch.no_grad():
                post_norm_ref = attn.k_norm(attn.k_proj(hidden_states))
                post_norm_ref = post_norm_ref.view(1, T, n_kv, head_dim).transpose(1, 2)
                raw_ref = attn.k_proj(hidden_states).view(1, T, n_kv, head_dim).transpose(1, 2)
                q_ref = (attn.q_proj(hidden_states) * attn.q_norm_scale).view(
                    1, T, n_q, head_dim
                ).transpose(1, 2)
                _, k_post_rope_ref = apply_rotary_pos_emb(q_ref, post_norm_ref, cos, sin)

            assert torch.equal(captured, post_norm_ref), (
                f"layer {layer_idx}: captured pre-RoPE K must equal the "
                "post-k_norm reshape exactly"
            )
            assert not torch.equal(captured, raw_ref), (
                f"layer {layer_idx}: captured K must NOT equal the raw "
                "pre-norm k_proj reshape (k_norm must actually be applied)"
            )
            assert not torch.allclose(captured, k_post_rope_ref), (
                f"layer {layer_idx}: captured K must be PRE-RoPE, not "
                "post-RoPE"
            )
    finally:
        capture.detach()


def test_no_key_norm_capture_matches_k_proj_reshape_byte_identical_regression_guard():
    """Regression guard: for a model with NO k_norm (the thesis's dense
    models, Mistral-7B-v0.3 / LLaMA-2-13B), PreRoPECapture must still hook
    k_proj directly and produce output BYTE-IDENTICAL to before this change
    (torch.equal, not allclose)."""
    n_layers, n_kv, n_q, head_dim, T, hidden_size = 2, 2, 2, 8, 6, 16
    model, attn_modules = _fake_dense_model(n_layers, hidden_size, n_q, n_kv, head_dim)
    capture = PreRoPECapture(model)
    assert capture.has_key_norm is False
    capture.attach()
    try:
        hidden_states = torch.randn(1, T, hidden_size)
        cos, sin = _rotary_table(T, head_dim)
        with torch.no_grad():
            for layer_idx in range(n_layers):
                attn_modules[layer_idx](
                    hidden_states, position_embeddings=(cos, sin),
                    attention_mask=None, past_key_values=None,
                )
        for layer_idx in range(n_layers):
            captured = capture.pop_pre_rope_k(layer_idx)
            attn = attn_modules[layer_idx]
            with torch.no_grad():
                ref = attn.k_proj(hidden_states).view(1, T, n_kv, head_dim).transpose(1, 2)
            assert torch.equal(captured, ref), (
                f"layer {layer_idx}: dense (no k_norm) capture must be "
                "byte-identical to the plain k_proj hook path"
            )
    finally:
        capture.detach()


def test_clip_qkv_is_mirrored_in_key_norm_capture():
    """OLMoE's optional clip_qkv clamps K AFTER k_norm but before reshape/
    RoPE, in the real forward -- our hook fires at k_norm's return, upstream
    of that clamp, so it must replicate it (see PreRoPECapture.
    _make_key_norm_hook)."""
    n_layers, n_kv, n_q, head_dim, T, hidden_size = 1, 2, 2, 4, 5, 8
    clip = 0.05  # small vs. randn-scale inputs -> the clamp visibly bites
    model, attn_modules = _fake_key_norm_model(
        n_layers, hidden_size, n_q, n_kv, head_dim, clip_qkv=clip
    )
    capture = PreRoPECapture(model)
    capture.attach()
    try:
        hidden_states = torch.randn(1, T, hidden_size)
        cos, sin = _rotary_table(T, head_dim)
        with torch.no_grad():
            attn_modules[0](
                hidden_states, position_embeddings=(cos, sin),
                attention_mask=None, past_key_values=None,
            )
        captured = capture.pop_pre_rope_k(0)
        attn = attn_modules[0]
        with torch.no_grad():
            unclamped = attn.k_norm(attn.k_proj(hidden_states))
            unclamped = unclamped.view(1, T, n_kv, head_dim).transpose(1, 2)
            clamped_ref = unclamped.clamp(min=-clip, max=clip)
        assert torch.equal(captured, clamped_ref), (
            "captured K must mirror the model's clip_qkv clamp"
        )
        assert not torch.equal(captured, unclamped), (
            "the clamp must actually change something, or this test proves nothing"
        )
    finally:
        capture.detach()


def test_q_norm_does_not_affect_captured_key():
    """q_norm (here: q_norm_scale) only feeds the query branch. Varying it
    on the SAME attention module (identical k_proj/k_norm weights) must
    leave the captured K completely unchanged -- confirms by construction,
    not just by reading the source, that q_norm cannot leak into the cached
    key path."""
    n_layers, n_kv, n_q, head_dim, T, hidden_size = 1, 2, 2, 4, 5, 8
    model, attn_modules = _fake_key_norm_model(n_layers, hidden_size, n_q, n_kv, head_dim)
    attn = attn_modules[0]
    capture = PreRoPECapture(model)
    capture.attach()
    try:
        hidden_states = torch.randn(1, T, hidden_size)
        cos, sin = _rotary_table(T, head_dim)

        attn.q_norm_scale = 1.0
        with torch.no_grad():
            attn(hidden_states, position_embeddings=(cos, sin), attention_mask=None, past_key_values=None)
        k_scale_1 = capture.pop_pre_rope_k(0).clone()

        capture.begin_forward()
        attn.q_norm_scale = 7.0
        with torch.no_grad():
            attn(hidden_states, position_embeddings=(cos, sin), attention_mask=None, past_key_values=None)
        k_scale_7 = capture.pop_pre_rope_k(0)

        assert torch.equal(k_scale_1, k_scale_7), (
            "captured K must not depend on q_norm at all"
        )
    finally:
        capture.detach()


def test_key_norm_width_mismatch_raises_runtime_error():
    """A 3-D key-norm output of the wrong width must raise loudly rather than
    silently mis-reshape."""
    n_layers, n_kv, n_q, head_dim, T, hidden_size = 1, 2, 2, 4, 3, 8
    model, attn_modules = _fake_key_norm_model(
        n_layers, hidden_size, n_q, n_kv, head_dim, bad_key_norm=True
    )
    capture = PreRoPECapture(model)
    capture.attach()
    try:
        hidden_states = torch.randn(1, T, hidden_size)
        cos, sin = _rotary_table(T, head_dim)
        with pytest.raises(RuntimeError, match="k_norm output width"):
            attn_modules[0](
                hidden_states, position_embeddings=(cos, sin),
                attention_mask=None, past_key_values=None,
            )
    finally:
        capture.detach()


def test_per_head_key_norm_4d_output_is_captured():
    """Qwen3-style per-head key-norm returns a 4-D (B, T, n_kv, d_h) tensor;
    the capture is that tensor transposed to (B, n_kv, T, d_h)."""
    n_kv, head_dim, T, hidden_size = 2, 4, 3, 8
    attn = _FakePerHeadKeyNormAttention(hidden_size, n_kv, head_dim, 0)
    model = _FakeCausalLM([_FakeLayer(attn)])
    model.config = SimpleNamespace(num_key_value_heads=n_kv, head_dim=head_dim)

    capture = PreRoPECapture(model)
    assert capture.has_key_norm is True
    capture.attach()
    try:
        with torch.no_grad():
            out = attn(torch.randn(1, T, hidden_size))
        assert torch.equal(capture.pop_pre_rope_k(0), out.transpose(1, 2))
    finally:
        capture.detach()


def test_per_head_key_norm_wrong_4d_shape_raises_runtime_error():
    n_kv, head_dim, T, hidden_size = 2, 4, 3, 8
    attn = _FakePerHeadKeyNormAttention(hidden_size, n_kv, head_dim, 0)
    model = _FakeCausalLM([_FakeLayer(attn)])
    model.config = SimpleNamespace(num_key_value_heads=n_kv + 1, head_dim=head_dim)

    capture = PreRoPECapture(model)
    capture.attach()
    try:
        with pytest.raises(RuntimeError, match="k_norm output width"):
            attn(torch.randn(1, T, hidden_size))
    finally:
        capture.detach()


def _tiny_qwen3(**overrides):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = dict(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=512,
    )
    cfg.update(overrides)
    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config(**cfg)).eval()


def test_qwen3_capture_rerope_matches_model_keys():
    """Captured post-k_norm pre-RoPE K, re-roped with the captured cos/sin,
    must equal the keys the model stores in its own cache (exact: same ops)."""
    from transformers import DynamicCache

    from kv_compress.integration.pre_rope_capture import apply_rope_to_k

    model = _tiny_qwen3()
    capture = PreRoPECapture(model)
    assert capture.has_key_norm is True
    capture.attach()
    try:
        ids = torch.randint(0, 128, (1, 37))
        cache = DynamicCache(config=model.config)
        with torch.no_grad():
            model(ids, past_key_values=cache, use_cache=True)
        cos, sin = capture.get_cos_sin()
        for i, layer in enumerate(cache.layers):
            k_pre = capture.pop_pre_rope_k(i)
            assert k_pre.shape == layer.keys.shape
            assert torch.equal(apply_rope_to_k(k_pre, cos, sin), layer.keys)
    finally:
        capture.detach()


def test_detach_leaves_no_hooks_on_the_model():
    """Cleanup contract: after detach() no forward hook of ours survives on
    any k_norm (or k_proj) submodule, and detach() is safe to call twice."""
    n_layers, n_kv, n_q, head_dim, hidden_size = 3, 2, 2, 4, 8
    kn_model, kn_attn = _fake_key_norm_model(n_layers, hidden_size, n_q, n_kv, head_dim)
    dense_model, dense_attn = _fake_dense_model(n_layers, hidden_size, n_q, n_kv, head_dim)

    for model, attn_modules, hooked in (
        (kn_model, kn_attn, "k_norm"),
        (dense_model, dense_attn, "k_proj"),
    ):
        capture = PreRoPECapture(model)
        capture.attach()
        assert all(
            len(getattr(a, hooked)._forward_hooks) == 1 for a in attn_modules
        ), f"{hooked} hook should be installed on every layer after attach()"
        capture.detach()
        capture.detach()  # idempotent
        for a in attn_modules:
            assert len(a.k_proj._forward_hooks) == 0
            if hooked == "k_norm":
                assert len(a.k_norm._forward_hooks) == 0


def main() -> int:
    tests = [
        test_has_key_norm_flag_detects_k_norm_submodule,
        test_key_norm_capture_is_post_norm_pre_rope,
        test_no_key_norm_capture_matches_k_proj_reshape_byte_identical_regression_guard,
        test_clip_qkv_is_mirrored_in_key_norm_capture,
        test_q_norm_does_not_affect_captured_key,
        test_key_norm_width_mismatch_raises_runtime_error,
        test_per_head_key_norm_4d_output_is_captured,
        test_per_head_key_norm_wrong_4d_shape_raises_runtime_error,
        test_qwen3_capture_rerope_matches_model_keys,
        test_detach_leaves_no_hooks_on_the_model,
    ]
    failed = 0
    for t in tests:
        print(f"\n[{t.__name__}]")
        try:
            t()
            print("  OK")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {e}")
    print(f"\n{'=' * 60}\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
