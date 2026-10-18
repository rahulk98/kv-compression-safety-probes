"""Triton and CPU implementations for FlashJoLT reconstruction and decode attention.

K is reconstructed from Tucker and residual terms, then RoPE is applied;
V is never rotated. Triton paths accumulate in fp32 with IEEE dot products,
tile rank and feature axes to fit shared memory, and decode residuals inline.
Triton remains an optional dependency so CPU fallback imports work without it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from . import launch_config
from . import reference as _reference
from .layout import (
    AttentionInputs,
    FactorizedKVGroup,
    GroupLayout,
    HeadMap,
    ResidualPayload,
    TuckerGroupFactors,
)
from .triton_unpack import TRITON_AVAILABLE, select_backend

logger = logging.getLogger(__name__)

if TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    from .triton_unpack import _decode_lane

    @triton.jit
    def _reconstruct_kernel(
        token_factor_ptr, core_ptr, feature_factor_ptr,
        packed_ptr, scale_ptr, rotation_ptr,
        out_ptr,
        T, token_rank, feature_rank, head_dim, n_kv, slot,
        bits: tl.constexpr, vpb: tl.constexpr, HAS_RESIDUAL: tl.constexpr,
        BLOCK_T: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_F: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Reconstruct one KV-head token block in fp32 registers."""
        pid_h = tl.program_id(0)
        pid_t = tl.program_id(1)

        t0 = pid_t * BLOCK_T
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T

        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < head_dim
        core_head_base = (slot * n_kv + pid_h) * token_rank * feature_rank

        # Chunk rank axes to bound shared-memory use; accumulate in fp32.
        low = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
        for f0 in range(0, feature_rank, BLOCK_F):
            f_offsets = f0 + tl.arange(0, BLOCK_F)
            f_mask = f_offsets < feature_rank
            # Stage 1: mid = U_T_blk @ core[slot, kv_head]  -- (BLOCK_T, BLOCK_F)
            mid = tl.zeros((BLOCK_T, BLOCK_F), dtype=tl.float32)
            for r0 in range(0, token_rank, BLOCK_R):
                r_offsets = r0 + tl.arange(0, BLOCK_R)
                r_mask = r_offsets < token_rank
                u_blk = tl.load(
                    token_factor_ptr + t_offsets[:, None] * token_rank + r_offsets[None, :],
                    mask=t_mask[:, None] & r_mask[None, :], other=0.0,
                ).to(tl.float32)
                core_blk = tl.load(
                    core_ptr + core_head_base + r_offsets[:, None] * feature_rank + f_offsets[None, :],
                    mask=r_mask[:, None] & f_mask[None, :], other=0.0,
                ).to(tl.float32)
                mid += tl.dot(u_blk, core_blk, input_precision="ieee")

            # Stage 2: low += mid @ feature_factor[:, f_chunk].T  -- (BLOCK_T, BLOCK_D)
            ff_blk = tl.load(
                feature_factor_ptr + d_offsets[:, None] * feature_rank + f_offsets[None, :],
                mask=d_mask[:, None] & f_mask[None, :], other=0.0,
            ).to(tl.float32)
            low += tl.dot(mid, tl.trans(ff_blk), input_precision="ieee")

        if HAS_RESIDUAL:
            # Absolute flat index matches reference._reconstruct_block's
            # elem_start = kv_head*T*head_dim + t*head_dim + d layout.
            idx = (pid_h * T + t_offsets[:, None]) * head_dim + d_offsets[None, :]
            lane_mask = t_mask[:, None] & d_mask[None, :]
            codes = _decode_lane(packed_ptr, idx, lane_mask, bits, vpb).to(tl.float32)
            scale_blk = tl.load(scale_ptr + pid_h * T + t_offsets, mask=t_mask, other=0.0).to(tl.float32)
            rotated = codes * scale_blk[:, None]
            rot_blk = tl.load(
                rotation_ptr + d_offsets[:, None] * head_dim + d_offsets[None, :],
                mask=d_mask[:, None] & d_mask[None, :], other=0.0,
            ).to(tl.float32)
            res = tl.dot(rotated, rot_blk, input_precision="ieee")
            low = low + res

        out_offsets = (pid_h * T + t_offsets[:, None]) * head_dim + d_offsets[None, :]
        tl.store(out_ptr + out_offsets, low, mask=t_mask[:, None] & d_mask[None, :])


_BLOCK_T = 64  # fixed token-block size; T > 64 exercises multiple program launches along axis 1
# Cap rank-axis tiles to bound Triton shared-memory use.
_BLOCK_RANK = 32


def _rank_block(rank: int) -> int:
    """Power-of-two `tl.dot`-compatible block size for a rank axis: at least
    16 (tl.dot's lane floor), at most `_BLOCK_RANK` (shared-memory cap --
    larger ranks loop over chunks instead of widening the tile)."""
    return min(_BLOCK_RANK, max(16, triton.next_power_of_2(rank)))


def reconstruct_dense_triton(
    tucker: TuckerGroupFactors,
    residual: ResidualPayload | None,
    slot: int,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Triton implementation of `reference.reconstruct_dense`'s exact signature and semantics."""
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "reconstruct_dense_triton() called but `triton` is not importable in "
            "this environment -- use reconstruct_dense() for automatic CPU "
            "fallback, or run this on a host with a CUDA Triton install."
        )
    if not torch.cuda.is_available() or not tucker.core.is_cuda:
        raise RuntimeError(
            "reconstruct_dense_triton() requires tucker tensors on a CUDA device "
            f"with CUDA available; got tucker.core.device={tucker.core.device}, "
            f"torch.cuda.is_available()={torch.cuda.is_available()}. Use "
            "reconstruct_dense() for automatic CPU fallback."
        )

    n_kv = tucker.num_kv_heads
    T = tucker.seq_len
    head_dim = tucker.head_dim
    token_rank = tucker.token_rank
    feature_rank = tucker.feature_rank
    device = tucker.core.device

    core = tucker.core.contiguous()
    token_factor = tucker.token_factor.contiguous()
    feature_factor = tucker.feature_factor.contiguous()

    has_residual = residual is not None and residual.bits > 0
    if has_residual:
        packed = residual.packed.contiguous()
        scale = residual.scale.contiguous()
        rotation = residual.rotation.contiguous()
        bits = residual.bits
        vpb = 1 if bits == 8 else 8 // bits
    else:
        # bits=0 uses dummy pointers; HAS_RESIDUAL prevents dereferences.
        packed = torch.empty(1, dtype=torch.uint8, device=device)
        scale = torch.empty(1, dtype=torch.float32, device=device)
        rotation = torch.empty(1, dtype=torch.float32, device=device)
        bits, vpb = 8, 1

    out = torch.empty(n_kv, T, head_dim, dtype=torch.float32, device=device)

    BLOCK_R = _rank_block(token_rank)
    BLOCK_F = _rank_block(feature_rank)
    BLOCK_D = max(16, triton.next_power_of_2(head_dim))
    grid = (n_kv, triton.cdiv(T, _BLOCK_T))
    _reconstruct_kernel[grid](
        token_factor, core, feature_factor,
        packed, scale, rotation,
        out,
        T, token_rank, feature_rank, head_dim, n_kv, slot,
        bits=bits, vpb=vpb, HAS_RESIDUAL=has_residual,
        BLOCK_T=_BLOCK_T, BLOCK_R=BLOCK_R, BLOCK_F=BLOCK_F, BLOCK_D=BLOCK_D,
    )
    return out if out_dtype == torch.float32 else out.to(out_dtype)


reconstruct_dense_cpu = _reference.reconstruct_dense


# ------------------------------------------------------------------ dispatch


def reconstruct_dense(
    tucker: TuckerGroupFactors,
    residual: ResidualPayload | None,
    slot: int,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Drop-in replacement for `reference.reconstruct_dense` with the identical signature and semantics: dispatches to the Triton kernel when a supported CUDA device is available, else automatically falls back to `reference.reconstruct_dense`."""
    backend = select_backend(tucker.core.device)
    logger.info(
        "reconstruct_dense dispatch: backend=%s device=%s slot=%s",
        backend, tucker.core.device, slot,
    )
    if backend == "triton":
        return reconstruct_dense_triton(tucker, residual, slot, out_dtype=out_dtype)
    dense = reconstruct_dense_cpu(tucker, residual, slot)
    return dense if out_dtype == torch.float32 else dense.to(out_dtype)


# RoPE for reconstructed K; V is never rotated. Tables come from the model.

if TRITON_AVAILABLE:

    @triton.jit
    def _rope_kernel(
        k_ptr, cos_ptr, sin_ptr, out_ptr,
        T, head_dim, half,
        BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Apply RoPE to one K tile, using global gathers for shifted dimensions."""
        pid_h = tl.program_id(0)
        pid_t = tl.program_id(1)
        t0 = pid_t * BLOCK_T
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < head_dim
        mask2d = t_mask[:, None] & d_mask[None, :]

        d_shifted = tl.where(d_offsets < half, d_offsets + half, d_offsets - half)
        sign = tl.where(d_offsets < half, -1.0, 1.0).to(tl.float32)

        base = pid_h * T * head_dim
        k = tl.load(
            k_ptr + base + t_offsets[:, None] * head_dim + d_offsets[None, :],
            mask=mask2d, other=0.0,
        ).to(tl.float32)
        k_shift = tl.load(
            k_ptr + base + t_offsets[:, None] * head_dim + d_shifted[None, :],
            mask=mask2d, other=0.0,
        ).to(tl.float32)
        cos = tl.load(
            cos_ptr + t_offsets[:, None] * head_dim + d_offsets[None, :],
            mask=mask2d, other=0.0,
        ).to(tl.float32)
        sin = tl.load(
            sin_ptr + t_offsets[:, None] * head_dim + d_offsets[None, :],
            mask=mask2d, other=0.0,
        ).to(tl.float32)

        out = k * cos + sign[None, :] * k_shift * sin
        tl.store(
            out_ptr + base + t_offsets[:, None] * head_dim + d_offsets[None, :],
            out, mask=mask2d,
        )


def _apply_rope_triton(k_pre: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE (Triton, in-kernel) to a dense `(n_kv, T, head_dim)`
    pre-RoPE K tensor. Internal helper for `reconstruct_dense_k_post_rope_triton`
    only -- always fp32 in, fp32 out (the caller casts to the attention
    compute dtype AFTER this returns, if requested; see that function).
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "_apply_rope_triton() called but `triton` is not importable in this environment."
        )
    if not torch.cuda.is_available() or not k_pre.is_cuda:
        raise RuntimeError(
            "_apply_rope_triton() requires k_pre on a CUDA device with CUDA available; "
            f"got k_pre.device={k_pre.device}, torch.cuda.is_available()={torch.cuda.is_available()}."
        )
    n_kv, T, head_dim = k_pre.shape
    if head_dim % 2 != 0:
        # RoPE requires an even head dimension.
        raise ValueError(f"_apply_rope_triton: head_dim must be even for RoPE, got {head_dim}")
    k_pre = k_pre.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    out = torch.empty_like(k_pre)
    BLOCK_D = max(16, triton.next_power_of_2(head_dim))
    grid = (n_kv, triton.cdiv(T, _BLOCK_T))
    _rope_kernel[grid](
        k_pre, cos, sin, out,
        T, head_dim, head_dim // 2,
        BLOCK_T=_BLOCK_T, BLOCK_D=BLOCK_D,
    )
    return out


def reconstruct_dense_k_post_rope_triton(
    tucker_k: TuckerGroupFactors,
    residual_k: ResidualPayload | None,
    slot: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct pre-RoPE K in fp32, apply RoPE, then cast to `out_dtype`."""
    k_pre = reconstruct_dense_triton(tucker_k, residual_k, slot)
    k_post = _apply_rope_triton(k_pre, cos, sin)
    return k_post if out_dtype == torch.float32 else k_post.to(out_dtype)


reconstruct_dense_k_post_rope_cpu = _reference.reconstruct_dense_k_post_rope


def reconstruct_dense_k_post_rope(
    tucker_k: TuckerGroupFactors,
    residual_k: ResidualPayload | None,
    slot: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Drop-in replacement for `reference.reconstruct_dense_k_post_rope`,
    dispatching triton/cpu exactly like `reconstruct_dense` above.
    """
    backend = select_backend(tucker_k.core.device)
    logger.info(
        "reconstruct_dense_k_post_rope dispatch: backend=%s device=%s slot=%s",
        backend, tucker_k.core.device, slot,
    )
    if backend == "triton":
        return reconstruct_dense_k_post_rope_triton(
            tucker_k, residual_k, slot, cos, sin, out_dtype=out_dtype
        )
    k_post = reconstruct_dense_k_post_rope_cpu(tucker_k, residual_k, slot, cos, sin)
    return k_post if out_dtype == torch.float32 else k_post.to(out_dtype)


# Dense post-RoPE tail passed to fused attention.


def reconstruct_dense_k_post_rope_with_tail(
    tucker_k: TuckerGroupFactors,
    residual_k: ResidualPayload | None,
    slot: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    tail_k: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compressed-prefill K (reconstructed + RoPE'd via
    `reconstruct_dense_k_post_rope`) concatenated with the dense decode tail
    K. `tail_k` is ALREADY POST-RoPE and is passed through completely
    untouched -- no RoPE call anywhere near it -- since rotating it again
    here would double-rotate positions the model already rotated once during
    real generation.
    """
    prefill = reconstruct_dense_k_post_rope(tucker_k, residual_k, slot, cos, sin, out_dtype=out_dtype)
    if tail_k is None or tail_k.shape[1] == 0:
        return prefill
    return torch.cat([prefill, tail_k.to(out_dtype)], dim=1)


def reconstruct_dense_v_with_tail(
    tucker_v: TuckerGroupFactors,
    residual_v: ResidualPayload | None,
    slot: int,
    tail_v: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compressed-prefill V (reconstructed via `reconstruct_dense`, no RoPE
    ever, for either the prefill or the tail) concatenated with the dense
    decode tail V. Same None/zero-length tail-equivalence as
    `reconstruct_dense_k_post_rope_with_tail`.
    """
    prefill = reconstruct_dense(tucker_v, residual_v, slot, out_dtype=out_dtype)
    if tail_v is None or tail_v.shape[1] == 0:
        return prefill
    return torch.cat([prefill, tail_v.to(out_dtype)], dim=1)


# Fused decode streams prefill and tail blocks through one online-softmax pass.
# Split partial states are merged by a second kernel.
# Query rows use the consecutive GQA mapping supplied by the caller.
# RoPE is applied to streamed K tiles in registers; V low-rank terms are
# accumulated in rank space and expanded once during reduction.


if TRITON_AVAILABLE:

    @triton.jit
    def _fused_decode_kernel(
        q_ptr,
        token_factor_k_ptr, core_k_ptr, feature_factor_k_ptr,
        packed_k_ptr, scale_k_ptr, rotation_k_ptr,
        token_factor_v_ptr,
        packed_v_ptr, scale_v_ptr, rotation_v_ptr,
        cos_ptr, sin_ptr,
        tail_k_ptr, tail_v_ptr,
        m_partial_ptr, l_partial_ptr, acc_partial_ptr, s_partial_ptr,
        T, T_tail,
        token_rank_k, feature_rank_k, token_rank_v,
        head_dim, n_kv, n_rep, slot, query_position,
        sm_scale,
        n_splits_alloc, blocks_per_split, n_blocks_prefill, n_blocks_total,
        bits_k: tl.constexpr, vpb_k: tl.constexpr, HAS_RESIDUAL_K: tl.constexpr,
        bits_v: tl.constexpr, vpb_v: tl.constexpr, HAS_RESIDUAL_V: tl.constexpr,
        CAUSAL: tl.constexpr,
        BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr,
        BLOCK_RK: tl.constexpr, BLOCK_FK: tl.constexpr,
        BLOCK_RV_FULL: tl.constexpr,
        BLOCK_D: tl.constexpr, BLOCK_DI: tl.constexpr,
    ):
        """Process one KV-head split and store its partial softmax state."""
        pid_h = tl.program_id(0)  # kv_head index
        pid_s = tl.program_id(1)  # split index along the virtual block axis

        # Empty trailing splits retain the neutral softmax state.
        b_lo = pid_s * blocks_per_split
        b_hi = tl.minimum(b_lo + blocks_per_split, n_blocks_total)

        half = head_dim // 2
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < head_dim
        d_shifted = tl.where(d_offsets < half, d_offsets + half, d_offsets - half)
        sign = tl.where(d_offsets < half, -1.0, 1.0).to(tl.float32)

        h_offsets = tl.arange(0, BLOCK_H)
        h_mask = h_offsets < n_rep
        q_rows = pid_h * n_rep + h_offsets

        q = tl.load(
            q_ptr + q_rows[:, None] * head_dim + d_offsets[None, :],
            mask=h_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)  # (BLOCK_H, BLOCK_D)

        # Online-softmax state; s holds the absorbed low-rank V contribution.
        m = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
        l = tl.zeros((BLOCK_H,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)
        rv_offsets = tl.arange(0, BLOCK_RV_FULL)
        rv_mask = rv_offsets < token_rank_v
        s = tl.zeros((BLOCK_H, BLOCK_RV_FULL), dtype=tl.float32)

        core_k_base = (slot * n_kv + pid_h) * token_rank_k * feature_rank_k

        # Compressed prefill blocks share this split's online-softmax state.
        p_hi = tl.minimum(b_hi, n_blocks_prefill)
        for b in range(b_lo, p_hi):
            t0 = b * BLOCK_T
            t_offsets = t0 + tl.arange(0, BLOCK_T)
            t_mask = t_offsets < T

            low_k = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
            low_k_shift = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
            for f0 in range(0, feature_rank_k, BLOCK_FK):
                f_offsets_k = f0 + tl.arange(0, BLOCK_FK)
                f_mask_k = f_offsets_k < feature_rank_k
                mid_k = tl.zeros((BLOCK_T, BLOCK_FK), dtype=tl.float32)
                for r0 in range(0, token_rank_k, BLOCK_RK):
                    r_offsets_k = r0 + tl.arange(0, BLOCK_RK)
                    r_mask_k = r_offsets_k < token_rank_k
                    u_k_blk = tl.load(
                        token_factor_k_ptr + t_offsets[:, None] * token_rank_k + r_offsets_k[None, :],
                        mask=t_mask[:, None] & r_mask_k[None, :], other=0.0,
                    ).to(tl.float32)
                    core_k_blk = tl.load(
                        core_k_ptr + core_k_base + r_offsets_k[:, None] * feature_rank_k + f_offsets_k[None, :],
                        mask=r_mask_k[:, None] & f_mask_k[None, :], other=0.0,
                    ).to(tl.float32)
                    mid_k += tl.dot(u_k_blk, core_k_blk, input_precision="ieee")
                ff_k_blk = tl.load(
                    feature_factor_k_ptr + d_offsets[:, None] * feature_rank_k + f_offsets_k[None, :],
                    mask=d_mask[:, None] & f_mask_k[None, :], other=0.0,
                ).to(tl.float32)
                ff_k_shift_blk = tl.load(
                    feature_factor_k_ptr + d_shifted[:, None] * feature_rank_k + f_offsets_k[None, :],
                    mask=d_mask[:, None] & f_mask_k[None, :], other=0.0,
                ).to(tl.float32)
                low_k += tl.dot(mid_k, tl.trans(ff_k_blk), input_precision="ieee")
                low_k_shift += tl.dot(mid_k, tl.trans(ff_k_shift_blk), input_precision="ieee")

            if HAS_RESIDUAL_K:
                scale_k_blk = tl.load(scale_k_ptr + pid_h * T + t_offsets, mask=t_mask, other=0.0).to(tl.float32)
                res_k = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
                res_k_shift = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
                # Tile the rotation contraction axis to bound shared memory.
                for di0 in range(0, head_dim, BLOCK_DI):
                    di_offsets = di0 + tl.arange(0, BLOCK_DI)
                    di_mask = di_offsets < head_dim
                    idx_k = (pid_h * T + t_offsets[:, None]) * head_dim + di_offsets[None, :]
                    lane_mask_k = t_mask[:, None] & di_mask[None, :]
                    codes_k = _decode_lane(packed_k_ptr, idx_k, lane_mask_k, bits_k, vpb_k).to(tl.float32)
                    rotated_k = codes_k * scale_k_blk[:, None]  # (BLOCK_T, BLOCK_DI)
                    rot_k_chunk = tl.load(
                        rotation_k_ptr + di_offsets[:, None] * head_dim + d_offsets[None, :],
                        mask=di_mask[:, None] & d_mask[None, :], other=0.0,
                    ).to(tl.float32)
                    rot_k_shift_chunk = tl.load(
                        rotation_k_ptr + di_offsets[:, None] * head_dim + d_shifted[None, :],
                        mask=di_mask[:, None] & d_mask[None, :], other=0.0,
                    ).to(tl.float32)
                    res_k += tl.dot(rotated_k, rot_k_chunk, input_precision="ieee")
                    res_k_shift += tl.dot(rotated_k, rot_k_shift_chunk, input_precision="ieee")
                low_k = low_k + res_k
                low_k_shift = low_k_shift + res_k_shift

            cos_blk = tl.load(
                cos_ptr + t_offsets[:, None] * head_dim + d_offsets[None, :],
                mask=t_mask[:, None] & d_mask[None, :], other=0.0,
            ).to(tl.float32)
            sin_blk = tl.load(
                sin_ptr + t_offsets[:, None] * head_dim + d_offsets[None, :],
                mask=t_mask[:, None] & d_mask[None, :], other=0.0,
            ).to(tl.float32)
            # K is stored pre-RoPE and rotated exactly once in registers.
            k_blk = low_k * cos_blk + sign[None, :] * low_k_shift * sin_blk

            # Accumulate low-rank V in rank space; expand it during reduction.
            u_v_blk = tl.load(
                token_factor_v_ptr + t_offsets[:, None] * token_rank_v + rv_offsets[None, :],
                mask=t_mask[:, None] & rv_mask[None, :], other=0.0,
            ).to(tl.float32)  # (BLOCK_T, BLOCK_RV_FULL)
            if HAS_RESIDUAL_V:
                scale_v_blk = tl.load(scale_v_ptr + pid_h * T + t_offsets, mask=t_mask, other=0.0).to(tl.float32)
                res_v = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
                for di0 in range(0, head_dim, BLOCK_DI):
                    di_offsets = di0 + tl.arange(0, BLOCK_DI)
                    di_mask = di_offsets < head_dim
                    idx_v = (pid_h * T + t_offsets[:, None]) * head_dim + di_offsets[None, :]
                    lane_mask_v = t_mask[:, None] & di_mask[None, :]
                    codes_v = _decode_lane(packed_v_ptr, idx_v, lane_mask_v, bits_v, vpb_v).to(tl.float32)
                    rotated_v = codes_v * scale_v_blk[:, None]  # (BLOCK_T, BLOCK_DI)
                    rot_v_chunk = tl.load(
                        rotation_v_ptr + di_offsets[:, None] * head_dim + d_offsets[None, :],
                        mask=di_mask[:, None] & d_mask[None, :], other=0.0,
                    ).to(tl.float32)
                    res_v += tl.dot(rotated_v, rot_v_chunk, input_precision="ieee")

            scores = tl.dot(q, tl.trans(k_blk), input_precision="ieee") * sm_scale
            valid = h_mask[:, None] & t_mask[None, :]
            if CAUSAL:
                valid = valid & (query_position >= t_offsets)[None, :]
            scores = tl.where(valid, scores, float("-inf"))

            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(m, block_max)
            # Keep the true max; use a finite substitute only in exponentials.
            new_max_safe = tl.where(
                (new_max == float("-inf")) | (new_max == float("inf")), 0.0, new_max
            )
            correction = tl.exp(m - new_max_safe)
            p = tl.exp(scores - new_max_safe[:, None])
            l = l * correction + tl.sum(p, axis=1)
            # Rescale residual/tail and absorbed V state together.
            acc = acc * correction[:, None]
            if HAS_RESIDUAL_V:
                acc += tl.dot(p, res_v, input_precision="ieee")
            s = s * correction[:, None] + tl.dot(p, u_v_blk, input_precision="ieee")
            m = new_max

        # Dense post-RoPE tail blocks use the same online-softmax state.
        tail_lo = tl.maximum(b_lo, n_blocks_prefill)
        for b in range(tail_lo, b_hi):
            t0 = (b - n_blocks_prefill) * BLOCK_T
            t_offsets = t0 + tl.arange(0, BLOCK_T)
            t_mask = t_offsets < T_tail
            base = (pid_h * T_tail + t_offsets[:, None]) * head_dim + d_offsets[None, :]
            mask2d = t_mask[:, None] & d_mask[None, :]
            k_blk = tl.load(tail_k_ptr + base, mask=mask2d, other=0.0).to(tl.float32)
            v_blk = tl.load(tail_v_ptr + base, mask=mask2d, other=0.0).to(tl.float32)
            key_positions = T + t_offsets

            scores = tl.dot(q, tl.trans(k_blk), input_precision="ieee") * sm_scale
            valid = h_mask[:, None] & t_mask[None, :]
            if CAUSAL:
                valid = valid & (query_position >= key_positions)[None, :]
            scores = tl.where(valid, scores, float("-inf"))

            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(m, block_max)
            new_max_safe = tl.where(
                (new_max == float("-inf")) | (new_max == float("inf")), 0.0, new_max
            )
            correction = tl.exp(m - new_max_safe)
            p = tl.exp(scores - new_max_safe[:, None])
            l = l * correction + tl.sum(p, axis=1)
            # Rescale the absorbed prefill state with the dense-tail update.
            acc = acc * correction[:, None] + tl.dot(p, v_blk, input_precision="ieee")
            s = s * correction[:, None]
            m = new_max

        # Store raw partial state; reduction merges before division.
        base_ml = (pid_h * n_splits_alloc + pid_s) * n_rep
        tl.store(m_partial_ptr + base_ml + h_offsets, m, mask=h_mask)
        tl.store(l_partial_ptr + base_ml + h_offsets, l, mask=h_mask)
        tl.store(
            acc_partial_ptr + (base_ml + h_offsets[:, None]) * head_dim + d_offsets[None, :],
            acc, mask=h_mask[:, None] & d_mask[None, :],
        )
        # Padded rank columns are masked and remain zero.
        tl.store(
            s_partial_ptr + (base_ml + h_offsets[:, None]) * BLOCK_RV_FULL + rv_offsets[None, :],
            s, mask=h_mask[:, None],
        )

    @triton.jit
    def _fused_decode_reduce_kernel(
        m_partial_ptr, l_partial_ptr, acc_partial_ptr, s_partial_ptr,
        core_v_ptr, feature_factor_v_ptr,
        out_ptr,
        n_splits, n_splits_alloc, n_rep, head_dim,
        token_rank_v, feature_rank_v, n_kv, slot,
        BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr,
        BLOCK_FV: tl.constexpr, BLOCK_RV_FULL: tl.constexpr,
    ):
        """Merge split softmax states and expand low-rank V once per head."""
        pid_h = tl.program_id(0)
        h_offsets = tl.arange(0, BLOCK_H)
        h_mask = h_offsets < n_rep
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < head_dim

        # Find the true maximum across splits.
        m = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
        for s_i in range(0, n_splits):
            base_ml = (pid_h * n_splits_alloc + s_i) * n_rep
            m_s = tl.load(m_partial_ptr + base_ml + h_offsets, mask=h_mask, other=float("-inf"))
            m = tl.maximum(m, m_s)
        # Use a finite substitute only inside exponentials.
        m_safe = tl.where((m == float("-inf")) | (m == float("inf")), 0.0, m)

        # Merge all partial numerators under the same coefficients.
        l = tl.zeros((BLOCK_H,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)
        rv_offsets = tl.arange(0, BLOCK_RV_FULL)
        rv_mask = rv_offsets < token_rank_v
        s_merged = tl.zeros((BLOCK_H, BLOCK_RV_FULL), dtype=tl.float32)
        for s_i in range(0, n_splits):
            base_ml = (pid_h * n_splits_alloc + s_i) * n_rep
            m_s = tl.load(m_partial_ptr + base_ml + h_offsets, mask=h_mask, other=float("-inf"))
            c_s = tl.exp(m_s - m_safe)  # exactly 0 for an empty (-inf) split
            l_s = tl.load(l_partial_ptr + base_ml + h_offsets, mask=h_mask, other=0.0)
            acc_s = tl.load(
                acc_partial_ptr + (base_ml + h_offsets[:, None]) * head_dim + d_offsets[None, :],
                mask=h_mask[:, None] & d_mask[None, :], other=0.0,
            )
            s_s = tl.load(
                s_partial_ptr + (base_ml + h_offsets[:, None]) * BLOCK_RV_FULL + rv_offsets[None, :],
                mask=h_mask[:, None] & rv_mask[None, :], other=0.0,
            )
            l += l_s * c_s
            acc += acc_s * c_s[:, None]
            s_merged += s_s * c_s[:, None]

        # Expand the merged low-rank V state once per KV head.
        core_v_base = (slot * n_kv + pid_h) * token_rank_v * feature_rank_v
        for f0 in range(0, feature_rank_v, BLOCK_FV):
            f_offsets = f0 + tl.arange(0, BLOCK_FV)
            f_mask = f_offsets < feature_rank_v
            core_blk = tl.load(
                core_v_ptr + core_v_base + rv_offsets[:, None] * feature_rank_v + f_offsets[None, :],
                mask=rv_mask[:, None] & f_mask[None, :], other=0.0,
            ).to(tl.float32)
            mid = tl.dot(s_merged, core_blk, input_precision="ieee")  # (BLOCK_H, BLOCK_FV)
            ff_blk = tl.load(
                feature_factor_v_ptr + d_offsets[:, None] * feature_rank_v + f_offsets[None, :],
                mask=d_mask[:, None] & f_mask[None, :], other=0.0,
            ).to(tl.float32)
            acc += tl.dot(mid, tl.trans(ff_blk), input_precision="ieee")

        # Return zeros for rows with no valid attention mass.
        l_safe = tl.where(l == 0.0, 1.0, l)
        out = acc / l_safe[:, None]
        q_rows = pid_h * n_rep + h_offsets
        tl.store(
            out_ptr + q_rows[:, None] * head_dim + d_offsets[None, :],
            out, mask=h_mask[:, None] & d_mask[None, :],
        )


if TRITON_AVAILABLE:

    @triton.jit
    def _fused_decode_qrope_kernel(
        q_ptr,
        token_factor_k_ptr, core_k_ptr, feature_factor_k_ptr,
        packed_k_ptr, scale_k_ptr, rotation_k_ptr,
        token_factor_v_ptr,
        packed_v_ptr, scale_v_ptr, rotation_v_ptr,
        cos_ptr, sin_ptr,
        tail_k_ptr, tail_v_ptr,
        m_partial_ptr, l_partial_ptr, acc_partial_ptr, s_partial_ptr,
        T, T_tail,
        token_rank_k, feature_rank_k, token_rank_v,
        head_dim, n_kv, n_rep, slot, query_position,
        sm_scale,
        n_splits_alloc, blocks_per_split, n_blocks_prefill, n_blocks_total,
        bits_k: tl.constexpr, vpb_k: tl.constexpr, HAS_RESIDUAL_K: tl.constexpr,
        bits_v: tl.constexpr, vpb_v: tl.constexpr, HAS_RESIDUAL_V: tl.constexpr,
        CAUSAL: tl.constexpr,
        BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_HT: tl.constexpr,
        BLOCK_RK: tl.constexpr, BLOCK_FK: tl.constexpr,
        BLOCK_RV_FULL: tl.constexpr,
        BLOCK_D: tl.constexpr, BLOCK_DI: tl.constexpr,
    ):
        """Q-side RoPE-transfer variant of the fused decode kernel."""
        pid_h = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_q = tl.program_id(2)

        b_lo = pid_s * blocks_per_split
        b_hi = tl.minimum(b_lo + blocks_per_split, n_blocks_total)

        half = head_dim // 2
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < head_dim
        d_shifted = tl.where(d_offsets < half, d_offsets + half, d_offsets - half)
        sign = tl.where(d_offsets < half, -1.0, 1.0).to(tl.float32)

        h_offsets = tl.arange(0, BLOCK_H)
        h_mask = h_offsets < n_rep
        # One program handles one query row; split/reduce retain GQA mapping.
        q_rows = pid_h * n_rep + pid_q + h_offsets
        q = tl.load(
            q_ptr + q_rows[:, None] * head_dim + d_offsets[None, :],
            mask=h_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)
        q_shift = tl.load(
            q_ptr + q_rows[:, None] * head_dim + d_shifted[None, :],
            mask=h_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)
        q_rot = sign[None, :] * q_shift

        # Flatten query and token axes for dot-compatible tile dimensions.
        ht_offsets = tl.arange(0, BLOCK_HT)
        ht_t_local = ht_offsets % BLOCK_T

        m = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
        l = tl.zeros((BLOCK_H,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)
        rv_offsets = tl.arange(0, BLOCK_RV_FULL)
        rv_mask = rv_offsets < token_rank_v
        s = tl.zeros((BLOCK_H, BLOCK_RV_FULL), dtype=tl.float32)

        core_k_base = (slot * n_kv + pid_h) * token_rank_k * feature_rank_k

        # Compressed prefill: Q-side RoPE transfer and V absorption.
        p_hi = tl.minimum(b_hi, n_blocks_prefill)
        for b in range(b_lo, p_hi):
            t0 = b * BLOCK_T
            t_offsets = t0 + tl.arange(0, BLOCK_T)
            t_mask = t_offsets < T

            cos_blk = tl.load(
                cos_ptr + t_offsets[:, None] * head_dim + d_offsets[None, :],
                mask=t_mask[:, None] & d_mask[None, :], other=0.0,
            ).to(tl.float32)
            sin_blk = tl.load(
                sin_ptr + t_offsets[:, None] * head_dim + d_offsets[None, :],
                mask=t_mask[:, None] & d_mask[None, :], other=0.0,
            ).to(tl.float32)
            # Invert RoPE on q for pre-RoPE compressed keys only.
            w_blk = (
                q[:, None, :] * cos_blk[None, :, :]
                - q_rot[:, None, :] * sin_blk[None, :, :]
            )
            w_flat = tl.reshape(w_blk, (BLOCK_HT, BLOCK_D))

            # Contract feature tiles before token-rank chunks to reuse z.
            scores_low_flat = tl.zeros((BLOCK_HT,), dtype=tl.float32)
            for f0 in range(0, feature_rank_k, BLOCK_FK):
                f_offsets = f0 + tl.arange(0, BLOCK_FK)
                f_mask = f_offsets < feature_rank_k
                z = tl.dot(
                    w_flat,
                    tl.load(
                        feature_factor_k_ptr
                        + d_offsets[:, None] * feature_rank_k
                        + f_offsets[None, :],
                        mask=d_mask[:, None] & f_mask[None, :], other=0.0,
                    ).to(tl.float32),
                    input_precision="ieee",
                )
                for r0 in range(0, token_rank_k, BLOCK_RK):
                    r_offsets = r0 + tl.arange(0, BLOCK_RK)
                    r_mask = r_offsets < token_rank_k
                    core_rf = tl.load(
                        core_k_ptr
                        + core_k_base
                        + r_offsets[:, None] * feature_rank_k
                        + f_offsets[None, :],
                        mask=r_mask[:, None] & f_mask[None, :], other=0.0,
                    ).to(tl.float32)
                    h_chunk = tl.dot(z, tl.trans(core_rf), input_precision="ieee")
                    u_t_flat = tl.load(
                        token_factor_k_ptr
                        + (t0 + ht_t_local)[:, None] * token_rank_k
                        + r_offsets[None, :],
                        mask=((t0 + ht_t_local)[:, None] < T) & r_mask[None, :], other=0.0,
                    ).to(tl.float32)
                    scores_low_flat += tl.sum(h_chunk * u_t_flat, axis=1)

            scores_res_flat = tl.zeros((BLOCK_HT,), dtype=tl.float32)
            if HAS_RESIDUAL_K:
                # Tile the position-only rotation contraction.
                scale_flat = tl.load(
                    scale_k_ptr + pid_h * T + (t0 + ht_t_local),
                    mask=(t0 + ht_t_local) < T, other=0.0,
                ).to(tl.float32)
                res_w_flat = tl.zeros((BLOCK_HT, BLOCK_D), dtype=tl.float32)
                for di0 in range(0, head_dim, BLOCK_DI):
                    di_offsets = di0 + tl.arange(0, BLOCK_DI)
                    di_mask = di_offsets < head_dim
                    d_shifted_chunk = tl.where(
                        di_offsets < half, di_offsets + half, di_offsets - half
                    )
                    sign_chunk = tl.where(di_offsets < half, -1.0, 1.0).to(tl.float32)
                    q_chunk = tl.load(
                        q_ptr + q_rows[:, None] * head_dim + di_offsets[None, :],
                        mask=h_mask[:, None] & di_mask[None, :], other=0.0,
                    ).to(tl.float32)
                    q_shift_chunk = tl.load(
                        q_ptr + q_rows[:, None] * head_dim + d_shifted_chunk[None, :],
                        mask=h_mask[:, None] & di_mask[None, :], other=0.0,
                    ).to(tl.float32)
                    cos_chunk = tl.load(
                        cos_ptr + t_offsets[:, None] * head_dim + di_offsets[None, :],
                        mask=t_mask[:, None] & di_mask[None, :], other=0.0,
                    ).to(tl.float32)
                    sin_chunk = tl.load(
                        sin_ptr + t_offsets[:, None] * head_dim + di_offsets[None, :],
                        mask=t_mask[:, None] & di_mask[None, :], other=0.0,
                    ).to(tl.float32)
                    w_chunk = (
                        q_chunk[:, None, :] * cos_chunk[None, :, :]
                        - (sign_chunk[None, :] * q_shift_chunk)[:, None, :]
                        * sin_chunk[None, :, :]
                    )
                    w_chunk_flat = tl.reshape(w_chunk, (BLOCK_HT, BLOCK_DI))
                    rot_k_chunk = tl.load(
                        rotation_k_ptr
                        + d_offsets[:, None] * head_dim
                        + di_offsets[None, :],
                        mask=d_mask[:, None] & di_mask[None, :], other=0.0,
                    ).to(tl.float32)
                    res_w_chunk = tl.dot(
                        w_chunk_flat,
                        tl.trans(rot_k_chunk),
                        input_precision="ieee",
                    )
                    res_w_flat += res_w_chunk
                # Rotate once per block, outside the code chunks.
                idx_k = (
                    pid_h * T + (t0 + ht_t_local)[:, None]
                ) * head_dim + d_offsets[None, :]
                lane_mask_k = ((t0 + ht_t_local)[:, None] < T) & d_mask[None, :]
                codes_k = _decode_lane(
                    packed_k_ptr, idx_k, lane_mask_k, bits_k, vpb_k
                ).to(tl.float32)
                scores_res_flat += tl.sum(
                    res_w_flat * codes_k * scale_flat[:, None], axis=1
                )

            # Accumulate low-rank V in rank space for reduction.
            u_v_blk = tl.load(
                token_factor_v_ptr + t_offsets[:, None] * token_rank_v + rv_offsets[None, :],
                mask=t_mask[:, None] & rv_mask[None, :], other=0.0,
            ).to(tl.float32)
            if HAS_RESIDUAL_V:
                scale_v_blk = tl.load(
                    scale_v_ptr + pid_h * T + t_offsets, mask=t_mask, other=0.0
                ).to(tl.float32)
                res_v = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
                for di0 in range(0, head_dim, BLOCK_DI):
                    di_offsets = di0 + tl.arange(0, BLOCK_DI)
                    di_mask = di_offsets < head_dim
                    idx_v = (pid_h * T + t_offsets[:, None]) * head_dim + di_offsets[None, :]
                    lane_mask_v = t_mask[:, None] & di_mask[None, :]
                    codes_v = _decode_lane(
                        packed_v_ptr, idx_v, lane_mask_v, bits_v, vpb_v
                    ).to(tl.float32)
                    rotated_v = codes_v * scale_v_blk[:, None]
                    rot_v_chunk = tl.load(
                        rotation_v_ptr
                        + di_offsets[:, None] * head_dim
                        + d_offsets[None, :],
                        mask=di_mask[:, None] & d_mask[None, :], other=0.0,
                    ).to(tl.float32)
                    res_v += tl.dot(rotated_v, rot_v_chunk, input_precision="ieee")

            scores = tl.reshape(
                (scores_low_flat + scores_res_flat) * sm_scale,
                (BLOCK_H, BLOCK_T),
            )
            valid = h_mask[:, None] & t_mask[None, :]
            if CAUSAL:
                valid = valid & (query_position >= t_offsets)[None, :]
            scores = tl.where(valid, scores, float("-inf"))

            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(m, block_max)
            new_max_safe = tl.where(
                (new_max == float("-inf")) | (new_max == float("inf")), 0.0, new_max
            )
            correction = tl.exp(m - new_max_safe)
            p = tl.exp(scores - new_max_safe[:, None])
            l = l * correction + tl.sum(p, axis=1)
            acc = acc * correction[:, None]
            if HAS_RESIDUAL_V:
                acc += tl.sum(p[:, :, None] * res_v[None, :, :], axis=1)
            s = s * correction[:, None] + tl.sum(
                p[:, :, None] * u_v_blk[None, :, :], axis=1
            )
            m = new_max

        # Dense tail is post-RoPE and uses the original query.
        tail_lo = tl.maximum(b_lo, n_blocks_prefill)
        for b in range(tail_lo, b_hi):
            t0 = (b - n_blocks_prefill) * BLOCK_T
            t_offsets = t0 + tl.arange(0, BLOCK_T)
            t_mask = t_offsets < T_tail
            base = (pid_h * T_tail + t_offsets[:, None]) * head_dim + d_offsets[None, :]
            mask2d = t_mask[:, None] & d_mask[None, :]
            k_blk = tl.load(tail_k_ptr + base, mask=mask2d, other=0.0).to(tl.float32)
            v_blk = tl.load(tail_v_ptr + base, mask=mask2d, other=0.0).to(tl.float32)
            key_positions = T + t_offsets

            scores = tl.sum(
                q[:, None, :] * k_blk[None, :, :], axis=2
            ) * sm_scale
            valid = h_mask[:, None] & t_mask[None, :]
            if CAUSAL:
                valid = valid & (query_position >= key_positions)[None, :]
            scores = tl.where(valid, scores, float("-inf"))

            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(m, block_max)
            new_max_safe = tl.where(
                (new_max == float("-inf")) | (new_max == float("inf")), 0.0, new_max
            )
            correction = tl.exp(m - new_max_safe)
            p = tl.exp(scores - new_max_safe[:, None])
            l = l * correction + tl.sum(p, axis=1)
            acc = acc * correction[:, None] + tl.sum(
                p[:, :, None] * v_blk[None, :, :], axis=1
            )
            s = s * correction[:, None]
            m = new_max

        base_ml = (pid_h * n_splits_alloc + pid_s) * n_rep + pid_q
        tl.store(m_partial_ptr + base_ml + h_offsets, m, mask=h_mask)
        tl.store(l_partial_ptr + base_ml + h_offsets, l, mask=h_mask)
        tl.store(
            acc_partial_ptr + (base_ml + h_offsets[:, None]) * head_dim + d_offsets[None, :],
            acc, mask=h_mask[:, None] & d_mask[None, :],
        )
        # Full padded width: columns >= token_rank_v are exact zeros by
        # construction (masked u_v loads), and the reduce masks them out on
        # load anyway.
        tl.store(
            s_partial_ptr + (base_ml + h_offsets[:, None]) * BLOCK_RV_FULL + rv_offsets[None, :],
            s, mask=h_mask[:, None],
        )


# The fused path uses smaller blocks because several tiles are live together.
_BLOCK_T_DECODE = 32
# Bound the residual rotation contraction tile.
_BLOCK_DI = 32
# Target several waves of programs while keeping scratch T-independent.
_SPLIT_WAVES = 4
_SM_COUNT_CACHE: dict[int, int] = {}


def _num_sms(device: torch.device) -> int:
    """Streaming-multiprocessor count for `device`, cached per device index
    (`torch.cuda.get_device_properties` is a driver call; this runs on every
    decode step)."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SM_COUNT_CACHE:
        _SM_COUNT_CACHE[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SM_COUNT_CACHE[idx]


def fused_decode_attention_cpu(
    tucker_k: TuckerGroupFactors,
    residual_k: ResidualPayload | None,
    tucker_v: TuckerGroupFactors,
    residual_v: ResidualPayload | None,
    slot: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    query_position: int,
    n_rep: int,
    tail_k: torch.Tensor | None = None,
    tail_v: torch.Tensor | None = None,
    sm_scale: float = 1.0,
    causal: bool = True,
    block_size: int = 128,
) -> torch.Tensor:
    """CPU fallback for `fused_decode_attention_triton` below."""
    n_kv = tucker_k.num_kv_heads
    n_q = query.shape[0]
    if n_q != n_kv * n_rep:
        raise ValueError(
            f"fused_decode_attention_cpu: query has {n_q} rows, expected n_kv*n_rep={n_kv * n_rep}"
        )
    tucker_k_slot = TuckerGroupFactors(
        core=tucker_k.core[slot:slot + 1],
        token_factor=tucker_k.token_factor,
        feature_factor=tucker_k.feature_factor,
    )
    tucker_v_slot = TuckerGroupFactors(
        core=tucker_v.core[slot:slot + 1],
        token_factor=tucker_v.token_factor,
        feature_factor=tucker_v.feature_factor,
    )
    layout = GroupLayout(groups=[[0]])
    head_map = HeadMap(num_query_heads=n_q, num_kv_heads=n_kv)
    group = FactorizedKVGroup(
        layout=layout, tucker_k=tucker_k_slot, tucker_v=tucker_v_slot,
        residual_k={0: residual_k}, residual_v={0: residual_v}, head_map=head_map,
    )
    group.validate()
    attn = AttentionInputs(
        query=query.unsqueeze(1),
        query_positions=torch.tensor([query_position], dtype=torch.int64, device=query.device),
        cos=cos, sin=sin, scale=sm_scale, causal=causal,
        tail_k=tail_k, tail_v=tail_v,
    )
    out = _reference.attention_on_factors(group, 0, attn, block_size=block_size)
    return out.squeeze(1)


def fused_decode_attention_qrope_cpu(
    tucker_k: TuckerGroupFactors,
    residual_k: ResidualPayload | None,
    tucker_v: TuckerGroupFactors,
    residual_v: ResidualPayload | None,
    slot: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    query_position: int,
    n_rep: int,
    tail_k: torch.Tensor | None = None,
    tail_v: torch.Tensor | None = None,
    sm_scale: float = 1.0,
    causal: bool = True,
    block_size: int = 128,
) -> torch.Tensor:
    """CPU oracle-backed twin of `fused_decode_attention_cpu`."""
    n_kv = tucker_k.num_kv_heads
    n_q = query.shape[0]
    if n_q != n_kv * n_rep:
        raise ValueError(
            f"fused_decode_attention_qrope_cpu: query has {n_q} rows, "
            f"expected n_kv*n_rep={n_kv * n_rep}"
        )
    tucker_k_slot = TuckerGroupFactors(
        core=tucker_k.core[slot:slot + 1],
        token_factor=tucker_k.token_factor,
        feature_factor=tucker_k.feature_factor,
    )
    tucker_v_slot = TuckerGroupFactors(
        core=tucker_v.core[slot:slot + 1],
        token_factor=tucker_v.token_factor,
        feature_factor=tucker_v.feature_factor,
    )
    layout = GroupLayout(groups=[[0]])
    head_map = HeadMap(num_query_heads=n_q, num_kv_heads=n_kv)
    group = FactorizedKVGroup(
        layout=layout, tucker_k=tucker_k_slot, tucker_v=tucker_v_slot,
        residual_k={0: residual_k}, residual_v={0: residual_v}, head_map=head_map,
    )
    group.validate()
    attn = AttentionInputs(
        query=query.unsqueeze(1),
        query_positions=torch.tensor([query_position], dtype=torch.int64, device=query.device),
        cos=cos, sin=sin, scale=sm_scale, causal=causal,
        tail_k=tail_k, tail_v=tail_v,
    )
    out = _reference.attention_on_factors_qrope(group, 0, attn, block_size=block_size)
    return out.squeeze(1)


@dataclass
class _FusedDecodePartials:
    """Per-split softmax state and launch geometry for reduction."""

    m_partial: torch.Tensor    # (n_kv * n_splits_alloc * n_rep,) fp32, TRUE running max
    l_partial: torch.Tensor    # (n_kv * n_splits_alloc * n_rep,) fp32, raw denominator
    acc_partial: torch.Tensor  # (n_kv * n_splits_alloc * n_rep * head_dim,) fp32; V residual + dense tail
    s_partial: torch.Tensor    # (n_kv * n_splits_alloc * n_rep * rv_pad,) fp32; absorbed rank-sized V state
    n_splits: int              # launched split count (only these entries are valid)
    n_splits_alloc: int        # T-independent scratch cap (row stride)
    n_kv: int
    n_rep: int
    n_q: int
    head_dim: int
    block_h: int
    block_d: int
    device: torch.device
    core_v: torch.Tensor           # contiguous (n_slots, n_kv, r_t_v, r_f_v)
    feature_factor_v: torch.Tensor  # contiguous (head_dim, r_f_v)
    token_rank_v: int
    feature_rank_v: int
    slot: int
    rv_pad: int    # padded s row width == BLOCK_RV_FULL (power of two >= 16)


def _fused_decode_split_partials(
    tucker_k: TuckerGroupFactors,
    residual_k: ResidualPayload | None,
    tucker_v: TuckerGroupFactors,
    residual_v: ResidualPayload | None,
    slot: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    query_position: int,
    n_rep: int,
    tail_k: torch.Tensor | None = None,
    tail_v: torch.Tensor | None = None,
    sm_scale: float = 1.0,
    causal: bool = True,
) -> _FusedDecodePartials:
    """Validate inputs, derive the flash-decoding split geometry, launch
    `_fused_decode_kernel`, and return the raw per-split partial state (see
    `_FusedDecodePartials` for why this is a separate, test-visible step).
    `fused_decode_attention_triton` is exactly this + `_fused_decode_reduce`
    + the nonfinite-denominator rejection.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "fused_decode_attention_triton() called but `triton` is not importable "
            "in this environment -- use fused_decode_attention() for automatic CPU "
            "fallback, or run this on a host with a CUDA Triton install."
        )
    if not torch.cuda.is_available() or not tucker_k.core.is_cuda:
        raise RuntimeError(
            "fused_decode_attention_triton() requires tucker tensors on a CUDA "
            f"device with CUDA available; got tucker_k.core.device={tucker_k.core.device}, "
            f"torch.cuda.is_available()={torch.cuda.is_available()}. Use "
            "fused_decode_attention() for automatic CPU fallback."
        )
    head_dim = tucker_k.head_dim
    if head_dim % 2 != 0:
        raise ValueError(f"fused_decode_attention_triton: head_dim must be even for RoPE, got {head_dim}")

    n_kv = tucker_k.num_kv_heads
    T = tucker_k.seq_len
    n_q = query.shape[0]
    if n_q != n_kv * n_rep:
        raise ValueError(
            f"fused_decode_attention_triton: query has {n_q} rows, expected n_kv*n_rep={n_kv * n_rep}"
        )
    device = tucker_k.core.device

    token_factor_k = tucker_k.token_factor.contiguous()
    core_k = tucker_k.core.contiguous()
    feature_factor_k = tucker_k.feature_factor.contiguous()
    token_factor_v = tucker_v.token_factor.contiguous()
    core_v = tucker_v.core.contiguous()
    feature_factor_v = tucker_v.feature_factor.contiguous()
    query = query.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    has_residual_k = residual_k is not None and residual_k.bits > 0
    if has_residual_k:
        packed_k = residual_k.packed.contiguous()
        scale_k = residual_k.scale.contiguous()
        rotation_k = residual_k.rotation.contiguous()
        bits_k = residual_k.bits
        vpb_k = 1 if bits_k == 8 else 8 // bits_k
    else:
        packed_k = torch.empty(1, dtype=torch.uint8, device=device)
        scale_k = torch.empty(1, dtype=torch.float32, device=device)
        rotation_k = torch.empty(1, dtype=torch.float32, device=device)
        bits_k, vpb_k = 8, 1

    has_residual_v = residual_v is not None and residual_v.bits > 0
    if has_residual_v:
        packed_v = residual_v.packed.contiguous()
        scale_v = residual_v.scale.contiguous()
        rotation_v = residual_v.rotation.contiguous()
        bits_v = residual_v.bits
        vpb_v = 1 if bits_v == 8 else 8 // bits_v
    else:
        packed_v = torch.empty(1, dtype=torch.uint8, device=device)
        scale_v = torch.empty(1, dtype=torch.float32, device=device)
        rotation_v = torch.empty(1, dtype=torch.float32, device=device)
        bits_v, vpb_v = 8, 1

    has_tail = tail_k is not None and tail_k.shape[1] > 0
    if has_tail:
        T_tail = tail_k.shape[1]
        tail_k_c = tail_k.contiguous()
        tail_v_c = tail_v.contiguous()
    else:
        T_tail = 0
        # Tail pointers are unused when T_tail is zero.
        tail_k_c = torch.empty(1, dtype=torch.float32, device=device)
        tail_v_c = torch.empty(1, dtype=torch.float32, device=device)

    # Select a deterministic shared-memory-safe launch configuration.
    _shape = launch_config.KernelShapeParams(
        token_rank_k=tucker_k.token_rank,
        feature_rank_k=tucker_k.feature_rank,
        token_rank_v=tucker_v.token_rank,
        head_dim=head_dim,
        n_rep=n_rep,
    )
    _limit_bytes = launch_config.query_device_shared_mem_limit(device)
    _selected = launch_config.select_launch_config(_shape, _limit_bytes)
    _cfg = _selected.config
    logger.info(
        "fused_decode_attention_triton: launch-config rung %d selected "
        "(num_stages=%d, rank_cap=%d, block_di=%d, block_t=%d), estimated "
        "%d bytes <= device limit %d bytes",
        _selected.rung_index, _cfg.num_stages, _cfg.rank_cap, _cfg.block_di,
        _cfg.block_t, _selected.estimated_bytes, _selected.limit_bytes,
    )

    BLOCK_RK = _cfg.block_rk
    BLOCK_FK = _cfg.block_fk
    BLOCK_D = _cfg.block_d
    BLOCK_H = _cfg.block_h
    BLOCK_DI = _cfg.block_di_eff
    BLOCK_T_DECODE = _cfg.block_t

    # Split virtual prefill/tail blocks; scratch uses a T-independent capacity.
    n_blocks_prefill = triton.cdiv(T, BLOCK_T_DECODE)
    n_blocks_total = n_blocks_prefill + triton.cdiv(T_tail, BLOCK_T_DECODE)
    n_splits_alloc = max(1, triton.cdiv(_SPLIT_WAVES * _num_sms(device), n_kv))
    n_splits = max(1, min(n_splits_alloc, n_blocks_total))
    blocks_per_split = triton.cdiv(n_blocks_total, n_splits)

    # Allocate fp32 partial states with T-independent row stride.
    BLOCK_RV_FULL = _cfg.block_rv_full
    m_partial = torch.empty(n_kv * n_splits_alloc * n_rep, dtype=torch.float32, device=device)
    l_partial = torch.empty(n_kv * n_splits_alloc * n_rep, dtype=torch.float32, device=device)
    acc_partial = torch.empty(
        n_kv * n_splits_alloc * n_rep * head_dim, dtype=torch.float32, device=device
    )
    s_partial = torch.empty(
        n_kv * n_splits_alloc * n_rep * BLOCK_RV_FULL, dtype=torch.float32, device=device
    )

    grid = (n_kv, n_splits)
    _fused_decode_kernel[grid](
        query,
        token_factor_k, core_k, feature_factor_k,
        packed_k, scale_k, rotation_k,
        token_factor_v,
        packed_v, scale_v, rotation_v,
        cos, sin,
        tail_k_c, tail_v_c,
        m_partial, l_partial, acc_partial, s_partial,
        T, T_tail,
        tucker_k.token_rank, tucker_k.feature_rank, tucker_v.token_rank,
        head_dim, n_kv, n_rep, slot, query_position,
        sm_scale,
        n_splits_alloc, blocks_per_split, n_blocks_prefill, n_blocks_total,
        bits_k=bits_k, vpb_k=vpb_k, HAS_RESIDUAL_K=has_residual_k,
        bits_v=bits_v, vpb_v=vpb_v, HAS_RESIDUAL_V=has_residual_v,
        CAUSAL=causal,
        BLOCK_T=BLOCK_T_DECODE, BLOCK_H=BLOCK_H,
        BLOCK_RK=BLOCK_RK, BLOCK_FK=BLOCK_FK,
        BLOCK_RV_FULL=BLOCK_RV_FULL,
        BLOCK_D=BLOCK_D, BLOCK_DI=BLOCK_DI,
        num_stages=_cfg.num_stages,
    )
    return _FusedDecodePartials(
        m_partial=m_partial, l_partial=l_partial, acc_partial=acc_partial,
        s_partial=s_partial,
        n_splits=n_splits, n_splits_alloc=n_splits_alloc,
        n_kv=n_kv, n_rep=n_rep, n_q=n_q, head_dim=head_dim,
        block_h=BLOCK_H, block_d=BLOCK_D, device=device,
        core_v=core_v, feature_factor_v=feature_factor_v,
        token_rank_v=tucker_v.token_rank, feature_rank_v=tucker_v.feature_rank,
        slot=slot, rv_pad=BLOCK_RV_FULL,
    )


def _fused_decode_qrope_split_partials(
    tucker_k: TuckerGroupFactors,
    residual_k: ResidualPayload | None,
    tucker_v: TuckerGroupFactors,
    residual_v: ResidualPayload | None,
    slot: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    query_position: int,
    n_rep: int,
    tail_k: torch.Tensor | None = None,
    tail_v: torch.Tensor | None = None,
    sm_scale: float = 1.0,
    causal: bool = True,
) -> _FusedDecodePartials:
    """Q-side RoPE-transfer launcher parallel to `_fused_decode_split_partials`."""
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "fused_decode_attention_qrope_triton() called but `triton` is not importable "
            "in this environment -- use fused_decode_attention() for automatic CPU "
            "fallback, or run this on a host with a CUDA Triton install."
        )
    if not torch.cuda.is_available() or not tucker_k.core.is_cuda:
        raise RuntimeError(
            "fused_decode_attention_qrope_triton() requires tucker tensors on a CUDA "
            f"device with CUDA available; got tucker_k.core.device={tucker_k.core.device}, "
            f"torch.cuda.is_available()={torch.cuda.is_available()}. Use "
            "fused_decode_attention() for automatic CPU fallback."
        )
    head_dim = tucker_k.head_dim
    if head_dim % 2 != 0:
        raise ValueError(
            "fused_decode_attention_qrope_triton: head_dim must be even for RoPE, "
            f"got {head_dim}"
        )

    n_kv = tucker_k.num_kv_heads
    T = tucker_k.seq_len
    n_q = query.shape[0]
    if n_q != n_kv * n_rep:
        raise ValueError(
            f"fused_decode_attention_qrope_triton: query has {n_q} rows, "
            f"expected n_kv*n_rep={n_kv * n_rep}"
        )
    device = tucker_k.core.device

    token_factor_k = tucker_k.token_factor.contiguous()
    core_k = tucker_k.core.contiguous()
    feature_factor_k = tucker_k.feature_factor.contiguous()
    token_factor_v = tucker_v.token_factor.contiguous()
    core_v = tucker_v.core.contiguous()
    feature_factor_v = tucker_v.feature_factor.contiguous()
    query = query.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    has_residual_k = residual_k is not None and residual_k.bits > 0
    if has_residual_k:
        packed_k = residual_k.packed.contiguous()
        scale_k = residual_k.scale.contiguous()
        rotation_k = residual_k.rotation.contiguous()
        bits_k = residual_k.bits
        vpb_k = 1 if bits_k == 8 else 8 // bits_k
    else:
        packed_k = torch.empty(1, dtype=torch.uint8, device=device)
        scale_k = torch.empty(1, dtype=torch.float32, device=device)
        rotation_k = torch.empty(1, dtype=torch.float32, device=device)
        bits_k, vpb_k = 8, 1

    has_residual_v = residual_v is not None and residual_v.bits > 0
    if has_residual_v:
        packed_v = residual_v.packed.contiguous()
        scale_v = residual_v.scale.contiguous()
        rotation_v = residual_v.rotation.contiguous()
        bits_v = residual_v.bits
        vpb_v = 1 if bits_v == 8 else 8 // bits_v
    else:
        packed_v = torch.empty(1, dtype=torch.uint8, device=device)
        scale_v = torch.empty(1, dtype=torch.float32, device=device)
        rotation_v = torch.empty(1, dtype=torch.float32, device=device)
        bits_v, vpb_v = 8, 1

    has_tail = tail_k is not None and tail_k.shape[1] > 0
    if has_tail:
        T_tail = tail_k.shape[1]
        tail_k_c = tail_k.contiguous()
        tail_v_c = tail_v.contiguous()
    else:
        T_tail = 0
        tail_k_c = torch.empty(1, dtype=torch.float32, device=device)
        tail_v_c = torch.empty(1, dtype=torch.float32, device=device)

    _shape = launch_config.KernelShapeParams(
        token_rank_k=tucker_k.token_rank,
        feature_rank_k=tucker_k.feature_rank,
        token_rank_v=tucker_v.token_rank,
        head_dim=head_dim,
        n_rep=n_rep,
    )
    _limit_bytes = launch_config.query_device_shared_mem_limit(device)
    _selected = launch_config.select_launch_config(_shape, _limit_bytes)
    _cfg = _selected.config
    logger.info(
        "fused_decode_attention_qrope_triton: launch-config rung %d selected "
        "(num_stages=%d, rank_cap=%d, block_di=%d, block_t=%d), estimated "
        "%d bytes <= device limit %d bytes",
        _selected.rung_index, _cfg.num_stages, _cfg.rank_cap, _cfg.block_di,
        _cfg.block_t, _selected.estimated_bytes, _selected.limit_bytes,
    )

    # Q-RoPE uses a smaller rank tile and one pipeline stage.
    BLOCK_RK = min(_cfg.block_rk, 16)
    BLOCK_FK = min(_cfg.block_fk, 16)
    BLOCK_D = _cfg.block_d
    # Q-RoPE tiles one query row while the reduce kernel keeps BLOCK_H=16.
    BLOCK_H = 1
    if BLOCK_H != 1:
        raise RuntimeError(
            "fused_decode_attention_qrope_triton: BLOCK_H must remain 1 "
            "because the third grid axis owns one query row per program"
        )
    BLOCK_DI = min(_cfg.block_di_eff, 16)
    BLOCK_T_DECODE = min(_cfg.block_t, 16)
    QROPE_NUM_STAGES = 1

    n_blocks_prefill = triton.cdiv(T, BLOCK_T_DECODE)
    n_blocks_total = n_blocks_prefill + triton.cdiv(T_tail, BLOCK_T_DECODE)
    n_splits_alloc = max(1, triton.cdiv(_SPLIT_WAVES * _num_sms(device), n_kv))
    n_splits = max(1, min(n_splits_alloc, n_blocks_total))
    blocks_per_split = triton.cdiv(n_blocks_total, n_splits)

    BLOCK_RV_FULL = _cfg.block_rv_full
    m_partial = torch.empty(n_kv * n_splits_alloc * n_rep, dtype=torch.float32, device=device)
    l_partial = torch.empty(n_kv * n_splits_alloc * n_rep, dtype=torch.float32, device=device)
    acc_partial = torch.empty(
        n_kv * n_splits_alloc * n_rep * head_dim, dtype=torch.float32, device=device
    )
    s_partial = torch.empty(
        n_kv * n_splits_alloc * n_rep * BLOCK_RV_FULL, dtype=torch.float32, device=device
    )

    # The third grid axis distributes consecutive query rows.
    grid = (n_kv, n_splits, n_rep)
    _fused_decode_qrope_kernel[grid](
        query,
        token_factor_k, core_k, feature_factor_k,
        packed_k, scale_k, rotation_k,
        token_factor_v,
        packed_v, scale_v, rotation_v,
        cos, sin,
        tail_k_c, tail_v_c,
        m_partial, l_partial, acc_partial, s_partial,
        T, T_tail,
        tucker_k.token_rank, tucker_k.feature_rank, tucker_v.token_rank,
        head_dim, n_kv, n_rep, slot, query_position,
        sm_scale,
        n_splits_alloc, blocks_per_split, n_blocks_prefill, n_blocks_total,
        bits_k=bits_k, vpb_k=vpb_k, HAS_RESIDUAL_K=has_residual_k,
        bits_v=bits_v, vpb_v=vpb_v, HAS_RESIDUAL_V=has_residual_v,
        CAUSAL=causal,
        BLOCK_T=BLOCK_T_DECODE, BLOCK_H=BLOCK_H, BLOCK_HT=BLOCK_T_DECODE,
        BLOCK_RK=BLOCK_RK, BLOCK_FK=BLOCK_FK,
        BLOCK_RV_FULL=BLOCK_RV_FULL,
        BLOCK_D=BLOCK_D, BLOCK_DI=BLOCK_DI,
        num_stages=QROPE_NUM_STAGES,
    )
    return _FusedDecodePartials(
        m_partial=m_partial, l_partial=l_partial, acc_partial=acc_partial,
        s_partial=s_partial,
        n_splits=n_splits, n_splits_alloc=n_splits_alloc,
        n_kv=n_kv, n_rep=n_rep, n_q=n_q, head_dim=head_dim,
        block_h=_cfg.block_h, block_d=BLOCK_D, device=device,
        core_v=core_v, feature_factor_v=feature_factor_v,
        token_rank_v=tucker_v.token_rank, feature_rank_v=tucker_v.feature_rank,
        slot=slot, rv_pad=BLOCK_RV_FULL,
    )


def _fused_decode_reduce(partials: _FusedDecodePartials) -> torch.Tensor:
    """Launch `_fused_decode_reduce_kernel` over the per-split partial state
    (merge + V-side absorption expansion) and return the merged fp32
    attention output `(n_q, head_dim)`. Internal counterpart of
    `_fused_decode_split_partials` (see `_FusedDecodePartials` for why the
    two steps are separately callable)."""
    out = torch.empty(
        partials.n_q, partials.head_dim, dtype=torch.float32, device=partials.device
    )

    # Select a deterministic shared-memory-safe reduce configuration.
    _reduce_shape = launch_config.ReduceKernelShapeParams(
        token_rank_v=partials.token_rank_v,
        feature_rank_v=partials.feature_rank_v,
        head_dim=partials.head_dim,
        n_rep=partials.n_rep,
    )
    _limit_bytes = launch_config.query_device_shared_mem_limit(partials.device)
    _selected = launch_config.select_reduce_launch_config(_reduce_shape, _limit_bytes)
    _cfg = _selected.config
    logger.info(
        "fused_decode_reduce: launch-config rung %d selected (num_stages=%d, "
        "block_fv=%d), estimated %d bytes <= device limit %d bytes",
        _selected.rung_index, _cfg.num_stages, _cfg.block_fv,
        _selected.estimated_bytes, _selected.limit_bytes,
    )

    if _cfg.block_rv_full != partials.rv_pad:
        raise RuntimeError(
            "fused_decode_reduce: launch-config's block_rv_full "
            f"({_cfg.block_rv_full}, recomputed from partials.token_rank_v) "
            f"disagrees with partials.rv_pad ({partials.rv_pad}, the actual "
            "s_partial row width the split kernel allocated) -- this should "
            "never happen since both derive from the same token_rank_v; a "
            "silent mismatch would make the reduce estimate model the wrong "
            "tile."
        )
    _fused_decode_reduce_kernel[(partials.n_kv,)](
        partials.m_partial, partials.l_partial, partials.acc_partial,
        partials.s_partial,
        partials.core_v, partials.feature_factor_v,
        out,
        partials.n_splits, partials.n_splits_alloc, partials.n_rep, partials.head_dim,
        partials.token_rank_v, partials.feature_rank_v, partials.n_kv, partials.slot,
        BLOCK_H=partials.block_h, BLOCK_D=partials.block_d,
        BLOCK_FV=_cfg.block_fv, BLOCK_RV_FULL=partials.rv_pad,
        num_stages=_cfg.num_stages,
    )
    return out


def fused_decode_attention_triton(
    tucker_k: TuckerGroupFactors,
    residual_k: ResidualPayload | None,
    tucker_v: TuckerGroupFactors,
    residual_v: ResidualPayload | None,
    slot: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    query_position: int,
    n_rep: int,
    tail_k: torch.Tensor | None = None,
    tail_v: torch.Tensor | None = None,
    sm_scale: float = 1.0,
    causal: bool = True,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Run the Triton fused decode-attention kernel."""
    partials = _fused_decode_split_partials(
        tucker_k, residual_k, tucker_v, residual_v, slot, cos, sin, query,
        query_position, n_rep, tail_k=tail_k, tail_v=tail_v, sm_scale=sm_scale,
        causal=causal,
    )
    out = _fused_decode_reduce(partials)
    n_kv = partials.n_kv
    n_splits = partials.n_splits
    n_splits_alloc = partials.n_splits_alloc
    n_rep = partials.n_rep
    l_partial = partials.l_partial

    # Triton cannot raise on nonfinite softmax state; reject it on the host.
    l_launched = l_partial.view(n_kv, n_splits_alloc, n_rep)[:, :n_splits, :]
    if not bool(torch.isfinite(l_launched).all()):
        raise FloatingPointError(
            "fused_decode_attention_triton: online-softmax denominator contains "
            "NaN/inf -- attention scores are not finite (e.g. a finite Q,K dot "
            "product overflowed fp32 to +inf). The normative reference "
            "(reference.py::_assert_softmax_denominator_invariants) rejects this "
            "state rather than emitting nonfinite output; this wrapper mirrors "
            "that rejection for the Triton path."
        )
    return out if out_dtype == torch.float32 else out.to(out_dtype)


def fused_decode_attention_qrope_triton(
    tucker_k: TuckerGroupFactors,
    residual_k: ResidualPayload | None,
    tucker_v: TuckerGroupFactors,
    residual_v: ResidualPayload | None,
    slot: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    query_position: int,
    n_rep: int,
    tail_k: torch.Tensor | None = None,
    tail_v: torch.Tensor | None = None,
    sm_scale: float = 1.0,
    causal: bool = True,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Q-side RoPE-transfer Triton fused decode wrapper."""
    partials = _fused_decode_qrope_split_partials(
        tucker_k, residual_k, tucker_v, residual_v, slot, cos, sin, query,
        query_position, n_rep, tail_k=tail_k, tail_v=tail_v, sm_scale=sm_scale,
        causal=causal,
    )
    out = _fused_decode_reduce(partials)
    n_kv = partials.n_kv
    n_splits = partials.n_splits
    n_splits_alloc = partials.n_splits_alloc
    n_rep = partials.n_rep
    l_partial = partials.l_partial
    l_launched = l_partial.view(n_kv, n_splits_alloc, n_rep)[:, :n_splits, :]
    if not bool(torch.isfinite(l_launched).all()):
        raise FloatingPointError(
            "fused_decode_attention_qrope_triton: online-softmax denominator "
            "contains NaN/inf -- attention scores are not finite (e.g. a finite "
            "Q,K dot product overflowed fp32 to +inf). The normative reference "
            "rejects this state rather than emitting nonfinite output; this "
            "wrapper mirrors that rejection for the Q-ROPE path."
        )
    return out if out_dtype == torch.float32 else out.to(out_dtype)


def fused_decode_attention(
    tucker_k: TuckerGroupFactors,
    residual_k: ResidualPayload | None,
    tucker_v: TuckerGroupFactors,
    residual_v: ResidualPayload | None,
    slot: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    query_position: int,
    n_rep: int,
    tail_k: torch.Tensor | None = None,
    tail_v: torch.Tensor | None = None,
    sm_scale: float = 1.0,
    causal: bool = True,
    out_dtype: torch.dtype = torch.float32,
    *,
    use_qrope_transfer: bool = False,
) -> torch.Tensor:
    """Drop-in dispatcher for fused decode attention.

    The default (`use_qrope_transfer=False`) is the existing path. The new
    keyword-only opt-in selects the parallel Q-side RoPE-transfer kernel on
    Triton/CUDA, or its Stage-1 oracle-backed CPU twin otherwise.
    """
    backend = select_backend(tucker_k.core.device)
    logger.info(
        "fused_decode_attention dispatch: backend=%s device=%s slot=%s n_rep=%s",
        backend, tucker_k.core.device, slot, n_rep,
    )
    if backend == "triton":
        if use_qrope_transfer:
            return fused_decode_attention_qrope_triton(
                tucker_k, residual_k, tucker_v, residual_v, slot, cos, sin, query,
                query_position, n_rep, tail_k=tail_k, tail_v=tail_v, sm_scale=sm_scale,
                causal=causal, out_dtype=out_dtype,
            )
        return fused_decode_attention_triton(
            tucker_k, residual_k, tucker_v, residual_v, slot, cos, sin, query,
            query_position, n_rep, tail_k=tail_k, tail_v=tail_v, sm_scale=sm_scale,
            causal=causal, out_dtype=out_dtype,
        )
    if use_qrope_transfer:
        out = fused_decode_attention_qrope_cpu(
            tucker_k, residual_k, tucker_v, residual_v, slot, cos, sin, query,
            query_position, n_rep, tail_k=tail_k, tail_v=tail_v, sm_scale=sm_scale,
            causal=causal,
        )
    else:
        out = fused_decode_attention_cpu(
            tucker_k, residual_k, tucker_v, residual_v, slot, cos, sin, query,
            query_position, n_rep, tail_k=tail_k, tail_v=tail_v, sm_scale=sm_scale, causal=causal,
        )
    return out if out_dtype == torch.float32 else out.to(out_dtype)
