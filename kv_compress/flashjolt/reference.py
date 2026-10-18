"""Pure-PyTorch FlashJoLT references and block-level attention primitives.

The dense reconstruction path is a test oracle. Attention streams compressed
prefill blocks and the post-RoPE decode tail through one fp32 softmax state.
The independent block implementations support cross-checks against Triton.
"""

from __future__ import annotations

import torch

from ..integration.pre_rope_capture import apply_rope_to_k, rotate_half
from ..residuals._quant import unpack
from .layout import (
    AttentionInputs,
    FactorizedKVGroup,
    HeadMap,
    ResidualPayload,
    TuckerGroupFactors,
)



def unpack_block(
    packed: torch.Tensor, bits: int, total_numel: int, start: int, end: int
) -> torch.Tensor:
    """Unpack only elements `[start, end)` of a bit-packed uint8 tensor, touching only the bytes that contain them, instead of the full `total_numel`-length codes tensor `unpack()` returns."""
    if bits == 8:
        return packed[start:end].to(torch.int8)
    vpb = 8 // bits
    byte_start = start // vpb
    byte_end = -(-end // vpb)  # ceil div
    sub_packed = packed[byte_start:byte_end]
    sub_numel = min(total_numel, byte_end * vpb) - byte_start * vpb
    codes = unpack(sub_packed, bits, sub_numel)
    local_start = start - byte_start * vpb
    local_end = end - byte_start * vpb
    return codes[local_start:local_end]




def reconstruct_dense(
    tucker: TuckerGroupFactors,
    residual: ResidualPayload | None,
    slot: int,
) -> torch.Tensor:
    """Dense (kv_heads, T, head_dim) reconstruction: Tucker low-rank plus JL residual (if present), fp32."""
    core_slot = tucker.core[slot]  # (kv_heads, token_rank, feature_rank)
    low = torch.einsum(
        "tr,hrf,df->htd", tucker.token_factor, core_slot, tucker.feature_factor
    )
    if residual is not None and residual.bits > 0:
        n_kv, T, head_dim = residual.orig_shape
        codes = (
            unpack(residual.packed, residual.bits, residual.numel)
            .to(torch.float32)
            .reshape(n_kv, T, head_dim)
        )
        scale = residual.scale.unsqueeze(-1)  # (n_kv, T, 1)
        rotated = codes * scale
        res = rotated @ residual.rotation  # (n_kv, T, head_dim) -- matches JLRotatedResidual.decode
        low = low + res
    return low


def reconstruct_dense_k_post_rope(
    tucker_k: TuckerGroupFactors,
    residual_k: ResidualPayload | None,
    slot: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """`reconstruct_dense` for K, with RoPE re-applied via the SAME
    production `apply_rope_to_k` used everywhere else in this module --
    reused, not reimplemented, so RoPE parity holds by construction.
    """
    k_pre = reconstruct_dense(tucker_k, residual_k, slot)  # (n_kv, T, head_dim)
    k_post = apply_rope_to_k(
        k_pre.unsqueeze(0), cos.unsqueeze(0), sin.unsqueeze(0), unsqueeze_dim=1
    )
    return k_post.squeeze(0)




def _reconstruct_block(
    tucker: TuckerGroupFactors,
    residual: ResidualPayload | None,
    slot: int,
    kv_head: int,
    t_start: int,
    t_end: int,
) -> torch.Tensor:
    """Reconstruct Tucker low-rank + residual for ONE kv head, tokens `[t_start, t_end)`."""
    core_slot_head = tucker.core[slot, kv_head]  # (token_rank, feature_rank)
    U_T_blk = tucker.token_factor[t_start:t_end]  # (blk, token_rank)
    low = U_T_blk @ core_slot_head @ tucker.feature_factor.T  # (blk, head_dim)
    if residual is not None and residual.bits > 0:
        n_kv, T, head_dim = residual.orig_shape
        block_len = t_end - t_start
        elem_start = kv_head * T * head_dim + t_start * head_dim
        elem_end = kv_head * T * head_dim + t_end * head_dim
        codes = unpack_block(
            residual.packed, residual.bits, residual.numel, elem_start, elem_end
        )
        codes = codes.to(torch.float32).reshape(block_len, head_dim)
        scale = residual.scale[kv_head, t_start:t_end].unsqueeze(-1)  # (blk, 1)
        rotated = codes * scale
        res = rotated @ residual.rotation  # (blk, head_dim)
        low = low + res
    return low


def _assert_softmax_denominator_invariants(l: torch.Tensor, acc: torch.Tensor) -> None:
    """Invariant check for the online-softmax running denominator `l` and accumulator `acc`, run immediately before the final `acc / l` division in `attention_on_factors`."""
    assert torch.all(torch.isfinite(l)), (
        "online-softmax denominator l contains NaN/inf -- this should be "
        "structurally unreachable (l is a sum of exp(...) terms, which are "
        "always finite for finite scores)"
    )
    assert torch.all(l >= 0), (
        "online-softmax denominator l is negative -- l is a running sum of "
        "exp(...) terms and must be non-negative"
    )
    assert not torch.any((l == 0) & torch.any(acc != 0, dim=-1)), (
        "online-softmax invariant violated: l == 0 (no mass accumulated) but "
        "acc != 0 for the same row -- l and acc share the same non-negative "
        "exponential weights, so this state should be unreachable"
    )


def _rope_block(
    k_blk: torch.Tensor, cos_blk: torch.Tensor, sin_blk: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE to a `(block_len, head_dim)` pre-RoPE K slice via the
    production `apply_rope_to_k`, reshaping to/from its `(B, n_kv, T,
    head_dim)` contract. Reused rather than reimplemented (see module
    docstring) so the RoPE parity requirement holds by construction.
    """
    k_b = k_blk.unsqueeze(0).unsqueeze(0)  # (1, 1, blk, head_dim)
    cos_b = cos_blk.unsqueeze(0)
    sin_b = sin_blk.unsqueeze(0)
    return apply_rope_to_k(k_b, cos_b, sin_b, unsqueeze_dim=1).squeeze(0).squeeze(0)




def attention_on_factors(
    group: FactorizedKVGroup,
    layer_idx: int,
    attn: AttentionInputs,
    *,
    block_size: int = 128,
) -> torch.Tensor:
    """Attention output `(num_query_heads, T_q, head_dim)` computed without ever materializing dense K or V."""
    group.validate()
    g_idx, slot = group.layout.layer_to_slot()[layer_idx]
    assert g_idx == 0, (
        "unreachable: FactorizedKVGroup.validate() enforces exactly one layout group, "
        "so layer_to_slot() can never return a non-zero group index here"
    )
    residual_k = group.residual_k.get(layer_idx)
    residual_v = group.residual_v.get(layer_idx)
    head_map: HeadMap = group.head_map
    attn.validate(head_map.num_query_heads, head_map.num_kv_heads, group.tucker_k.head_dim, group.tucker_k.seq_len)

    T = group.tucker_k.seq_len
    head_dim = group.tucker_k.head_dim
    T_q = attn.query.shape[1]
    device = attn.query.device
    n_q = head_map.num_query_heads

    out = torch.empty(n_q, T_q, head_dim, dtype=torch.float32, device=device)

    for kv_head in range(head_map.num_kv_heads):
        q_heads = head_map.query_heads_for_kv(kv_head)
        Q = attn.query[q_heads]  # (n_rep, T_q, head_dim)
        n_rep = Q.shape[0]

        m = torch.full((n_rep, T_q), float("-inf"), dtype=torch.float32, device=device)
        l = torch.zeros((n_rep, T_q), dtype=torch.float32, device=device)
        acc = torch.zeros((n_rep, T_q, head_dim), dtype=torch.float32, device=device)

        def _step(K_blk: torch.Tensor, V_blk: torch.Tensor, key_positions: torch.Tensor) -> None:
            nonlocal m, l, acc
            scores = torch.einsum("rtd,sd->rts", Q, K_blk) * attn.scale  # (n_rep, T_q, blk)
            if attn.causal:
                allowed = attn.query_positions.unsqueeze(-1) >= key_positions.unsqueeze(0)  # (T_q, blk)
                scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
            if attn.attn_mask is not None:
                # Slice per-query-head masks to this KV head's query rows.
                if attn.attn_mask.dim() == 2:
                    mask_slice = attn.attn_mask[:, key_positions].unsqueeze(0)  # (1, T_q, blk)
                else:
                    mask_slice = attn.attn_mask[q_heads][..., key_positions]  # (n_rep, T_q, blk)
                scores = scores + mask_slice
            block_max = scores.amax(dim=-1)  # (n_rep, T_q), may be -inf if fully masked
            new_max = torch.maximum(m, block_max)
            # Use a finite exponent base for empty or infinite maxima.
            new_max_safe = torch.where(torch.isinf(new_max), torch.zeros_like(new_max), new_max)
            correction = torch.exp(m - new_max_safe)
            p = torch.exp(scores - new_max_safe.unsqueeze(-1))
            l = l * correction + p.sum(dim=-1)
            acc = acc * correction.unsqueeze(-1) + torch.einsum("rts,sd->rtd", p, V_blk)
            # Preserve the true running maximum, including -inf.
            m = new_max

        for t0 in range(0, T, block_size):
            t1 = min(t0 + block_size, T)
            k_blk_pre = _reconstruct_block(group.tucker_k, residual_k, slot, kv_head, t0, t1)
            k_blk = _rope_block(k_blk_pre, attn.cos[t0:t1], attn.sin[t0:t1])
            v_blk = _reconstruct_block(group.tucker_v, residual_v, slot, kv_head, t0, t1)
            key_positions = torch.arange(t0, t1, device=device)
            _step(k_blk, v_blk, key_positions)

        # Empty tails are equivalent to no tail and skip the reduction.
        if attn.tail_k is not None and attn.tail_k.shape[1] > 0:
            T_tail = attn.tail_k.shape[1]
            tail_k_h = attn.tail_k[kv_head]  # (T_tail, head_dim), already post-RoPE
            tail_v_h = attn.tail_v[kv_head]
            key_positions = torch.arange(T, T + T_tail, device=device)
            _step(tail_k_h, tail_v_h, key_positions)

        # Fully masked rows return zeros; validate the denominator first.
        _assert_softmax_denominator_invariants(l, acc)
        l_safe = torch.where(l == 0, torch.ones_like(l), l)
        out[q_heads] = acc / l_safe.unsqueeze(-1)

    return out




def attention_on_factors_qrope(
    group: FactorizedKVGroup,
    layer_idx: int,
    attn: AttentionInputs,
    *,
    block_size: int = 128,
) -> torch.Tensor:
    """[STAGE 1 / 2] Q-side RoPE-transfer variant of `attention_on_factors`."""
    group.validate()
    g_idx, slot = group.layout.layer_to_slot()[layer_idx]
    assert g_idx == 0, (
        "unreachable: FactorizedKVGroup.validate() enforces exactly one layout group, "
        "so layer_to_slot() can never return a non-zero group index here"
    )
    residual_k = group.residual_k.get(layer_idx)
    residual_v = group.residual_v.get(layer_idx)
    head_map: HeadMap = group.head_map
    attn.validate(head_map.num_query_heads, head_map.num_kv_heads, group.tucker_k.head_dim, group.tucker_k.seq_len)

    T = group.tucker_k.seq_len
    head_dim = group.tucker_k.head_dim
    T_q = attn.query.shape[1]
    device = attn.query.device
    n_q = head_map.num_query_heads

    out = torch.empty(n_q, T_q, head_dim, dtype=torch.float32, device=device)

    for kv_head in range(head_map.num_kv_heads):
        q_heads = head_map.query_heads_for_kv(kv_head)
        Q = attn.query[q_heads]  # (n_rep, T_q, head_dim), POST-RoPE
        n_rep = Q.shape[0]

        m = torch.full((n_rep, T_q), float("-inf"), dtype=torch.float32, device=device)
        l = torch.zeros((n_rep, T_q), dtype=torch.float32, device=device)
        acc = torch.zeros((n_rep, T_q, head_dim), dtype=torch.float32, device=device)

        def _step(scores_blk: torch.Tensor, V_blk: torch.Tensor, key_positions: torch.Tensor) -> None:
            """qrope mirror of `attention_on_factors._step` -- keep the two in lockstep (stage-1 spec, consequence 6)."""
            nonlocal m, l, acc
            if attn.causal:
                allowed = attn.query_positions.unsqueeze(-1) >= key_positions.unsqueeze(0)  # (T_q, blk)
                scores_blk = scores_blk.masked_fill(~allowed.unsqueeze(0), float("-inf"))
            if attn.attn_mask is not None:
                # Slice per-head masks to this KV head's query rows.
                if attn.attn_mask.dim() == 2:
                    mask_slice = attn.attn_mask[:, key_positions].unsqueeze(0)  # (1, T_q, blk)
                else:
                    mask_slice = attn.attn_mask[q_heads][..., key_positions]  # (n_rep, T_q, blk)
                scores_blk = scores_blk + mask_slice
            block_max = scores_blk.amax(dim=-1)  # (n_rep, T_q), may be -inf if fully masked
            new_max = torch.maximum(m, block_max)
            # Use a finite exponent base without changing the true maximum.
            new_max_safe = torch.where(torch.isinf(new_max), torch.zeros_like(new_max), new_max)
            correction = torch.exp(m - new_max_safe)
            p = torch.exp(scores_blk - new_max_safe.unsqueeze(-1))
            l = l * correction + p.sum(dim=-1)
            acc = acc * correction.unsqueeze(-1) + torch.einsum("rts,sd->rtd", p, V_blk)
            # Preserve the true running maximum, including -inf.
            m = new_max

        G_k = group.tucker_k.core[slot, kv_head]  # (token_rank, feature_rank)
        U_D_k = group.tucker_k.feature_factor  # (head_dim, feature_rank)
        U_T = group.tucker_k.token_factor  # (T, token_rank)
        R_kT = residual_k.rotation.T if (residual_k is not None and residual_k.bits > 0) else None

        for t0 in range(0, T, block_size):
            t1 = min(t0 + block_size, T)
            block_len = t1 - t0
            cos_blk = attn.cos[t0:t1]  # (blk, head_dim) -- absolute positions t0..t1
            sin_blk = attn.sin[t0:t1]
            U_T_blk = U_T[t0:t1]  # (blk, token_rank)

            # Contract residual codes directly without a dense residual tile.
            scaled_k = None
            if residual_k is not None and residual_k.bits > 0:
                n_kv, T_k, _d = residual_k.orig_shape
                elem_start = kv_head * T_k * head_dim + t0 * head_dim
                elem_end = kv_head * T_k * head_dim + t1 * head_dim
                codes = unpack_block(
                    residual_k.packed, residual_k.bits, residual_k.numel, elem_start, elem_end
                )
                codes = codes.to(torch.float32).reshape(block_len, head_dim)
                scaled_k = codes * residual_k.scale[kv_head, t0:t1].unsqueeze(-1)  # (blk, head_dim)

            # Build inverse-RoPE query tiles from the shared query vector.
            scores = torch.empty(n_rep, T_q, block_len, dtype=torch.float32, device=device)
            for r in range(n_rep):
                for t in range(T_q):
                    q_vec = Q[r, t]  # (head_dim,), raw POST-RoPE q
                    w_blk = q_vec * cos_blk - rotate_half(q_vec) * sin_blk  # (blk, head_dim) = RoPE^T q per n
                    # Low-rank score (consequence 1): ((w_n @ U_D) @ G^T) . U_T[n,:]
                    mid = w_blk @ U_D_k  # (blk, feature_rank)
                    mid = mid @ G_k.T  # (blk, token_rank)
                    row = (mid * U_T_blk).sum(-1)  # (blk,)
                    # Residual score (consequence 2): (w_n @ R_K^T) . (codes_k[n] * scale_k[n])
                    if scaled_k is not None:
                        row = row + ((w_blk @ R_kT) * scaled_k).sum(-1)
                    scores[r, t] = row * attn.scale

            v_blk = _reconstruct_block(group.tucker_v, residual_v, slot, kv_head, t0, t1)
            key_positions = torch.arange(t0, t1, device=device)
            _step(scores, v_blk, key_positions)

        # Dense tail is already post-RoPE and uses the original query.
        if attn.tail_k is not None and attn.tail_k.shape[1] > 0:
            T_tail = attn.tail_k.shape[1]
            tail_k_h = attn.tail_k[kv_head]  # (T_tail, head_dim), already post-RoPE
            tail_v_h = attn.tail_v[kv_head]
            key_positions = torch.arange(T, T + T_tail, device=device)
            tail_scores = torch.einsum("rtd,sd->rts", Q, tail_k_h) * attn.scale
            _step(tail_scores, tail_v_h, key_positions)

        # Fully masked rows return zeros after denominator validation.
        _assert_softmax_denominator_invariants(l, acc)
        l_safe = torch.where(l == 0, torch.ones_like(l), l)
        out[q_heads] = acc / l_safe.unsqueeze(-1)

    return out
