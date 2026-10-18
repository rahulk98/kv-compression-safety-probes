"""Frozen tensor contract consumed by FlashJoLT decode paths.

Tucker and residual payloads use fp32 tensors and contiguous storage. K is
pre-RoPE and rotated during reconstruction; V is not rotated. Canonicalization
copies strided fitted factors once so kernels can rely on simple strides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

VALID_BITS = (0, 2, 4, 8)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _check_fp32(name: str, t: torch.Tensor) -> None:
    _check(
        t.dtype == torch.float32,
        f"{name}: expected dtype=torch.float32, got {t.dtype}",
    )




@dataclass(frozen=True)
class GroupLayout:
    """Maps compressed-KV-cache LAYERS to (group_idx, slot) pairs for the joint Tucker fit (the layer-grouping the allocator/decomposition operate on -- see `kv_compress/grouping/`)."""

    groups: list[list[int]]

    def layer_to_slot(self) -> dict[int, tuple[int, int]]:
        out: dict[int, tuple[int, int]] = {}
        for g_idx, group in enumerate(self.groups):
            for slot, layer_idx in enumerate(group):
                out[layer_idx] = (g_idx, slot)
        return out

    def validate(self) -> None:
        _check(len(self.groups) > 0, "GroupLayout: groups must be non-empty")
        seen: set[int] = set()
        for g_idx, group in enumerate(self.groups):
            _check(len(group) > 0, f"GroupLayout: group {g_idx} is empty")
            for layer_idx in group:
                _check(
                    layer_idx not in seen,
                    f"GroupLayout: layer {layer_idx} appears in more than one group",
                )
                seen.add(layer_idx)




@dataclass
class TuckerGroupFactors:
    """One joint Tucker fit shared by every layer in a group, for ONE of K or V (K and V each get their own `TuckerGroupFactors` instance -- ranks and shapes may differ, see item 5 / `ResidualPayload.bits`)."""

    core: torch.Tensor
    token_factor: torch.Tensor
    feature_factor: torch.Tensor

    @property
    def num_slots(self) -> int:
        return int(self.core.shape[0])

    @property
    def num_kv_heads(self) -> int:
        return int(self.core.shape[1])

    @property
    def token_rank(self) -> int:
        return int(self.core.shape[2])

    @property
    def feature_rank(self) -> int:
        return int(self.core.shape[3])

    @property
    def seq_len(self) -> int:
        return int(self.token_factor.shape[0])

    @property
    def head_dim(self) -> int:
        return int(self.feature_factor.shape[0])

    def validate(self) -> None:
        _check(self.core.dim() == 4, f"TuckerGroupFactors.core: expected 4D, got {self.core.dim()}D")
        _check(self.token_factor.dim() == 2, "TuckerGroupFactors.token_factor: expected 2D")
        _check(self.feature_factor.dim() == 2, "TuckerGroupFactors.feature_factor: expected 2D")
        _check_fp32("TuckerGroupFactors.core", self.core)
        _check_fp32("TuckerGroupFactors.token_factor", self.token_factor)
        _check_fp32("TuckerGroupFactors.feature_factor", self.feature_factor)
        _check(self.core.is_contiguous(), "TuckerGroupFactors.core must be contiguous")
        _check(self.token_factor.is_contiguous(), "TuckerGroupFactors.token_factor must be contiguous")
        _check(self.feature_factor.is_contiguous(), "TuckerGroupFactors.feature_factor must be contiguous")
        _check(
            self.core.shape[2] == self.token_factor.shape[1],
            f"token_rank mismatch: core={self.core.shape[2]} vs "
            f"token_factor={self.token_factor.shape[1]}",
        )
        _check(
            self.core.shape[3] == self.feature_factor.shape[1],
            f"feature_rank mismatch: core={self.core.shape[3]} vs "
            f"feature_factor={self.feature_factor.shape[1]}",
        )


def canonicalize_tucker_factors(
    core: torch.Tensor,
    token_factor: torch.Tensor,
    feature_factor: torch.Tensor,
) -> TuckerGroupFactors:
    """Build a contract-conformant `TuckerGroupFactors` from raw
    `Tucker4DApproximation` output (`.core`, `.factors[2]`, `.factors[3]`),
    which may be non-contiguous column-slice views (see module docstring).
    `.contiguous()` is a value-preserving copy-if-needed; it never changes
    numerics, only storage layout.
    """
    out = TuckerGroupFactors(
        core=core.to(torch.float32).contiguous(),
        token_factor=token_factor.to(torch.float32).contiguous(),
        feature_factor=feature_factor.to(torch.float32).contiguous(),
    )
    out.validate()
    return out


def adopt_canonical_tucker_factors(approx) -> TuckerGroupFactors:
    """Canonicalize factors and write them back into ``approx``.

    The exact-sizing pass releases oversized SVD parent storage, while
    preserving values and making repeated calls idempotent.
    """
    out = canonicalize_tucker_factors(approx.core, approx.factors[2], approx.factors[3])

    def _exactly_sized(t: torch.Tensor) -> torch.Tensor:
        if t.untyped_storage().nbytes() == t.nelement() * t.element_size():
            return t
        return t.clone()

    out.core = _exactly_sized(out.core)
    out.token_factor = _exactly_sized(out.token_factor)
    out.feature_factor = _exactly_sized(out.feature_factor)
    approx.core = out.core
    approx.factors[2] = out.token_factor
    approx.factors[3] = out.feature_factor
    return out




@dataclass
class ResidualPayload:
    """Per-layer K/V residual payload.

    For nonzero bits, ``packed`` is flat uint8 data, ``scale`` has shape
    ``(kv_heads, T)``, and ``rotation`` has shape ``(head_dim, head_dim)``.
    ``bits == 0`` requires empty packed and scale tensors.
    """

    packed: torch.Tensor
    scale: torch.Tensor
    bits: int
    numel: int
    rotation: torch.Tensor
    orig_shape: tuple[int, int, int]

    def validate(self) -> None:
        _check(self.bits in VALID_BITS, f"ResidualPayload.bits must be one of {VALID_BITS}, got {self.bits}")
        _check(len(self.orig_shape) == 3, f"ResidualPayload.orig_shape must be 3-tuple, got {self.orig_shape}")
        n_kv, T, head_dim = self.orig_shape
        _check_fp32("ResidualPayload.rotation", self.rotation)
        _check(
            tuple(self.rotation.shape) == (head_dim, head_dim),
            f"ResidualPayload.rotation shape {tuple(self.rotation.shape)} "
            f"!= (head_dim, head_dim)=({head_dim}, {head_dim})",
        )
        if self.bits == 0:
            # bits=0 requires empty payload tensors.
            _check(
                self.packed.numel() == 0,
                f"ResidualPayload.packed must be empty when bits=0, got numel={self.packed.numel()}",
            )
            _check(
                self.scale.numel() == 0,
                f"ResidualPayload.scale must be empty when bits=0, got numel={self.scale.numel()}",
            )
            return
        _check(self.packed.dtype == torch.uint8, f"ResidualPayload.packed must be uint8, got {self.packed.dtype}")
        _check(self.packed.dim() == 1, f"ResidualPayload.packed must be flat (1D), got {self.packed.dim()}D")
        _check_fp32("ResidualPayload.scale", self.scale)
        _check(
            tuple(self.scale.shape) == (n_kv, T),
            f"ResidualPayload.scale shape {tuple(self.scale.shape)} != (n_kv, T)=({n_kv}, {T})",
        )
        _check(
            self.numel == n_kv * T * head_dim,
            f"ResidualPayload.numel={self.numel} != prod(orig_shape)={n_kv * T * head_dim}",
        )
        expected_packed_len = (self.numel * self.bits + 7) // 8 if self.bits != 8 else self.numel
        _check(
            self.packed.numel() == expected_packed_len,
            f"ResidualPayload.packed length {self.packed.numel()} != expected {expected_packed_len} "
            f"for numel={self.numel} bits={self.bits}",
        )


def canonicalize_residual_payload(
    packed: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
    numel: int,
    rotation: torch.Tensor,
    orig_shape: tuple[int, int, int],
) -> ResidualPayload:
    """Build a contract-conformant `ResidualPayload` from raw
    `JLRotatedResidual`/`_JLPayload` output. Flattens `packed` to 1D (a
    value-preserving reshape/view -- see the `bits == 8` finding in
    `ResidualPayload`'s docstring for why this is necessary) and upcasts
    `scale`/`rotation` to fp32.
    """
    out = ResidualPayload(
        packed=packed.reshape(-1).to(torch.uint8),
        scale=scale.to(torch.float32),
        bits=int(bits),
        numel=int(numel),
        rotation=rotation.to(torch.float32),
        orig_shape=tuple(int(s) for s in orig_shape),
    )
    out.validate()
    return out




@dataclass
class CompressionSpec:
    """Separate K/V ranks and residual bit widths for one group (item 5).
    Purely descriptive/validating -- `TuckerGroupFactors` and
    `ResidualPayload` carry the actual tensors; this cross-checks them.
    """

    token_rank_k: int
    feature_rank_k: int
    token_rank_v: int
    feature_rank_v: int
    bits_k: int
    bits_v: int

    def validate(self) -> None:
        for name, val in (
            ("token_rank_k", self.token_rank_k),
            ("feature_rank_k", self.feature_rank_k),
            ("token_rank_v", self.token_rank_v),
            ("feature_rank_v", self.feature_rank_v),
        ):
            _check(val >= 1, f"CompressionSpec.{name} must be >= 1, got {val}")
        _check(self.bits_k in VALID_BITS, f"CompressionSpec.bits_k must be one of {VALID_BITS}, got {self.bits_k}")
        _check(self.bits_v in VALID_BITS, f"CompressionSpec.bits_v must be one of {VALID_BITS}, got {self.bits_v}")




@dataclass(frozen=True)
class HeadMap:
    """Query-head -> KV-head mapping, covering both GQA (Mistral: 32 query / 8 kv heads) and MHA (LLaMA-2-13B: 40 query / 40 kv heads)."""

    num_query_heads: int
    num_kv_heads: int

    def group_size(self) -> int:
        return self.num_query_heads // self.num_kv_heads

    def kv_head_for_query(self, q_head: int) -> int:
        return q_head // self.group_size()

    def query_heads_for_kv(self, kv_head: int) -> list[int]:
        n_rep = self.group_size()
        start = kv_head * n_rep
        return list(range(start, start + n_rep))

    def validate(self) -> None:
        _check(self.num_query_heads >= 1, "HeadMap.num_query_heads must be >= 1")
        _check(self.num_kv_heads >= 1, "HeadMap.num_kv_heads must be >= 1")
        _check(
            self.num_query_heads % self.num_kv_heads == 0,
            f"HeadMap: num_query_heads={self.num_query_heads} must be a multiple "
            f"of num_kv_heads={self.num_kv_heads}",
        )




@dataclass
class AttentionInputs:
    """Inputs for streamed attention over compressed K/V and a dense tail.

    ``query`` and ``cos/sin`` are fp32; query is post-RoPE while compressed K
    is pre-RoPE. Masks may be 2D shared or 3D per query head. Empty tails are
    valid and equivalent to no tail.
    """

    query: torch.Tensor
    query_positions: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    scale: float
    causal: bool
    attn_mask: torch.Tensor | None = None
    tail_k: torch.Tensor | None = None
    tail_v: torch.Tensor | None = None

    def validate(self, num_query_heads: int, num_kv_heads: int, head_dim: int, seq_len: int) -> None:
        _check(self.query.dim() == 3, f"AttentionInputs.query: expected 3D, got {self.query.dim()}D")
        _check_fp32("AttentionInputs.query", self.query)
        _check(
            self.query.shape[0] == num_query_heads,
            f"AttentionInputs.query num_query_heads={self.query.shape[0]} != "
            f"head_map.num_query_heads={num_query_heads}",
        )
        _check(
            self.query.shape[-1] == head_dim,
            f"AttentionInputs.query head_dim={self.query.shape[-1]} != {head_dim}",
        )
        _check(
            self.query_positions.dim() == 1,
            f"AttentionInputs.query_positions: expected 1D, got {self.query_positions.dim()}D",
        )
        _check(
            self.query_positions.dtype == torch.int64,
            f"AttentionInputs.query_positions: expected dtype=torch.int64, got {self.query_positions.dtype}",
        )
        _check(
            self.query_positions.shape[0] == self.query.shape[1],
            "AttentionInputs.query_positions length must match query's T_q",
        )
        _check_fp32("AttentionInputs.cos", self.cos)
        _check_fp32("AttentionInputs.sin", self.sin)
        _check(
            tuple(self.cos.shape) == (seq_len, head_dim),
            f"AttentionInputs.cos shape {tuple(self.cos.shape)} != (T, head_dim)=({seq_len}, {head_dim})",
        )
        _check(
            tuple(self.sin.shape) == (seq_len, head_dim),
            f"AttentionInputs.sin shape {tuple(self.sin.shape)} != (T, head_dim)=({seq_len}, {head_dim})",
        )

        devices = {
            "query": self.query.device,
            "query_positions": self.query_positions.device,
            "cos": self.cos.device,
            "sin": self.sin.device,
        }

        has_tail_k = self.tail_k is not None
        has_tail_v = self.tail_v is not None
        _check(has_tail_k == has_tail_v, "AttentionInputs: tail_k and tail_v must both be set or both be None")
        T_tail = 0
        if has_tail_k:
            _check(self.tail_k.dim() == 3, f"AttentionInputs.tail_k: expected 3D, got {self.tail_k.dim()}D")
            _check(self.tail_v.dim() == 3, f"AttentionInputs.tail_v: expected 3D, got {self.tail_v.dim()}D")
            _check_fp32("AttentionInputs.tail_k", self.tail_k)
            _check_fp32("AttentionInputs.tail_v", self.tail_v)
            _check(
                self.tail_k.shape == self.tail_v.shape,
                f"AttentionInputs: tail_k shape {tuple(self.tail_k.shape)} != tail_v shape {tuple(self.tail_v.shape)}",
            )
            _check(
                self.tail_k.shape[0] == num_kv_heads and self.tail_k.shape[-1] == head_dim,
                f"AttentionInputs.tail_k shape {tuple(self.tail_k.shape)} inconsistent with "
                f"(num_kv_heads, T_tail, head_dim)=({num_kv_heads}, ?, {head_dim})",
            )
            T_tail = int(self.tail_k.shape[1])
            devices["tail_k"] = self.tail_k.device
            devices["tail_v"] = self.tail_v.device

        if self.attn_mask is not None:
            _check(
                self.attn_mask.dim() in (2, 3),
                f"AttentionInputs.attn_mask: expected 2D (T_q, T_total) or 3D "
                f"(num_query_heads, T_q, T_total), got {self.attn_mask.dim()}D",
            )
            _check_fp32("AttentionInputs.attn_mask", self.attn_mask)
            T_total = seq_len + T_tail
            _check(
                self.attn_mask.shape[-2] == self.query.shape[1],
                f"AttentionInputs.attn_mask T_q={self.attn_mask.shape[-2]} != query's T_q={self.query.shape[1]}",
            )
            _check(
                self.attn_mask.shape[-1] == T_total,
                f"AttentionInputs.attn_mask T_total={self.attn_mask.shape[-1]} != "
                f"seq_len+T_tail={T_total}",
            )
            if self.attn_mask.dim() == 3:
                _check(
                    self.attn_mask.shape[0] == num_query_heads,
                    f"AttentionInputs.attn_mask per-head axis={self.attn_mask.shape[0]} != "
                    f"num_query_heads={num_query_heads}",
                )
            devices["attn_mask"] = self.attn_mask.device

        unique_devices = {str(d) for d in devices.values()}
        _check(
            len(unique_devices) == 1,
            f"AttentionInputs: all tensors must share one device, got {devices}",
        )




@dataclass
class FactorizedKVGroup:
    """Everything needed to reconstruct/attend for ONE Tucker group (K and V
    together): the two Tucker fits, the per-layer residuals, and the group
    layout. One instance per group produced by `_compress_prefill` /
    `TuckerJointFastCompressor.fit`.
    """

    layout: GroupLayout
    tucker_k: TuckerGroupFactors
    tucker_v: TuckerGroupFactors
    residual_k: dict[int, ResidualPayload | None]  # layer_idx -> payload or None (bits=0)
    residual_v: dict[int, ResidualPayload | None]
    head_map: HeadMap

    def validate(self) -> None:
        self.layout.validate()
        self.tucker_k.validate()
        self.tucker_v.validate()
        self.head_map.validate()
        _check(
            self.tucker_k.num_kv_heads == self.head_map.num_kv_heads,
            f"FactorizedKVGroup: tucker_k.num_kv_heads={self.tucker_k.num_kv_heads} != "
            f"head_map.num_kv_heads={self.head_map.num_kv_heads}",
        )
        _check(
            self.tucker_v.num_kv_heads == self.head_map.num_kv_heads,
            f"FactorizedKVGroup: tucker_v.num_kv_heads={self.tucker_v.num_kv_heads} != "
            f"head_map.num_kv_heads={self.head_map.num_kv_heads}",
        )
        _check(
            self.tucker_k.seq_len == self.tucker_v.seq_len,
            f"FactorizedKVGroup: K seq_len={self.tucker_k.seq_len} != V seq_len={self.tucker_v.seq_len}",
        )
        _check(
            self.tucker_k.head_dim == self.tucker_v.head_dim,
            f"FactorizedKVGroup: K head_dim={self.tucker_k.head_dim} != V head_dim={self.tucker_v.head_dim}",
        )
        _check(
            self.tucker_k.num_slots == self.tucker_v.num_slots,
            f"FactorizedKVGroup: K num_slots={self.tucker_k.num_slots} != "
            f"V num_slots={self.tucker_v.num_slots}",
        )
        # One instance carries one Tucker fit per side, so layout has one group.
        _check(
            len(self.layout.groups) == 1,
            f"FactorizedKVGroup: layout must describe exactly one group (got "
            f"{len(self.layout.groups)}) -- this instance holds a single Tucker "
            f"fit per side, so extra groups would have no matching factors",
        )
        n_slots = self.tucker_k.num_slots
        for layers in self.layout.groups:
            _check(
                len(layers) == n_slots,
                f"FactorizedKVGroup: layout has {len(layers)} layers but tucker core has "
                f"{n_slots} slots",
            )
        group_layers = self.layout.groups[0]
        for layer_idx in group_layers:
            _check(
                layer_idx in self.residual_k,
                f"FactorizedKVGroup: residual_k is missing an entry for layer {layer_idx} "
                "-- a missing key silently becomes None via dict.get() at decode time, "
                "dropping an intended residual rather than raising",
            )
            _check(
                layer_idx in self.residual_v,
                f"FactorizedKVGroup: residual_v is missing an entry for layer {layer_idx} "
                "-- a missing key silently becomes None via dict.get() at decode time, "
                "dropping an intended residual rather than raising",
            )
        for layer_idx, payload in self.residual_k.items():
            if payload is not None:
                payload.validate()
                _check(
                    payload.orig_shape == (self.tucker_k.num_kv_heads, self.tucker_k.seq_len, self.tucker_k.head_dim),
                    f"FactorizedKVGroup: residual_k[{layer_idx}].orig_shape "
                    f"{payload.orig_shape} inconsistent with tucker_k",
                )
        for layer_idx, payload in self.residual_v.items():
            if payload is not None:
                payload.validate()
                _check(
                    payload.orig_shape == (self.tucker_v.num_kv_heads, self.tucker_v.seq_len, self.tucker_v.head_dim),
                    f"FactorizedKVGroup: residual_v[{layer_idx}].orig_shape "
                    f"{payload.orig_shape} inconsistent with tucker_v",
                )
