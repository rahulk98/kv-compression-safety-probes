from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompressedCacheConfig:
    """Configuration for `CompressedKVCache`.

    Mirrors the relevant subset of `CompressionConfig` from the offline
    pipeline. The cache always uses pre-RoPE K capture and per-prompt
    allocation at first-decode (xKV-style runtime adaptation), so options
    that don't apply at runtime are not exposed.
    """

    target_compression_ratio: float | None = 4.0
    memory_budget_bytes: int | None = None

    decomposition: str = "tucker2_partial_lowrank"
    allocation: str = "tucker+jl_joint"
    residual: str = "jl_rotated"

    # Randomized-SVD cap; None selects the T-aware policy.
    q_cap: int | None = None

    grouping: str = "adaptive"
    alignment_source: str = "k_only"

    dtype_bytes: int = 2
    residual_seed: int = 0

    # Fused decode is opt-in; unsupported inputs use the dense path at preflight.
    use_fused_decode: bool = False

    # Storage dtype for fitted Tucker factors; identity factors are not cast.
    factor_storage_dtype: str = "bf16"

    def __post_init__(self) -> None:
        budgets = [self.target_compression_ratio, self.memory_budget_bytes]
        n_set = sum(b is not None for b in budgets)
        if n_set != 1:
            raise ValueError(
                "CompressedCacheConfig requires exactly one of "
                "(target_compression_ratio, memory_budget_bytes); "
                f"got {n_set} set."
            )
        if self.decomposition not in ("tucker2_partial", "tucker2_partial_lowrank"):
            raise ValueError(
                "decomposition must be one of "
                "('tucker2_partial', 'tucker2_partial_lowrank'); "
                f"got {self.decomposition!r}"
            )
        if self.q_cap is not None and int(self.q_cap) <= 0:
            raise ValueError(
                f"q_cap must be a positive int or None; got {self.q_cap!r}"
            )
        # Auto q_cap requires a ratio to resolve before prefill.
        if (
            self.decomposition == "tucker2_partial_lowrank"
            and self.memory_budget_bytes is not None
            and self.q_cap is None
        ):
            raise ValueError(
                "decomposition='tucker2_partial_lowrank' with a "
                "memory_budget_bytes target requires an explicit q_cap; the "
                "R-keyed auto-policy needs a target_compression_ratio. Either "
                "pass q_cap=32|64|... or use decomposition='tucker2_partial'."
            )
        if self.allocation != "tucker+jl_joint":
            raise ValueError(
                "v1 only supports allocation='tucker+jl_joint'; "
                f"got {self.allocation!r}"
            )
        if self.residual != "jl_rotated":
            raise ValueError(
                "v1 only supports residual='jl_rotated'; "
                f"got {self.residual!r}"
            )
        if self.alignment_source not in ("k_only", "v_only", "joint"):
            raise ValueError(
                f"alignment_source must be k_only/v_only/joint; got {self.alignment_source!r}"
            )
        if self.dtype_bytes not in (1, 2, 4):
            raise ValueError(f"dtype_bytes must be 1/2/4; got {self.dtype_bytes}")
        if not isinstance(self.use_fused_decode, bool):
            raise ValueError(
                f"use_fused_decode must be a bool; got {self.use_fused_decode!r}"
            )
        if self.factor_storage_dtype not in ("bf16", "fp32"):
            raise ValueError(
                "factor_storage_dtype must be one of ('bf16', 'fp32'); "
                f"got {self.factor_storage_dtype!r}"
            )

    def budget_mode(self) -> str:
        return "ratio" if self.target_compression_ratio is not None else "bytes"
