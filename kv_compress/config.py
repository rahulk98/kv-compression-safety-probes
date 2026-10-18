from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompressionConfig:
    target_compression_ratio: float | None = None
    memory_budget_bytes: int | None = None
    max_reconstruction_error: float | None = None

    grouping: str = "adaptive"
    decomposition: str = "tucker2_partial"
    # Reserved for the loss registry; validated to the single supported value.
    loss: str = "factor_space"
    allocation: str = "lagrangian_tucker2"

    k_target: str = "auto"

    # Residual encoder selection; joint allocation may choose per-group bits.
    residual: str = "none"
    residual_seed: int = 0

    # Policy for selecting residual-bearing groups.
    residual_bits_policy: str = "fixed"
    residual_byte_fraction: float = 0.05

    alignment_source: str = "k_only"
    dtype_bytes: int = 2

    calibration_cache: str | None = None
    model_name: str | None = None

    def __post_init__(self) -> None:
        budgets = [
            self.target_compression_ratio,
            self.memory_budget_bytes,
            self.max_reconstruction_error,
        ]
        n_set = sum(b is not None for b in budgets)
        if n_set != 1:
            raise ValueError(
                "CompressionConfig requires exactly one of "
                "(target_compression_ratio, memory_budget_bytes, max_reconstruction_error); "
                f"got {n_set} set."
            )
        if self.k_target not in ("auto", "pre_rope", "post_rope", "both"):
            raise ValueError(f"k_target must be one of auto/pre_rope/post_rope/both; got {self.k_target!r}")
        if self.alignment_source not in ("k_only", "v_only", "joint"):
            raise ValueError(f"alignment_source must be k_only/v_only/joint; got {self.alignment_source!r}")
        if self.allocation not in (
            "lagrangian",
            "lagrangian_tucker2",
            "lagrangian_token",
            "tucker2_grid",
            "tucker+jl_joint",
        ):
            raise ValueError(f"unknown allocation strategy: {self.allocation!r}")
        # Validate the reserved loss identifier.
        if self.loss != "factor_space":
            raise ValueError(f"loss must be factor_space (reserved, unused); got {self.loss!r}")
        # Keep the experimental projection encoder out of the public config.
        if self.residual not in ("none", "jl_rotated"):
            if self.residual == "jl_projection":
                raise ValueError(
                    "residual='jl_projection' is registered in residuals.REGISTRY but is "
                    "intentionally NOT selectable through CompressionConfig. "
                    "JLProjectionResidual is a deliberately retained EXPERIMENTAL encoder: "
                    "the true dimension-reducing Johnson-Lindenstrauss variant, kept as the "
                    "negative-result artifact behind results.md 11.7, where it beats the "
                    "shipping rotation encoder only in the 2 bits/elem cell and loses 24/24 "
                    "paired cells at 3, 4 and 8 bits/elem. The shipping residual is "
                    "'jl_rotated' (rename-not-adopt decision, 2026-07-17). To use the "
                    "projection in an experiment, instantiate "
                    "residuals.REGISTRY['jl_projection'] directly; do not widen this check."
                )
            raise ValueError(
                "residual must be none or jl_rotated ('jl_projection' is registered but "
                f"deliberately not selectable here); got {self.residual!r}"
            )
        if self.residual_bits_policy not in ("fixed", "adaptive", "auto"):
            raise ValueError(
                f"residual_bits_policy must be fixed, adaptive, or auto; got {self.residual_bits_policy!r}"
            )
        if self.allocation == "tucker+jl_joint":
            if self.residual != "jl_rotated":
                raise ValueError(
                    "tucker+jl_joint allocator requires residual='jl_rotated' "
                    f"(got {self.residual!r}); the joint allocator co-decides "
                    "residual bits per group and needs the JL encoder."
                )
            if self.residual_bits_policy != "fixed":
                raise ValueError(
                    "tucker+jl_joint allocator subsumes residual_bits_policy; "
                    f"set residual_bits_policy='fixed' (got {self.residual_bits_policy!r})."
                )
        if not (0.0 < self.residual_byte_fraction <= 1.0):
            raise ValueError(
                f"residual_byte_fraction must be in (0, 1]; got {self.residual_byte_fraction}"
            )
        if self.dtype_bytes not in (1, 2, 4):
            raise ValueError(f"dtype_bytes must be 1/2/4; got {self.dtype_bytes}")

    def budget_mode(self) -> str:
        if self.target_compression_ratio is not None:
            return "ratio"
        if self.memory_budget_bytes is not None:
            return "bytes"
        return "error"
