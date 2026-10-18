"""Deprecated full 4D Tucker decomposition retained for comparisons.

Use ``tucker2_partial`` for production compression.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import prod

import tensorly as tl
import torch

from ..allocation.memory import mem_tucker4d_compressed
from .base import Approximation, Decomposition

tl.set_backend("pytorch")


def _st_hosvd(
    X: torch.Tensor,
    ranks: tuple[int, ...],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Sequentially Truncated HOSVD (Vannieuwenhoven et al., 2012)."""
    Z = X
    factors: list[torch.Tensor] = []
    for mode in range(X.ndim):
        unf = tl.unfold(Z, mode)                   # (Z.shape[mode], ∏ other dims)
        U, _, _ = torch.linalg.svd(unf, full_matrices=False)
        U_r = U[:, :ranks[mode]]                   # (dim_mode, r_mode)
        factors.append(U_r)
        # Restore the contracted mode after tensordot places it last.
        Z = torch.tensordot(Z, U_r, dims=([mode], [0])).movedim(-1, mode)
    return Z, factors  # Z is the core G


@dataclass
class Tucker4DApproximation(Approximation):
    """Fitted 4D Tucker approximation from `Tucker2Partial`/ `Tucker2PartialMode2LowRank.fit`: modes 0 (layer-group) and 1 (head) are always identity (`factors[0]`/`factors[1]`, full-rank `torch.eye`), modes 2 (token) and 3 (feature/head-dim) are the actually-fitted low-rank factors (`factors[2]`/`factors[3]`)."""

    core: torch.Tensor
    factors: list[torch.Tensor]
    ranks: tuple[int, ...]
    shape: tuple[int, ...]
    orig_dtype: torch.dtype
    # Set by quantize_payload_; None preserves dtype_bytes accounting.
    _quant_bits: int | None = None
    _quant_bytes: int | None = None

    def cast_storage_bf16_(self) -> None:
        """In-place storage-dtype cast: `core` and the fitted (non-identity) `factors[2]`/`factors[3]` -> `torch.bfloat16`."""
        self.core = self.core.to(torch.bfloat16)
        self.factors[2] = self.factors[2].to(torch.bfloat16)
        self.factors[3] = self.factors[3].to(torch.bfloat16)

    def reconstruct(self) -> torch.Tensor:
        # Reconstruct in fp32 regardless of factor storage dtype.
        core32 = self.core.to(torch.float32)
        factors32 = [f.to(torch.float32) for f in self.factors]
        X_hat = tl.tucker_to_tensor((core32, factors32))
        return X_hat.to(self.orig_dtype)

    def payload_tensors(self) -> list[torch.Tensor]:
        return [self.core, *self.factors]

    def footprint_bytes(self, dtype_bytes: int = 2) -> int:
        if self._quant_bits is not None:
            return int(self._quant_bytes)
        return mem_tucker4d_compressed(self.ranks, self.shape, dtype_bytes)


class Tucker4D(Decomposition):
    """DEPRECATED — use Tucker2Partial instead. See module docstring."""

    def fit(
        self,
        X: torch.Tensor,
        ranks: tuple[int, ...],
    ) -> Tucker4DApproximation:
        warnings.warn(
            "Tucker4D is deprecated and slow; use tucker2_partial instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        orig_dtype = X.dtype
        X32 = X.to(torch.float32)
        core, factors = _st_hosvd(X32, ranks)
        return Tucker4DApproximation(
            core=core,
            factors=list(factors),
            ranks=tuple(int(r) for r in ranks),
            shape=tuple(X.shape),
            orig_dtype=orig_dtype,
        )

    def hosvd_singular_values(self, X: torch.Tensor) -> list[torch.Tensor]:
        # Allocators use original-tensor HOSVD spectra as a rank proxy.
        X32 = X.to(torch.float32)
        sigmas: list[torch.Tensor] = []
        for mode in range(X32.ndim):
            unf = tl.unfold(X32, mode)
            sigmas.append(torch.linalg.svdvals(unf))
        return sigmas

    def max_ranks(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        total = prod(shape)
        return tuple(min(shape[m], total // shape[m]) for m in range(len(shape)))
