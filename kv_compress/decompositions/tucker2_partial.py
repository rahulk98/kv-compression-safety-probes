from __future__ import annotations

from math import prod

import tensorly as tl
import torch

from ..allocation.memory import mem_tucker4d_compressed
from .base import Decomposition
from .tucker4d import Tucker4DApproximation

tl.set_backend("pytorch")


class Tucker2Partial(Decomposition):
    """4D Tucker decomposition that only fits factor matrices for modes 2 and 3."""

    _ACTIVE = (2, 3)

    def fit(
        self,
        X: torch.Tensor,
        ranks: tuple[int, ...],
    ) -> Tucker4DApproximation:
        orig_dtype = X.dtype
        X32 = X.to(torch.float32)

        Z = X32
        factors: list[torch.Tensor] = []
        for mode in range(X32.ndim):
            if mode in self._ACTIVE:
                unf = tl.unfold(Z, mode)
                U, _, _ = torch.linalg.svd(unf, full_matrices=False)
                U_r = U[:, : ranks[mode]]
                factors.append(U_r)
                Z = torch.tensordot(Z, U_r, dims=([mode], [0])).movedim(-1, mode)
            else:
                # Identity factor on this mode — full rank, no rotation.
                d = X32.shape[mode]
                factors.append(torch.eye(d, dtype=X32.dtype, device=X32.device))

        return Tucker4DApproximation(
            core=Z,
            factors=factors,
            ranks=tuple(int(r) for r in ranks),
            shape=tuple(X.shape),
            orig_dtype=orig_dtype,
        )

    def hosvd_singular_values(self, X: torch.Tensor) -> list[torch.Tensor]:
        # Active modes get their actual unfolding singular values. Pinned modes
        # return a single-element [||X||_F] so that the allocator's per-group
        # Frobenius norm — computed as sqrt((sigs[0]**2).sum()) via the mode-k
        # Frobenius property — stays correct without extra SVD work.
        X32 = X.to(torch.float32)
        norm = float(torch.linalg.vector_norm(X32))
        placeholder = torch.tensor([norm], dtype=X32.dtype, device=X32.device)
        sigmas: list[torch.Tensor] = []
        for mode in range(X32.ndim):
            if mode in self._ACTIVE:
                unf = tl.unfold(X32, mode)
                sigmas.append(torch.linalg.svdvals(unf))
            else:
                sigmas.append(placeholder)
        return sigmas

    def max_ranks(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        total = prod(shape)
        return tuple(min(shape[m], total // shape[m]) for m in range(len(shape)))
