from __future__ import annotations

import time
from math import ceil

import tensorly as tl
import torch

from .. import instrumentation
from .tucker2_partial import Tucker2Partial
from .tucker4d import Tucker4DApproximation

tl.set_backend("pytorch")


class Tucker2PartialMode2LowRank(Tucker2Partial):
    """Tucker2Partial with randomized low-rank SVD on the token mode."""

    def __init__(
        self, q_cap: int = 32, niter: int = 1, mode3_gram: bool = False
    ) -> None:
        self.q_cap = int(q_cap)
        self.niter = int(niter)
        self.mode3_gram = bool(mode3_gram)
        self._mode2_cache: dict[int, torch.Tensor] = {}

    def _lowrank_mode2(
        self, X32: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Optional timing is host-side and unsynchronized.
        active = instrumentation.is_active()
        t0 = time.perf_counter() if active else None
        unf = tl.unfold(X32, 2)
        q = min(self.q_cap, X32.shape[2], unf.shape[1])
        U, S, _ = torch.svd_lowrank(unf, q=q, niter=self.niter)
        if active:
            instrumentation.record_mode2_call(
                q_cap=self.q_cap, q_actual=q, niter=self.niter,
                elapsed_s=time.perf_counter() - t0,
            )
        return U, S

    @staticmethod
    def _mode3_gram(X32: torch.Tensor) -> tuple[torch.Tensor, int]:
        unf = tl.unfold(X32, 3)
        gram = unf @ unf.transpose(0, 1)
        return gram, min(unf.shape)

    def _mode3_singvals(self, X32: torch.Tensor) -> torch.Tensor:
        gram, k = self._mode3_gram(X32)
        evals = torch.linalg.eigvalsh(gram.to(torch.float64))
        # Ascending eigenvalues -> descending singular values; trim to the
        # exact-SVD length min(unf.shape) so toy shapes with d_h > L*n_kv*T
        # match the svdvals contract.
        return evals.clamp(min=0.0).sqrt().flip(0)[:k].to(X32.dtype)

    def _mode3_factor(self, Z: torch.Tensor) -> torch.Tensor:
        gram, _ = self._mode3_gram(Z)
        evals, evecs = torch.linalg.eigh(gram.to(torch.float64))
        return evecs.flip(1).to(Z.dtype)

    def fit(
        self,
        X: torch.Tensor,
        ranks: tuple[int, ...],
    ) -> Tucker4DApproximation:
        orig_dtype = X.dtype
        X32 = X.to(torch.float32)

        factors: list[torch.Tensor] = [
            torch.eye(X32.shape[0], dtype=X32.dtype, device=X32.device),
            torch.eye(X32.shape[1], dtype=X32.dtype, device=X32.device),
        ]

        U2 = self._mode2_cache.pop(id(X), None)
        if U2 is None or U2.device != X32.device:
            U2, _ = self._lowrank_mode2(X32)
        U2_r = U2[:, : ranks[2]].to(device=X32.device, dtype=X32.dtype)
        factors.append(U2_r)
        Z = torch.tensordot(X32, U2_r, dims=([2], [0])).movedim(-1, 2)

        if self.mode3_gram:
            U3 = self._mode3_factor(Z)
        else:
            U3, _, _ = torch.linalg.svd(tl.unfold(Z, 3), full_matrices=False)
        U3_r = U3[:, : ranks[3]]
        factors.append(U3_r)
        core = torch.tensordot(Z, U3_r, dims=([3], [0])).movedim(-1, 3)

        return Tucker4DApproximation(
            core=core,
            factors=factors,
            ranks=tuple(int(r) for r in ranks),
            shape=tuple(X.shape),
            orig_dtype=orig_dtype,
        )

    def hosvd_singular_values(self, X: torch.Tensor) -> list[torch.Tensor]:
        X32 = X.to(torch.float32)
        norm = float(torch.linalg.vector_norm(X32))
        placeholder = torch.tensor([norm], dtype=X32.dtype, device=X32.device)

        U2, S2 = self._lowrank_mode2(X32)
        # Tail-mass accounting: see class docstring. ``norm`` is reused from
        # the placeholder computation above (same ||X||_F).
        residual2 = torch.clamp(
            norm ** 2 - (S2.to(torch.float32) ** 2).sum(),
            min=0.0,
        ).sqrt().reshape(1)
        S2_for_alloc = torch.cat(
            [S2, residual2.to(device=S2.device, dtype=S2.dtype)]
        )
        self._mode2_cache[id(X)] = U2

        if self.mode3_gram:
            S3 = self._mode3_singvals(X32)
        else:
            S3 = torch.linalg.svdvals(tl.unfold(X32, 3))
        return [placeholder, placeholder, S2_for_alloc, S3]

    def max_ranks(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        ranks = list(super().max_ranks(shape))
        ranks[2] = min(ranks[2], self.q_cap)
        return tuple(ranks)


# T-aware q_cap schedule constants. See
# ``docs/superpowers/specs/2026-06-08-t-aware-qcap-policy-design.md``.
QCAP_ALPHA_DENOM = 32  # q_alpha = ceil(T / 32); pins q_cap=32 at the validated T=1024 point
QCAP_HARD_MAX = 512    # speedup guard; no binding through T=16384


def _q_min(target_ratio: float) -> int:
    """Validated small-T floor: 32 through R=4, 64 from R=5 (the LLaMA cliff)."""
    return 32 if float(target_ratio) <= 4.0 else 64


def resolve_q_cap(
    target_ratio: float,
    override: int | None = None,
    *,
    seq_len: int | None = None,
) -> int:
    """Resolve the mode-2 randomized-SVD cap from the target ratio and context length."""
    if override is not None:
        return int(override)
    q_min = _q_min(target_ratio)
    if seq_len is None:
        return q_min
    q_alpha = ceil(int(seq_len) / QCAP_ALPHA_DENOM)
    return min(max(q_min, q_alpha), QCAP_HARD_MAX)
