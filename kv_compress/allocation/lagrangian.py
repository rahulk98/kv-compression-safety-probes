from __future__ import annotations

import torch

from ..decompositions.base import Decomposition
from .base import BudgetSpec, RankAllocator
from .memory import mem_original, mem_tucker4d_compressed


def _bytes_for_rank(
    ranks: list[int],
    mode: int,
    r_new: int,
    dims: tuple[int, ...],
    dtype_bytes: int,
) -> int:
    trial = list(ranks)
    trial[mode] = r_new
    return mem_tucker4d_compressed(tuple(trial), dims, dtype_bytes)


def _greedy_ranks_for_group(
    lam: float,
    sigmas: list[torch.Tensor],
    dims: tuple[int, ...],
    max_ranks: tuple[int, ...],
    active_modes: tuple[int, ...],
    dtype_bytes: int,
) -> list[int]:
    """Per-group rank greedy. Accept r→r+1 at mode m iff σ²[r] > lam·Δbytes_tucker.

    Pinned modes (not in active_modes) start at max rank — i.e. uncompressed
    along that axis. Active modes start at 1 and grow until no marginal move is
    profitable at the current λ.
    """
    n_modes = len(sigmas)
    ranks = [
        int(max_ranks[m]) if m not in active_modes else 1
        for m in range(n_modes)
    ]
    improved = True
    while improved:
        improved = False
        for m in active_modes:
            if ranks[m] >= max_ranks[m]:
                continue
            sigma = sigmas[m]
            idx = ranks[m]
            if idx >= sigma.numel():
                continue
            delta_err = float(sigma[idx] ** 2)
            cur_b = mem_tucker4d_compressed(tuple(ranks), dims, dtype_bytes)
            new_b = _bytes_for_rank(ranks, m, ranks[m] + 1, dims, dtype_bytes)
            delta_bytes = new_b - cur_b
            if delta_bytes <= 0:
                continue
            if delta_err > lam * delta_bytes:
                ranks[m] += 1
                improved = True
    return ranks


class LagrangianRankAllocator(RankAllocator):
    def __init__(
        self,
        max_outer: int = 30,
        tol_frac: float = 0.005,
        active_modes: tuple[int, ...] | None = None,
    ):
        self.max_outer = max_outer
        self.tol_frac = tol_frac
        self.active_modes = active_modes

    def _solve_inner(
        self,
        lam: float,
        per_group_sigmas: list[list[torch.Tensor]],
        per_group_dims: list[tuple[int, ...]],
        per_group_max: list[tuple[int, ...]],
        dtype_bytes: int,
    ) -> list[list[int]]:
        all_ranks: list[list[int]] = []
        for g in range(len(per_group_sigmas)):
            n_modes = len(per_group_sigmas[g])
            active = tuple(range(n_modes)) if self.active_modes is None else self.active_modes
            ranks = _greedy_ranks_for_group(
                lam,
                per_group_sigmas[g],
                per_group_dims[g],
                per_group_max[g],
                active,
                dtype_bytes,
            )
            all_ranks.append(ranks)
        return all_ranks

    def _run(
        self,
        groups: list[torch.Tensor],
        budget: BudgetSpec,
        decomposition: Decomposition,
    ) -> list[tuple[int, ...]]:
        dtype_bytes = budget.dtype_bytes

        per_group_dims = [tuple(int(s) for s in g.shape) for g in groups]
        per_group_max = [decomposition.max_ranks(d) for d in per_group_dims]
        per_group_sigmas = [
            [s.detach().cpu() for s in decomposition.hosvd_singular_values(g)]
            for g in groups
        ]
        # Normalize by the Frobenius norm for relative-error allocation.
        per_group_norms = [float((sigs[0] ** 2).sum()) ** 0.5 for sigs in per_group_sigmas]
        per_group_sigmas = [
            [s / max(norm, 1e-12) for s in sigs]
            for sigs, norm in zip(per_group_sigmas, per_group_norms)
        ]
        total_original_bytes = sum(mem_original(d, dtype_bytes) for d in per_group_dims)
        if budget.mode == "ratio":
            target_bytes = int(total_original_bytes / budget.value)
        elif budget.mode == "bytes":
            target_bytes = int(budget.value)
        else:
            target_bytes = None
            full_energy = sum(
                sum(float((s ** 2).sum()) for s in sigs) / max(1, len(sigs))
                for sigs in per_group_sigmas
            )
            target_resid = (float(budget.value) ** 2) * full_energy

        def total_bytes_for_ranks(ranks_list: list[list[int]]) -> int:
            return sum(
                mem_tucker4d_compressed(tuple(ranks_list[g]), per_group_dims[g], dtype_bytes)
                for g in range(len(groups))
            )

        def residual_energy_for_ranks(ranks_list: list[list[int]]) -> float:
            tot = 0.0
            for g in range(len(per_group_sigmas)):
                for m in range(len(per_group_sigmas[g])):
                    s = per_group_sigmas[g][m]
                    r = ranks_list[g][m]
                    if r < s.numel():
                        tot += float((s[r:] ** 2).sum())
            return tot

        all_sigmas = torch.cat([s.flatten() for sigs in per_group_sigmas for s in sigs])
        max_sigma2 = float((all_sigmas ** 2).max()) if all_sigmas.numel() else 1.0
        lo, hi = 1e-12, max(max_sigma2, 1e-6) * 1e6

        best_ranks = None
        for _ in range(self.max_outer):
            mid = (lo * hi) ** 0.5
            ranks_list = self._solve_inner(
                mid, per_group_sigmas, per_group_dims, per_group_max, dtype_bytes
            )
            if target_bytes is not None:
                cur = total_bytes_for_ranks(ranks_list)
                if abs(cur - target_bytes) <= self.tol_frac * target_bytes:
                    best_ranks = ranks_list
                    break
                if cur > target_bytes:
                    lo = mid
                else:
                    hi = mid
                best_ranks = ranks_list
            else:
                resid = residual_energy_for_ranks(ranks_list)
                if abs(resid - target_resid) <= self.tol_frac * max(target_resid, 1e-12):
                    best_ranks = ranks_list
                    break
                if resid > target_resid:
                    hi = mid
                else:
                    lo = mid
                best_ranks = ranks_list

        assert best_ranks is not None
        return [tuple(int(r) for r in best_ranks[g]) for g in range(len(groups))]

    def allocate(
        self,
        groups: list[torch.Tensor],
        budget: BudgetSpec,
        decomposition: Decomposition,
    ) -> list[tuple[int, ...]]:
        return self._run(groups, budget, decomposition)

    def allocate_joint(
        self,
        k_groups: list[torch.Tensor],
        v_groups: list[torch.Tensor],
        budget: BudgetSpec,
        decomposition: Decomposition,
    ) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
        n_k = len(k_groups)
        all_ranks = self._run(k_groups + v_groups, budget, decomposition)
        return all_ranks[:n_k], all_ranks[n_k:]
