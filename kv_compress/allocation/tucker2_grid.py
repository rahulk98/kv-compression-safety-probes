from __future__ import annotations

import torch

from ..decompositions.base import Decomposition
from .base import BudgetSpec, RankAllocator
from .memory import mem_original, mem_tucker4d_compressed


def _residual_energy_table(sq: torch.Tensor) -> torch.Tensor:
    """Suffix-sum table: out[r] = sum_{i>=r} sq[i] for r in 0..len(sq).

    Padded so out[len(sq)] == 0 (residual after keeping every singular value).
    """
    if sq.numel() == 0:
        return torch.zeros(1, dtype=sq.dtype, device=sq.device)
    flipped = torch.flip(sq, dims=[0])
    suffix = torch.flip(torch.cumsum(flipped, dim=0), dims=[0])
    return torch.cat([suffix, torch.zeros(1, dtype=sq.dtype, device=sq.device)])


class Tucker2GridAllocator(RankAllocator):
    """Exact rank allocator for the (r_T, r_d) sub-problem."""

    active_modes: tuple[int, ...] = (2, 3)

    def __init__(self, max_outer: int = 30, tol_frac: float = 0.005):
        self.max_outer = max_outer
        self.tol_frac = tol_frac

    def _solve_inner(
        self,
        lam: float,
        per_group_eps_T: list[torch.Tensor],
        per_group_eps_d: list[torch.Tensor],
        per_group_dims: list[tuple[int, ...]],
        per_group_max: list[tuple[int, ...]],
        dtype_bytes: int,
    ) -> list[list[int]]:
        all_ranks: list[list[int]] = []
        for g in range(len(per_group_dims)):
            L_g, n_h, T, d_h = per_group_dims[g]
            r_L_max = per_group_max[g][0]
            r_h_max = per_group_max[g][1]
            T_max = min(per_group_max[g][2], per_group_eps_T[g].numel() - 1)
            d_max = min(per_group_max[g][3], per_group_eps_d[g].numel() - 1)

            eps_T = per_group_eps_T[g][1 : T_max + 1]
            eps_d = per_group_eps_d[g][1 : d_max + 1]

            r_T_vec = torch.arange(1, T_max + 1, dtype=torch.float64)
            r_d_vec = torch.arange(1, d_max + 1, dtype=torch.float64)

            core_bytes = (r_L_max * r_h_max) * r_T_vec[:, None] * r_d_vec[None, :] * dtype_bytes
            fac_T = (T * r_T_vec * dtype_bytes)[:, None].expand(T_max, d_max)
            fac_d = (d_h * r_d_vec * dtype_bytes)[None, :].expand(T_max, d_max)
            mem_grid = core_bytes + fac_T + fac_d                                  # (T_max, d_max)

            err_grid = eps_T.to(torch.float64)[:, None] + eps_d.to(torch.float64)[None, :]
            obj = err_grid + lam * mem_grid                                        # (T_max, d_max)

            flat_idx = int(torch.argmin(obj).item())
            i, j = divmod(flat_idx, d_max)
            r_T_opt = i + 1
            r_d_opt = j + 1

            ranks = [int(r_L_max), int(r_h_max), int(r_T_opt), int(r_d_opt)]
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
        per_group_norms = [float((sigs[0] ** 2).sum()) ** 0.5 for sigs in per_group_sigmas]
        per_group_eps_T: list[torch.Tensor] = []
        per_group_eps_d: list[torch.Tensor] = []
        for sigs, norm in zip(per_group_sigmas, per_group_norms):
            scale = max(norm, 1e-12)
            sq_T = (sigs[2] / scale) ** 2
            sq_d = (sigs[3] / scale) ** 2
            per_group_eps_T.append(_residual_energy_table(sq_T))
            per_group_eps_d.append(_residual_energy_table(sq_d))

        total_original_bytes = sum(mem_original(d, dtype_bytes) for d in per_group_dims)
        if budget.mode == "ratio":
            target_bytes = int(total_original_bytes / budget.value)
            target_resid = None
        elif budget.mode == "bytes":
            target_bytes = int(budget.value)
            target_resid = None
        else:
            target_bytes = None
            full_energy = sum(
                float(eps_T[0]) + float(eps_d[0])
                for eps_T, eps_d in zip(per_group_eps_T, per_group_eps_d)
            ) / 2.0  # average across the two active modes
            target_resid = (float(budget.value) ** 2) * full_energy

        def total_bytes_for_ranks(ranks_list: list[list[int]]) -> int:
            return sum(
                mem_tucker4d_compressed(tuple(ranks_list[g]), per_group_dims[g], dtype_bytes)
                for g in range(len(groups))
            )

        def residual_energy_for_ranks(ranks_list: list[list[int]]) -> float:
            tot = 0.0
            for g in range(len(groups)):
                r_T = ranks_list[g][2]
                r_d = ranks_list[g][3]
                tot += float(per_group_eps_T[g][r_T]) + float(per_group_eps_d[g][r_d])
            return tot

        lo, hi = 1e-18, 1.0
        hi = 1e6

        best_ranks = None
        for _ in range(self.max_outer):
            mid = (lo * hi) ** 0.5
            ranks_list = self._solve_inner(
                mid, per_group_eps_T, per_group_eps_d, per_group_dims, per_group_max, dtype_bytes
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
