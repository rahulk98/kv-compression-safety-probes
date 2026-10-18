from __future__ import annotations

import torch

from .. import instrumentation
from ..decompositions.base import Decomposition
from ..residuals.jl_rotated import JLRotatedResidual
from .base import BudgetSpec, RankAllocator
from .lagrangian import _greedy_ranks_for_group
from .memory import mem_original, mem_tucker4d_compressed

_BIT_CHOICES: tuple[int, ...] = (0, 2, 4, 8)


class JointTuckerJLAllocator(RankAllocator):
    """Joint Lagrangian over Tucker ranks AND JL residual bits per group."""

    def __init__(
        self,
        max_outer: int = 50,
        tol_frac: float = 0.005,
        active_modes: tuple[int, ...] = (2, 3),
        residual_seed: int = 0,
        calibration_n: int = 64,
        *,
        force_bits: int | None = None,
        uniform_rank: bool = False,
        couple_kv: bool = True,
        max_token_rank: int | None = None,
    ):
        # Ablation knobs default off; defaults preserve the production solve.
        if force_bits is not None and force_bits not in _BIT_CHOICES:
            raise ValueError(
                f"force_bits must be one of {_BIT_CHOICES}, got {force_bits}"
            )
        self.max_outer = max_outer
        self.tol_frac = tol_frac
        self.active_modes = active_modes
        self.residual_seed = residual_seed
        self.calibration_n = calibration_n
        self.force_bits = force_bits
        self.uniform_rank = uniform_rank
        self.couple_kv = couple_kv
        self.max_token_rank = max_token_rank
        self._eps2_cache: dict[int, dict[int, float]] = {}
        self._enc_cache: dict[int, JLRotatedResidual] = {}

    def _encoder(self, bits: int) -> JLRotatedResidual:
        enc = self._enc_cache.get(bits)
        if enc is None:
            enc = JLRotatedResidual(bits=bits, seed=self.residual_seed)
            self._enc_cache[bits] = enc
        return enc

    def _calibrate_eps2(self, last_dim: int) -> dict[int, float]:
        cached = self._eps2_cache.get(last_dim)
        if cached is not None:
            return cached
        gen = torch.Generator(device="cpu").manual_seed(self.residual_seed + 1)
        n = self.calibration_n
        E = torch.randn(n, n, last_dim, generator=gen, dtype=torch.float32)
        denom = float((E ** 2).sum())
        eps2 = {0: 1.0}
        for b in (2, 4, 8):
            enc = self._encoder(b)
            E_hat = enc.decode(enc.encode(E))
            num = float(((E - E_hat) ** 2).sum())
            eps2[b] = num / max(denom, 1e-12)
        self._eps2_cache[last_dim] = eps2
        return eps2

    def _residual_bytes(self, bits: int, dims: tuple[int, ...]) -> int:
        if bits == 0:
            return 0
        return int(self._encoder(bits).expected_footprint_bytes(dims))

    @staticmethod
    def _tail_energy(
        sigmas: list[torch.Tensor],
        ranks: list[int],
        active_modes: tuple[int, ...],
    ) -> float:
        tot = 0.0
        for m in active_modes:
            s = sigmas[m]
            r = ranks[m]
            if r < s.numel():
                tot += float((s[r:] ** 2).sum())
        return tot

    def _solve_inner_joint(
        self,
        lam: float,
        per_group_sigmas: list[list[torch.Tensor]],
        per_group_dims: list[tuple[int, ...]],
        per_group_max: list[tuple[int, ...]],
        per_group_eps2: list[dict[int, float]],
        dtype_bytes: int,
    ) -> tuple[list[list[int]], list[int]]:
        all_ranks: list[list[int]] = []
        all_bits: list[int] = []
        for g in range(len(per_group_sigmas)):
            sigmas = per_group_sigmas[g]
            dims = per_group_dims[g]
            max_r = per_group_max[g]
            eps2 = per_group_eps2[g]
            best: tuple[float, list[int], int] | None = None
            bit_choices = (
                _BIT_CHOICES if self.force_bits is None else (self.force_bits,)
            )
            for b in bit_choices:
                eps = eps2[b]
                lam_eff = lam / max(eps, 1e-12)
                ranks_b = _greedy_ranks_for_group(
                    lam_eff, sigmas, dims, max_r, self.active_modes, dtype_bytes
                )
                tail = self._tail_energy(sigmas, ranks_b, self.active_modes)
                tucker_b = mem_tucker4d_compressed(tuple(ranks_b), dims, dtype_bytes)
                jl_b = self._residual_bytes(b, dims)
                L = eps * tail + lam * (tucker_b + jl_b)
                if best is None or L < best[0]:
                    best = (L, ranks_b, b)
            assert best is not None
            all_ranks.append(best[1])
            all_bits.append(best[2])
        return all_ranks, all_bits

    def _group_ranks_from_shared(
        self, shared: dict[int, int], max_ranks: tuple[int, ...]
    ) -> list[int]:
        """Expand a shared active-mode rank dict into a full per-group rank
        tuple: pinned modes (not in active_modes) at this group's max,
        active modes at the shared level clamped to this group's max."""
        return [
            int(max_ranks[m])
            if m not in self.active_modes
            else min(int(shared[m]), int(max_ranks[m]))
            for m in range(len(max_ranks))
        ]

    def _solve_inner_uniform(
        self,
        lam: float,
        per_group_sigmas: list[list[torch.Tensor]],
        per_group_dims: list[tuple[int, ...]],
        per_group_max: list[tuple[int, ...]],
        per_group_eps2: list[dict[int, float]],
        dtype_bytes: int,
    ) -> tuple[list[list[int]], list[int]]:
        """A5 uniform-rank inner solve. One active-mode rank level is shared across every group (instead of the per-group greedy in ``_solve_inner_joint``)."""
        n_groups = len(per_group_sigmas)
        active = self.active_modes
        # Shared rank level per active mode, clamped to the smallest per-group
        # max so it is valid for every group.
        min_max = {
            m: min(int(per_group_max[g][m]) for g in range(n_groups))
            for m in active
        }
        shared = {m: 1 for m in active}
        improved = True
        while improved:
            improved = False
            for m in active:
                if shared[m] >= min_max[m]:
                    continue
                sum_err = 0.0
                sum_db = 0
                idx = shared[m]
                for g in range(n_groups):
                    sigma = per_group_sigmas[g][m]
                    if idx < sigma.numel():
                        sum_err += float(sigma[idx] ** 2)
                    ranks_g = self._group_ranks_from_shared(
                        shared, per_group_max[g]
                    )
                    cur_b = mem_tucker4d_compressed(
                        tuple(ranks_g), per_group_dims[g], dtype_bytes
                    )
                    trial = list(ranks_g)
                    trial[m] = shared[m] + 1
                    new_b = mem_tucker4d_compressed(
                        tuple(trial), per_group_dims[g], dtype_bytes
                    )
                    sum_db += new_b - cur_b
                if sum_db > 0 and sum_err > lam * sum_db:
                    shared[m] += 1
                    improved = True

        bit_choices = (
            _BIT_CHOICES if self.force_bits is None else (self.force_bits,)
        )
        all_ranks: list[list[int]] = []
        all_bits: list[int] = []
        for g in range(n_groups):
            ranks_g = self._group_ranks_from_shared(shared, per_group_max[g])
            sigmas = per_group_sigmas[g]
            dims = per_group_dims[g]
            eps2 = per_group_eps2[g]
            tail = self._tail_energy(sigmas, ranks_g, active)
            tucker_b = mem_tucker4d_compressed(tuple(ranks_g), dims, dtype_bytes)
            best: tuple[float, int] | None = None
            for b in bit_choices:
                jl_b = self._residual_bytes(b, dims)
                L = eps2[b] * tail + lam * (tucker_b + jl_b)
                if best is None or L < best[0]:
                    best = (L, b)
            assert best is not None
            all_ranks.append(ranks_g)
            all_bits.append(best[1])
        return all_ranks, all_bits

    def _run_joint(
        self,
        groups: list[torch.Tensor],
        budget: BudgetSpec,
        decomposition: Decomposition,
    ) -> tuple[list[tuple[int, ...]], list[int]]:
        dtype_bytes = budget.dtype_bytes

        per_group_dims = [tuple(int(s) for s in g.shape) for g in groups]
        per_group_max = [decomposition.max_ranks(d) for d in per_group_dims]
        if self.max_token_rank is not None:
            per_group_max = [
                tuple(
                    m[i] if i != 2 else min(m[2], self.max_token_rank)
                    for i in range(len(m))
                )
                for m in per_group_max
            ]
        per_group_sigmas_raw = [
            [s.detach().cpu() for s in decomposition.hosvd_singular_values(g)]
            for g in groups
        ]
        # Mode-k Frobenius property: Σᵢ σ_{g,m}[i]² = ||X_g||²_F for any mode m.
        per_group_norms = [float((sigs[0] ** 2).sum()) ** 0.5 for sigs in per_group_sigmas_raw]
        per_group_sigmas = [
            [s / max(norm, 1e-12) for s in sigs]
            for sigs, norm in zip(per_group_sigmas_raw, per_group_norms)
        ]
        # Calibrate ε²(b) per last-dim. In practice d_h is constant across groups
        # for a given model so this is one cache key.
        per_group_eps2 = [self._calibrate_eps2(d[-1]) for d in per_group_dims]

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

        def total_bytes_for(ranks_list: list[list[int]], bits_list: list[int]) -> int:
            tot = 0
            for g in range(len(groups)):
                tot += mem_tucker4d_compressed(tuple(ranks_list[g]), per_group_dims[g], dtype_bytes)
                tot += self._residual_bytes(bits_list[g], per_group_dims[g])
            return tot

        def residual_energy_for(ranks_list: list[list[int]], bits_list: list[int]) -> float:
            tot = 0.0
            for g in range(len(groups)):
                tail = self._tail_energy(per_group_sigmas[g], ranks_list[g], self.active_modes)
                tot += per_group_eps2[g][bits_list[g]] * tail
            return tot

        all_sigmas = torch.cat([s.flatten() for sigs in per_group_sigmas for s in sigs])
        max_sigma2 = float((all_sigmas ** 2).max()) if all_sigmas.numel() else 1.0
        lo, hi = 1e-12, max(max_sigma2, 1e-6) * 1e6

        inner = (
            self._solve_inner_uniform
            if self.uniform_rank
            else self._solve_inner_joint
        )
        best: tuple[list[list[int]], list[int]] | None = None
        for _ in range(self.max_outer):
            mid = (lo * hi) ** 0.5
            ranks_list, bits_list = inner(
                mid,
                per_group_sigmas,
                per_group_dims,
                per_group_max,
                per_group_eps2,
                dtype_bytes,
            )
            if target_bytes is not None:
                cur = total_bytes_for(ranks_list, bits_list)
                if abs(cur - target_bytes) <= self.tol_frac * target_bytes:
                    best = (ranks_list, bits_list)
                    break
                if cur > target_bytes:
                    lo = mid  # too many bytes -> raise lambda
                else:
                    hi = mid
                best = (ranks_list, bits_list)
            else:
                resid = residual_energy_for(ranks_list, bits_list)
                if abs(resid - target_resid) <= self.tol_frac * max(target_resid, 1e-12):
                    best = (ranks_list, bits_list)
                    break
                if resid > target_resid:
                    hi = mid
                else:
                    lo = mid
                best = (ranks_list, bits_list)

        assert best is not None
        ranks_out = [tuple(int(r) for r in best[0][g]) for g in range(len(groups))]
        bits_out = [int(b) for b in best[1]]

        # Optional read-only objective instrumentation.
        if instrumentation.is_active():
            for g in range(len(groups)):
                b = bits_out[g]
                eps = per_group_eps2[g][b]
                tail = self._tail_energy(per_group_sigmas[g], list(ranks_out[g]), self.active_modes)
                tucker_bytes = mem_tucker4d_compressed(ranks_out[g], per_group_dims[g], dtype_bytes)
                jl_bytes = self._residual_bytes(b, per_group_dims[g])
                objective = eps * tail + mid * (tucker_bytes + jl_bytes)
                instrumentation.record_allocation_objective(
                    lam=mid, ranks=ranks_out[g], bits=b, objective=objective,
                )
        return ranks_out, bits_out

    def allocate(
        self,
        groups: list[torch.Tensor],
        budget: BudgetSpec,
        decomposition: Decomposition,
    ) -> list[tuple[int, ...]]:
        ranks, _ = self._run_joint(groups, budget, decomposition)
        return ranks

    def allocate_joint(
        self,
        k_groups: list[torch.Tensor],
        v_groups: list[torch.Tensor],
        budget: BudgetSpec,
        decomposition: Decomposition,
    ) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
        rk, rv, _, _ = self.allocate_joint_with_bits(k_groups, v_groups, budget, decomposition)
        return rk, rv

    def allocate_joint_with_bits(
        self,
        k_groups: list[torch.Tensor],
        v_groups: list[torch.Tensor],
        budget: BudgetSpec,
        decomposition: Decomposition,
    ) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], list[int], list[int]]:
        if not self.couple_kv:
            # Allocate K and V independently while preserving budget semantics.
            if budget.mode == "bytes":
                side_budget = BudgetSpec(
                    mode="bytes",
                    value=float(budget.value) / 2.0,
                    dtype_bytes=budget.dtype_bytes,
                )
            else:
                side_budget = budget
            ranks_k, bits_k = self._run_joint(k_groups, side_budget, decomposition)
            ranks_v, bits_v = self._run_joint(v_groups, side_budget, decomposition)
            return ranks_k, ranks_v, bits_k, bits_v
        n_k = len(k_groups)
        ranks, bits = self._run_joint(k_groups + v_groups, budget, decomposition)
        return ranks[:n_k], ranks[n_k:], bits[:n_k], bits[n_k:]
