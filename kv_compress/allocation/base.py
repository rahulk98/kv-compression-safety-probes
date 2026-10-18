from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch

from ..decompositions.base import Decomposition


@dataclass
class BudgetSpec:
    mode: Literal["ratio", "bytes", "error"]
    value: float
    dtype_bytes: int = 2


class RankAllocator(ABC):
    @abstractmethod
    def allocate(
        self,
        groups: list[torch.Tensor],
        budget: BudgetSpec,
        decomposition: Decomposition,
    ) -> list[tuple[int, ...]]: ...

    @abstractmethod
    def allocate_joint(
        self,
        k_groups: list[torch.Tensor],
        v_groups: list[torch.Tensor],
        budget: BudgetSpec,
        decomposition: Decomposition,
    ) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]: ...

    def allocate_joint_with_bits(
        self,
        k_groups: list[torch.Tensor],
        v_groups: list[torch.Tensor],
        budget: BudgetSpec,
        decomposition: Decomposition,
    ) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], list[int], list[int]]:
        """Default: allocators that don't co-decide residual bits return zeros.

        The pipeline can call this method uniformly. Allocators that jointly
        decide residual bit-widths (e.g. JointTuckerJLAllocator) override.
        """
        rk, rv = self.allocate_joint(k_groups, v_groups, budget, decomposition)
        return rk, rv, [0] * len(rk), [0] * len(rv)
