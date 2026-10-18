from __future__ import annotations

import torch

from .base import GroupingStrategy


class FixedSizeGrouping(GroupingStrategy):
    def __init__(self, size: int):
        if size < 1:
            raise ValueError("group size must be >= 1")
        self.size = size

    def assign(
        self,
        layers: dict[int, dict[str, torch.Tensor]],
    ) -> list[list[int]]:
        idxs = sorted(layers.keys())
        return [idxs[i : i + self.size] for i in range(0, len(idxs), self.size)]
