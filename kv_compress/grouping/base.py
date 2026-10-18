from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class GroupingStrategy(ABC):
    @abstractmethod
    def assign(
        self,
        layers: dict[int, dict[str, torch.Tensor]],
    ) -> list[list[int]]: ...
