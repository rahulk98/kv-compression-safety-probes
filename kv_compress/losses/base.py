from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class LossFormulation(ABC):
    @abstractmethod
    def __call__(self, X: torch.Tensor, X_hat: torch.Tensor) -> torch.Tensor: ...
