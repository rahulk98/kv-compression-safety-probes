from __future__ import annotations

import torch

from .base import LossFormulation


class FactorSpaceLoss(LossFormulation):
    def __call__(self, X: torch.Tensor, X_hat: torch.Tensor) -> torch.Tensor:
        X32 = X.to(torch.float32)
        X_hat32 = X_hat.to(torch.float32)
        denom = torch.linalg.norm(X32)
        if denom == 0:
            return torch.tensor(0.0, dtype=torch.float32)
        return torch.linalg.norm(X32 - X_hat32) / denom
