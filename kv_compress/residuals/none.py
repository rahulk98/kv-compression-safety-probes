from __future__ import annotations

import torch

from .base import EncodedResidual, ResidualEncoder


class NoResidual(ResidualEncoder):
    def __init__(self, **_kwargs):
        pass

    def encode(self, E: torch.Tensor) -> EncodedResidual:
        return EncodedResidual(
            payload={"dtype": E.dtype, "device": E.device},
            shape=tuple(int(s) for s in E.shape),
            nbytes=0,
        )

    def decode(self, encoded: EncodedResidual) -> torch.Tensor:
        meta = encoded.payload or {}
        return torch.zeros(encoded.shape, dtype=meta.get("dtype"), device=meta.get("device"))

    def expected_footprint_bytes(self, shape: tuple[int, ...]) -> int:
        return 0
