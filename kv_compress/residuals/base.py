from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass
class EncodedResidual:
    payload: object
    shape: tuple[int, ...]
    nbytes: int


class ResidualEncoder(ABC):
    @abstractmethod
    def encode(self, E: torch.Tensor) -> EncodedResidual: ...

    @abstractmethod
    def decode(self, encoded: EncodedResidual) -> torch.Tensor: ...

    @abstractmethod
    def expected_footprint_bytes(self, shape: tuple[int, ...]) -> int:
        """Bytes this encoder will consume for a tensor of the given shape, without encoding."""
        ...

    def footprint_bytes(self, encoded: EncodedResidual) -> int:
        return encoded.nbytes

    def attach_to_attention(self, model):
        return nullcontext()
