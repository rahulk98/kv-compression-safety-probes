from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Approximation(ABC):
    ranks: tuple[int, ...]
    shape: tuple[int, ...]

    @abstractmethod
    def reconstruct(self) -> torch.Tensor: ...

    @abstractmethod
    def footprint_bytes(self, dtype_bytes: int = 2) -> int: ...

    def payload_tensors(self) -> list[torch.Tensor]:
        """Tensors this approximation stores persistently (e.g. a Tucker
        core + factor matrices). Default: none -- a safe fallback for
        subclasses that don't opt into `quantize_payload_`.
        """
        return []

    def quantize_payload_(self, bits: int) -> None:
        """In-place quantize-dequantize (Q-DQ) every tensor returned by
        `payload_tensors()` to `bits` bits/element
        (`kv_compress.residuals._quant.quantize_dequantize_block`), and
        record the achieved bit-width so a subsequent `footprint_bytes()`
        call prices the payload at `bits` (plus per-block fp32 scales)
        instead of the default `dtype_bytes` convention.

        Mutates the payload tensors in place (`Tensor.copy_`), so any other
        object already holding a reference to them (e.g. a cached
        reconstruction) sees the quantized values on its next use -- this
        applies the quantization for real; it is not a hypothetical
        re-pricing.

        Raises NotImplementedError if `payload_tensors()` returns an empty
        list (the subclass has not opted in).
        """
        from ..residuals._quant import quantize_dequantize_block

        tensors = self.payload_tensors()
        if not tensors:
            raise NotImplementedError(
                f"{type(self).__name__} has no payload tensors to quantize "
                "(payload_tensors() returned an empty list); override it to "
                "support quantize_payload_."
            )
        total_bytes = 0
        for t in tensors:
            x_hat, nbytes = quantize_dequantize_block(t, bits)
            t.copy_(x_hat)
            total_bytes += nbytes
        self._quant_bits = int(bits)
        self._quant_bytes = int(total_bytes)


class Decomposition(ABC):
    @abstractmethod
    def fit(
        self,
        X: torch.Tensor,
        ranks: tuple[int, ...],
    ) -> "Approximation": ...

    @abstractmethod
    def hosvd_singular_values(self, X: torch.Tensor) -> list[torch.Tensor]: ...

    @abstractmethod
    def max_ranks(self, shape: tuple[int, ...]) -> tuple[int, ...]: ...
