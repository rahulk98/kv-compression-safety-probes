from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .base import EncodedResidual, ResidualEncoder
from ._quant import SCALE_BYTES, SEED_BYTES, pack, quant_range, unpack


@dataclass
class _JLProjPayload:
    packed: torch.Tensor       # uint8 bit-packed codes of the projected coeffs
    scale: torch.Tensor        # per-row scales, shape = E.shape[:-1]
    seed: int
    bits: int
    k: int                     # projected dimension (coeffs kept per row)
    numel: int                 # projected element count before bit-pack padding
    orig_dtype: torch.dtype


class JLProjectionResidual(ResidualEncoder):
    """True Johnson-Lindenstrauss residual: random projection R^d -> R^k with k <= d along the last dim, then per-row quantisation of the k kept coefficients."""

    def __init__(self, bits: int = 4, k: int | None = None, seed: int = 0):
        if bits not in (2, 4, 8):
            raise ValueError("bits must be 2, 4, or 8")
        if k is not None and k < 1:
            raise ValueError(f"k must be >= 1; got {k}")
        self.bits = bits
        self.k = k
        self.seed = seed
        self._proj_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def _resolve_k(self, d: int) -> int:
        k = d if self.k is None else self.k
        if k > d:
            raise ValueError(f"k={k} exceeds last-dim size d={d}")
        return k

    def _projection(self, d: int, device) -> torch.Tensor:
        """Return P of shape (k, d) with orthonormal rows, cached per (d, device)."""
        key = (d, device)
        cached = self._proj_cache.get(key)
        if cached is not None:
            return cached
        k = self._resolve_k(d)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        A = torch.randn(d, k, generator=gen)
        Q, _ = torch.linalg.qr(A)
        P = Q.T.contiguous().to(device=device, dtype=torch.float32)
        self._proj_cache[key] = P
        return P

    def expected_footprint_bytes(self, shape: tuple[int, ...]) -> int:
        if not shape:
            return SEED_BYTES
        d = shape[-1]
        if d <= 0:
            return SEED_BYTES
        k = self._resolve_k(d)
        n_rows = math.prod(shape) // d
        proj_numel = n_rows * k
        return math.ceil(proj_numel * self.bits / 8) + n_rows * SCALE_BYTES + SEED_BYTES

    def encode(self, E: torch.Tensor) -> EncodedResidual:
        orig_dtype = E.dtype
        E32 = E.to(torch.float32)
        d = E.shape[-1]
        P = self._projection(d, E.device)
        Y = E32 @ P.T  # (..., k)

        qmin, qmax = quant_range(self.bits)
        safe_qmax = max(qmax, 1)
        amax = Y.abs().amax(dim=-1, keepdim=True)
        scale_vals = torch.where(amax > 0, amax / safe_qmax, torch.ones_like(amax))
        q = torch.round(Y / scale_vals).clamp(qmin, qmax).to(torch.int8)

        packed = pack(q, self.bits)
        n_scales = scale_vals.numel()
        nbytes = packed.numel() + n_scales * SCALE_BYTES + SEED_BYTES

        payload = _JLProjPayload(
            packed=packed,
            scale=scale_vals.squeeze(-1).to(torch.float32),
            seed=self.seed,
            bits=self.bits,
            k=int(P.shape[0]),
            numel=q.numel(),
            orig_dtype=orig_dtype,
        )
        return EncodedResidual(payload=payload, shape=tuple(int(s) for s in E.shape), nbytes=nbytes)

    def decode(self, encoded: EncodedResidual) -> torch.Tensor:
        p: _JLProjPayload = encoded.payload  # type: ignore[assignment]
        proj_shape = (*encoded.shape[:-1], p.k)
        q = unpack(p.packed, p.bits, p.numel).to(torch.float32).reshape(proj_shape)
        P = self._projection(encoded.shape[-1], q.device)
        scale = p.scale.to(q.device).reshape(*encoded.shape[:-1], 1)
        Y = q * scale
        return (Y @ P).to(p.orig_dtype)
