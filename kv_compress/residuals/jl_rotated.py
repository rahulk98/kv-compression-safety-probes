from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .base import EncodedResidual, ResidualEncoder
from ._quant import SCALE_BYTES, SEED_BYTES, pack, quant_range, unpack


@dataclass
class _JLPayload:
    packed: torch.Tensor       # uint8 bit-packed codes
    scale: torch.Tensor        # per-row scales, shape = E.shape[:-1]
    seed: int
    bits: int
    numel: int                 # original number of elements before bit-pack padding
    orig_dtype: torch.dtype


class JLRotatedResidual(ResidualEncoder):
    """Random-orthogonal rotation along the last dim, then per-row quantisation.

    The rotation smooths heavy-tailed coordinates (QuaRot/SpinQuant style); the
    per-row scale (one float per slice along the last dim) avoids the global-amax
    failure mode where ~99% of values round to zero at low bit-widths.
    """

    def __init__(self, bits: int = 4, seed: int = 0):
        if bits not in (2, 4, 8):
            raise ValueError("bits must be 2, 4, or 8")
        self.bits = bits
        self.seed = seed
        self._rot_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

    def _rotation(self, d: int, device, dtype) -> torch.Tensor:
        key = (d, device, dtype)
        cached = self._rot_cache.get(key)
        if cached is not None:
            return cached
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        A = torch.randn(d, d, generator=gen)
        Q, _ = torch.linalg.qr(A)
        Q = Q.to(device=device, dtype=dtype)
        self._rot_cache[key] = Q
        return Q

    def expected_footprint_bytes(self, shape: tuple[int, ...]) -> int:
        if not shape:
            return SEED_BYTES
        numel = math.prod(shape)
        n_scales = numel // shape[-1] if shape[-1] > 0 else 1
        return math.ceil(numel * self.bits / 8) + n_scales * SCALE_BYTES + SEED_BYTES

    def encode(self, E: torch.Tensor) -> EncodedResidual:
        orig_dtype = E.dtype
        E32 = E.to(torch.float32)
        R = self._rotation(E.shape[-1], E.device, torch.float32)
        rotated = E32 @ R.T

        qmin, qmax = quant_range(self.bits)
        safe_qmax = max(qmax, 1)
        amax = rotated.abs().amax(dim=-1, keepdim=True)
        scale_vals = torch.where(amax > 0, amax / safe_qmax, torch.ones_like(amax))
        q = torch.round(rotated / scale_vals).clamp(qmin, qmax).to(torch.int8)

        packed = pack(q, self.bits)
        n_scales = scale_vals.numel()
        nbytes = packed.numel() + n_scales * SCALE_BYTES + SEED_BYTES

        payload = _JLPayload(
            packed=packed,
            scale=scale_vals.squeeze(-1).to(torch.float32),
            seed=self.seed,
            bits=self.bits,
            numel=q.numel(),
            orig_dtype=orig_dtype,
        )
        return EncodedResidual(payload=payload, shape=tuple(int(s) for s in E.shape), nbytes=nbytes)

    def decode(self, encoded: EncodedResidual) -> torch.Tensor:
        p: _JLPayload = encoded.payload  # type: ignore[assignment]
        q = unpack(p.packed, p.bits, p.numel).to(torch.float32).reshape(encoded.shape)
        R = self._rotation(encoded.shape[-1], q.device, torch.float32)
        scale = p.scale.to(q.device).reshape(*encoded.shape[:-1], 1)
        rotated = q * scale
        return (rotated @ R).to(p.orig_dtype)
