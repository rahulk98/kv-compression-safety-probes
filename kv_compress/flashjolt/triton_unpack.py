"""Triton and CPU implementations of range-limited bit unpacking.

The Triton path uses absolute element indices, so ranges may start or end
mid-byte. Inputs use the 2, 4, or 8-bit layout from ``residuals._quant``;
Triton is optional and CPU dispatch uses the independent reference.
"""

from __future__ import annotations

import logging

import torch

from . import reference as _reference

logger = logging.getLogger(__name__)

VALID_UNPACK_BITS = (2, 4, 8)

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover -- exercised by definition on this Mac dev machine
    TRITON_AVAILABLE = False


# Define Triton kernels only when Triton is available.

if TRITON_AVAILABLE:

    @triton.jit
    def _decode_lane(packed_ptr, idx, mask, bits: tl.constexpr, vpb: tl.constexpr):
        """Decode packed bit-codes at ABSOLUTE index `idx` (any shape -- 1D for the standalone unpack kernel, 2D for `kernel.py`'s fused reconstruction kernel) into signed int32 values, entirely in registers -- no decoded-codes array is ever written to global memory."""
        if bits == 8:
            byte_val = tl.load(packed_ptr + idx, mask=mask, other=0).to(tl.int32)
            signed = tl.where(byte_val >= 128, byte_val - 256, byte_val)
        else:
            byte_idx = idx // vpb
            sub_pos = idx % vpb
            shift = bits * sub_pos
            field_mask = (1 << bits) - 1
            offset = 1 << (bits - 1)
            byte_val = tl.load(packed_ptr + byte_idx, mask=mask, other=0).to(tl.int32)
            field = (byte_val >> shift) & field_mask
            signed = field - offset
        return tl.where(mask, signed, 0)

    @triton.jit
    def _unpack_block_kernel(
        packed_ptr,
        out_ptr,
        start,
        n,
        bits: tl.constexpr,
        vpb: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """One program instance decodes up to `BLOCK_SIZE` output elements.
        `idx = start + local_offset` is each lane's ABSOLUTE index into the
        conceptual `total_numel`-length codes array; `byte_idx = idx // vpb`
        and `sub_pos = idx % vpb` (bits < 8 only) are derived straight from
        that absolute index, which is what makes an arbitrary mid-byte
        `start`/`end` need no special-casing (see module docstring).
        """
        pid = tl.program_id(0)
        local_offset = tl.arange(0, BLOCK_SIZE)  # this program's lane -> output position pid*BLOCK_SIZE + local_offset
        out_pos = pid * BLOCK_SIZE + local_offset
        mask = out_pos < n
        idx = start + out_pos  # absolute index into the conceptual total_numel-length codes array
        signed = _decode_lane(packed_ptr, idx, mask, bits, vpb)
        tl.store(out_ptr + out_pos, signed.to(tl.int8), mask=mask)


def _validate_bits(bits: int) -> None:
    if bits not in VALID_UNPACK_BITS:
        raise ValueError(f"bits must be one of {VALID_UNPACK_BITS}, got {bits}")


def unpack_block_triton(
    packed: torch.Tensor, bits: int, total_numel: int, start: int, end: int
) -> torch.Tensor:
    """Triton implementation of `reference.unpack_block`'s exact signature and semantics: unpack elements `[start, end)` of a bit-packed uint8 tensor to signed int8 codes, for bits in {2, 4, 8}."""
    _validate_bits(bits)
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "unpack_block_triton() called but `triton` is not importable in this "
            "environment -- use unpack_block() for automatic CPU fallback, or run "
            "this on a host with a CUDA Triton install (see module docstring)."
        )
    if not torch.cuda.is_available() or not packed.is_cuda:
        raise RuntimeError(
            "unpack_block_triton() requires `packed` on a CUDA device with CUDA "
            f"available; got packed.device={packed.device}, "
            f"torch.cuda.is_available()={torch.cuda.is_available()}. Use "
            "unpack_block() for automatic CPU fallback."
        )

    # Guard 1: the kernel assumes `packed` is contiguous (it advances
    # `packed_ptr` by plain element offsets); a strided view would decode
    # silently wrong values. See docstring above for why `.contiguous()`
    # over raising.
    packed = packed.contiguous()

    # Guard 2: clamp `end` to `total_numel` so this matches
    # `_quant.unpack(...)[start:end]`'s Python slice semantics exactly,
    # instead of allocating/reading past the last real element.
    end = min(end, total_numel)
    n = max(end - start, 0)
    out = torch.empty(n, dtype=torch.int8, device=packed.device)
    if n == 0:
        return out

    # Guard 3: int32 index-arithmetic overflow check. See docstring above
    # for why this is validation, not a cast to int64.
    if end > (1 << 31) - 1:
        raise ValueError(
            f"unpack_block_triton: end={end} exceeds 2**31-1; the kernel's "
            "index arithmetic is int32 and would silently overflow past this "
            "bound. Split the request into smaller blocks."
        )

    vpb = 1 if bits == 8 else 8 // bits
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _unpack_block_kernel[grid](
        packed, out, start, n, bits=bits, vpb=vpb, BLOCK_SIZE=BLOCK_SIZE
    )
    return out



unpack_block_cpu = _reference.unpack_block




def select_backend(device: torch.device | str) -> str:
    """Which backend `unpack_block()` will use for a tensor on `device`,
    without running anything -- `"triton"` when Triton is importable AND a
    CUDA device is available AND `device` itself is CUDA, else `"cpu"`.
    Exposed directly (not just as a log line) so tests and callers can
    assert on the dispatch decision itself.
    """
    device = torch.device(device)
    if TRITON_AVAILABLE and torch.cuda.is_available() and device.type == "cuda":
        return "triton"
    return "cpu"


def unpack_block(
    packed: torch.Tensor, bits: int, total_numel: int, start: int, end: int
) -> torch.Tensor:
    """Drop-in replacement for `reference.unpack_block` with the identical
    signature and semantics: dispatches to the Triton kernel
    (`unpack_block_triton`) when a supported CUDA device is available, else
    automatically falls back to the existing CPU implementation
    (`reference.unpack_block`, unchanged). The chosen backend is logged at
    INFO level and also queryable ahead of time via `select_backend`.
    """
    backend = select_backend(packed.device)
    logger.info(
        "unpack_block dispatch: backend=%s device=%s bits=%s total_numel=%s start=%s end=%s",
        backend, packed.device, bits, total_numel, start, end,
    )
    if backend == "triton":
        return unpack_block_triton(packed, bits, total_numel, start, end)
    return unpack_block_cpu(packed, bits, total_numel, start, end)
