from __future__ import annotations

from math import prod

import torch

SCALE_BYTES = 4
SEED_BYTES = 8


def quant_range(bits: int) -> tuple[int, int]:
    qmax = (1 << (bits - 1)) - 1
    return -qmax, qmax


def largest_axis(shape: tuple[int, ...]) -> int:
    """Index of the largest dimension in `shape` (ties -> lowest index)."""
    if not shape:
        return 0
    return max(range(len(shape)), key=lambda i: shape[i])


def payload_quant_bytes(shape: tuple[int, ...], bits: int) -> int:
    """Return packed payload bytes plus one fp32 scale per largest-axis block."""
    if bits not in (2, 4, 8):
        raise ValueError(f"bits must be 2, 4, or 8; got {bits}")
    if not shape:
        return 0
    ax = largest_axis(shape)
    numel = prod(shape)
    block_len = shape[ax]
    n_blocks = numel // block_len if block_len > 0 else 1
    return (numel * bits + 7) // 8 + n_blocks * SCALE_BYTES


def quantize_dequantize_block(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, int]:
    """Quantize and immediately dequantize using symmetric absmax blocks.

    Returns the reconstructed tensor and its packed payload byte count.
    Supported bit widths are 2, 4, and 8.
    """
    if bits not in (2, 4, 8):
        raise ValueError(f"bits must be 2, 4, or 8; got {bits}")
    shape = tuple(x.shape)
    if not shape:
        return x.clone(), 0

    ax = largest_axis(shape)
    orig_dtype = x.dtype
    x32 = x.to(torch.float32)

    qmin, qmax = quant_range(bits)
    safe_qmax = max(qmax, 1)
    amax = x32.abs().amax(dim=ax, keepdim=True)
    scale = torch.where(amax > 0, amax / safe_qmax, torch.ones_like(amax))
    q = torch.round(x32 / scale).clamp(qmin, qmax).to(torch.int8)
    x_hat = (q.to(torch.float32) * scale).to(orig_dtype)

    return x_hat, payload_quant_bytes(shape, bits)


def pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack signed int8 codes into a flat bit-packed uint8 tensor."""
    if bits == 8:
        return codes.reshape(-1).to(torch.uint8)
    vpb = 8 // bits
    offset = 1 << (bits - 1)
    flat = codes.reshape(-1).to(torch.int32) + offset
    pad = (-flat.numel()) % vpb
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    grouped = flat.reshape(-1, vpb)
    shifts = torch.tensor([bits * i for i in range(vpb)], dtype=torch.int32, device=grouped.device)
    return (grouped << shifts).sum(dim=1).to(torch.uint8)


def unpack(packed: torch.Tensor, bits: int, numel: int) -> torch.Tensor:
    """Unpack a bit-packed uint8 tensor back to signed int8 codes."""
    if bits == 8:
        return packed[:numel].to(torch.int8)
    vpb = 8 // bits
    offset = 1 << (bits - 1)
    mask = (1 << bits) - 1
    p = packed.to(torch.int32)
    parts = []
    for i in range(vpb):
        shift = bits * i
        parts.append((p >> shift) & mask)
    flat = torch.stack(parts, dim=1).reshape(-1)
    return (flat[:numel] - offset).to(torch.int8)
