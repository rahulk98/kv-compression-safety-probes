from __future__ import annotations

from math import prod


def mem_tucker4d_breakdown(
    ranks: tuple[int, ...],
    dims: tuple[int, ...],
    dtype_bytes: int = 2,
) -> tuple[int, int]:
    core = int(prod(ranks) * dtype_bytes)
    factors = int(sum(d * r for d, r in zip(dims, ranks)) * dtype_bytes)
    return core, factors


def mem_tucker4d_compressed(
    ranks: tuple[int, ...],
    dims: tuple[int, ...],
    dtype_bytes: int = 2,
) -> int:
    core, factors = mem_tucker4d_breakdown(ranks, dims, dtype_bytes)
    return core + factors


def mem_original(dims: tuple[int, ...], dtype_bytes: int = 2) -> int:
    return int(prod(dims) * dtype_bytes)
