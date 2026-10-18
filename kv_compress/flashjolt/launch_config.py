"""Shared-memory-aware launch selection for FlashJoLT Triton kernels."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover -- torch is a hard dependency elsewhere in this repo
    TORCH_AVAILABLE = False

# Fallback limit from the A100-80GB PCIe hardware limit.
A100_SHARED_MEM_LIMIT_BYTES = 166912

_FP32_BYTES = 4

# Keep these defaults independent of the optional Triton import.
_PRODUCTION_RANK_CAP = 32
_PRODUCTION_BLOCK_DI = 32
_PRODUCTION_BLOCK_T = 32
# Explicit pipeline depth used by the launch selector.
_PRODUCTION_NUM_STAGES = 2


def _next_pow2(n: int) -> int:
    """Pure-Python `next_power_of_2`, matching `triton.next_power_of_2`'s
    result for `n >= 1` without requiring `triton` to be importable (see
    module docstring)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _rank_block(rank: int, cap: int) -> int:
    """Same shape as `kernel.py::_rank_block`, parameterised by `cap` so a
    ladder rung can try a smaller cap. At least 16 (`tl.dot`'s lane floor),
    at most `cap`."""
    return min(cap, max(16, _next_pow2(rank)))


@dataclass(frozen=True)
class KernelShapeParams:
    """The fixed problem shape for one fused-decode launch -- everything
    `_fused_decode_kernel`'s tile sizes depend on that a launch-config rung
    is NOT allowed to change (changing any of these would change which
    values are computed, not just how they are tiled)."""

    token_rank_k: int
    feature_rank_k: int
    token_rank_v: int
    head_dim: int
    n_rep: int


@dataclass(frozen=True)
class LaunchConfig:
    """One fully-resolved, launchable tile/stage configuration for
    `_fused_decode_kernel`. `block_d`/`block_h`/`block_rv_full` are DERIVED
    from `shape` and never varied by the ladder (see `KernelShapeParams`);
    `block_t`, `rank_cap` (which sets `block_rk`/`block_fk`), `block_di`,
    and `num_stages` are the levers `select_launch_config`'s ladder varies.
    """

    block_t: int
    rank_cap: int
    block_di: int
    num_stages: int
    shape: KernelShapeParams

    @property
    def block_rk(self) -> int:
        return _rank_block(self.shape.token_rank_k, self.rank_cap)

    @property
    def block_fk(self) -> int:
        return _rank_block(self.shape.feature_rank_k, self.rank_cap)

    @property
    def block_d(self) -> int:
        return max(16, _next_pow2(self.shape.head_dim))

    @property
    def block_h(self) -> int:
        return max(16, _next_pow2(self.shape.n_rep))

    @property
    def block_di_eff(self) -> int:
        """Limit the contraction tile to the padded feature width."""
        return min(self.block_di, self.block_d)

    @property
    def block_rv_full(self) -> int:
        """Return the padded, non-tunable V token-rank width."""
        return max(16, _next_pow2(self.shape.token_rank_v))


def estimate_shared_mem_bytes(config: LaunchConfig) -> int:
    """Estimate the kernel's conservative fp32 shared-memory footprint."""
    bt, bd, bh = config.block_t, config.block_d, config.block_h
    brk, bfk = config.block_rk, config.block_fk
    bdi = config.block_di_eff
    brv = config.block_rv_full

    pipelined_elems = (
        bt * brk  # u_k_blk
        + brk * bfk  # core_k_blk
    )
    single_buffered_elems = (
        2 * bd * bfk  # ff_k_blk + ff_k_shift_blk
        + 3 * bdi * bd  # rot_k_chunk + rot_k_shift_chunk + rot_v_chunk
        + 2 * bt * bd  # cos_blk + sin_blk
        + bt * brv  # u_v_blk
        + bh * bt  # scores
    )
    total_elems = config.num_stages * pipelined_elems + single_buffered_elems
    return total_elems * _FP32_BYTES


# Fixed ladder; inputs always select the first fitting rung.
_DEFAULT_LADDER: tuple[tuple[int, int, int, int], ...] = (
    (_PRODUCTION_NUM_STAGES, _PRODUCTION_RANK_CAP, _PRODUCTION_BLOCK_DI, _PRODUCTION_BLOCK_T),  # rung 0
    (1, _PRODUCTION_RANK_CAP, _PRODUCTION_BLOCK_DI, _PRODUCTION_BLOCK_T),  # rung 1: num_stages 2 -> 1
    (1, _PRODUCTION_RANK_CAP, _PRODUCTION_BLOCK_DI, 16),  # rung 2: also halve the token block
    (1, 16, _PRODUCTION_BLOCK_DI, 16),  # rung 3: also halve the rank-block cap
    (1, 16, 16, 16),  # rung 4: also halve the residual-rotation chunk
)


@dataclass(frozen=True)
class SelectedLaunchConfig:
    """Result of `select_launch_config`: the chosen `LaunchConfig`, its
    estimated footprint, the limit it was checked against, and which rung of
    the ladder it came from (0 == current production defaults)."""

    config: LaunchConfig
    estimated_bytes: int
    limit_bytes: int
    rung_index: int


def select_launch_config(
    shape: KernelShapeParams,
    limit_bytes: int,
    ladder: tuple[tuple[int, int, int, int], ...] = _DEFAULT_LADDER,
) -> SelectedLaunchConfig:
    """Walk `ladder` in order and return the first rung whose estimated shared-memory requirement fits under `limit_bytes`."""
    if not ladder:
        raise RuntimeError(
            "FlashJoLT fused decode kernel: select_launch_config called with an "
            "empty ladder -- there is no rung to try, so no launch configuration "
            "can be selected."
        )
    last_config: LaunchConfig | None = None
    last_bytes: int | None = None
    for rung_index, (num_stages, rank_cap, block_di, block_t) in enumerate(ladder):
        config = LaunchConfig(
            block_t=block_t,
            rank_cap=rank_cap,
            block_di=block_di,
            num_stages=num_stages,
            shape=shape,
        )
        needed = estimate_shared_mem_bytes(config)
        logger.info(
            "flashjolt fused-decode launch-config rung %d: num_stages=%d rank_cap=%d "
            "block_di=%d block_t=%d -> estimated %d bytes (limit %d bytes)",
            rung_index, num_stages, rank_cap, block_di, block_t, needed, limit_bytes,
        )
        if needed <= limit_bytes:
            return SelectedLaunchConfig(
                config=config, estimated_bytes=needed, limit_bytes=limit_bytes, rung_index=rung_index,
            )
        last_config, last_bytes = config, needed

    raise RuntimeError(
        "FlashJoLT fused decode kernel: no launch configuration in the "
        f"{len(ladder)}-rung ladder fits the device's shared-memory limit. "
        f"Smallest rung tried (num_stages={last_config.num_stages}, "
        f"rank_cap={last_config.rank_cap}, block_di={last_config.block_di}, "
        f"block_t={last_config.block_t}) needs an estimated {last_bytes} bytes; "
        f"the device provides {limit_bytes} bytes. Shape: "
        f"token_rank_k={shape.token_rank_k}, feature_rank_k={shape.feature_rank_k}, "
        f"token_rank_v={shape.token_rank_v}, head_dim={shape.head_dim}, n_rep={shape.n_rep}. "
        "This is a hard failure by design: the launch-config selector never silently "
        "falls back to a different attention path -- that decision belongs solely to "
        "kv_compress/integration/flashjolt_attention.py's preflight gate."
    )


# Reduce-kernel launch configuration.


@dataclass(frozen=True)
class ReduceKernelShapeParams:
    """Fixed problem shape for one `_fused_decode_reduce_kernel` launch --
    the values the reduce ladder is NOT allowed to change (changing any would
    change which values are computed, not just how they are tiled). Mirrors
    `KernelShapeParams` for the split kernel."""

    token_rank_v: int
    feature_rank_v: int
    head_dim: int
    n_rep: int


@dataclass(frozen=True)
class ReduceLaunchConfig:
    """One fully-resolved tile/stage configuration for `_fused_decode_reduce_kernel`."""

    num_stages: int
    feature_rank_cap: int
    shape: ReduceKernelShapeParams

    @property
    def block_rv_full(self) -> int:
        """Full padded token-rank-V width -- NEVER reduced by the ladder (same
        genuine problem-shape constant as `LaunchConfig.block_rv_full`; see
        its docstring)."""
        return max(16, _next_pow2(self.shape.token_rank_v))

    @property
    def block_fv(self) -> int:
        """`_rank_block` for the V feature-rank axis, parameterised by the rung's `feature_rank_cap` so a lower rung can chunk it finer (the reduce kernel's feature loop is `for f0 in range(0, feature_rank_v, BLOCK_FV)` -- a smaller cap is more chunks)."""
        return _rank_block(self.shape.feature_rank_v, self.feature_rank_cap)

    @property
    def block_d(self) -> int:
        return max(16, _next_pow2(self.shape.head_dim))

    @property
    def block_h(self) -> int:
        return max(16, _next_pow2(self.shape.n_rep))


def estimate_reduce_shared_mem_bytes(config: ReduceLaunchConfig) -> int:
    """Conservative estimate of `_fused_decode_reduce_kernel`'s dynamic shared-memory footprint in bytes (fp32, 4 bytes/element), modelling the V-side-absorption EXPANSION loop -- the reduce kernel's shared-memory- dominant region and the one that overflowed at the R3/`ctx_len=32768`/ `q_cap=512` shape."""
    brv = config.block_rv_full
    bfv = config.block_fv
    bd = config.block_d
    bh = config.block_h

    pipelined_elems = (
        brv * bfv  # core_blk
        + bd * bfv  # ff_blk
    )
    single_buffered_elems = (
        bh * brv  # s_merged
        + bh * bfv  # mid
    )
    total_elems = config.num_stages * pipelined_elems + single_buffered_elems
    return total_elems * _FP32_BYTES


# Fixed reduce-kernel ladder; shape-derived blocks are not varied.
_DEFAULT_REDUCE_LADDER: tuple[tuple[int, int], ...] = (
    (_PRODUCTION_NUM_STAGES, _PRODUCTION_RANK_CAP),  # rung 0: production defaults
    (1, _PRODUCTION_RANK_CAP),  # rung 1: num_stages 2 -> 1
    (1, 16),  # rung 2: also halve the feature-rank block cap
)


@dataclass(frozen=True)
class SelectedReduceLaunchConfig:
    """Result of `select_reduce_launch_config`: the chosen
    `ReduceLaunchConfig`, its estimated footprint, the limit it was checked
    against, and which rung of the ladder it came from (0 == production
    defaults)."""

    config: ReduceLaunchConfig
    estimated_bytes: int
    limit_bytes: int
    rung_index: int


def select_reduce_launch_config(
    shape: ReduceKernelShapeParams,
    limit_bytes: int,
    ladder: tuple[tuple[int, int], ...] = _DEFAULT_REDUCE_LADDER,
) -> SelectedReduceLaunchConfig:
    """Walk `ladder` in order and return the first rung whose estimated reduce-kernel shared-memory requirement fits under `limit_bytes`."""
    if not ladder:
        raise RuntimeError(
            "FlashJoLT fused decode reduce kernel: select_reduce_launch_config "
            "called with an empty ladder -- there is no rung to try, so no launch "
            "configuration can be selected."
        )
    last_config: ReduceLaunchConfig | None = None
    last_bytes: int | None = None
    for rung_index, (num_stages, feature_rank_cap) in enumerate(ladder):
        config = ReduceLaunchConfig(
            num_stages=num_stages,
            feature_rank_cap=feature_rank_cap,
            shape=shape,
        )
        needed = estimate_reduce_shared_mem_bytes(config)
        logger.info(
            "flashjolt fused-decode REDUCE launch-config rung %d: num_stages=%d "
            "feature_rank_cap=%d -> estimated %d bytes (limit %d bytes)",
            rung_index, num_stages, feature_rank_cap, needed, limit_bytes,
        )
        if needed <= limit_bytes:
            return SelectedReduceLaunchConfig(
                config=config, estimated_bytes=needed, limit_bytes=limit_bytes, rung_index=rung_index,
            )
        last_config, last_bytes = config, needed

    raise RuntimeError(
        "FlashJoLT fused decode reduce kernel: no launch configuration in the "
        f"{len(ladder)}-rung reduce ladder fits the device's shared-memory limit. "
        f"Smallest rung tried (num_stages={last_config.num_stages}, "
        f"feature_rank_cap={last_config.feature_rank_cap}) needs an estimated "
        f"{last_bytes} bytes; the device provides {limit_bytes} bytes. Shape: "
        f"token_rank_v={shape.token_rank_v}, feature_rank_v={shape.feature_rank_v}, "
        f"head_dim={shape.head_dim}, n_rep={shape.n_rep}. This is a hard failure by "
        "design: the launch-config selector never silently falls back to a different "
        "attention path -- that decision belongs solely to "
        "kv_compress/integration/flashjolt_attention.py's preflight gate."
    )


def query_device_shared_mem_limit(device: "torch.device") -> int:
    """Query the real per-block opt-in dynamic shared-memory ceiling for `device` -- the same figure Triton's own `OutOfResources` error reports as "Hardware limit" -- via `torch.cuda.get_device_properties(device)."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(device)
            limit = getattr(props, "shared_memory_per_block_optin", None)
            if limit:
                return int(limit)
        except Exception:
            logger.warning(
                "flashjolt: failed to query the device's shared-memory limit; "
                "falling back to the documented A100 constant (%d bytes)",
                A100_SHARED_MEM_LIMIT_BYTES,
                exc_info=True,
            )
    return A100_SHARED_MEM_LIMIT_BYTES
