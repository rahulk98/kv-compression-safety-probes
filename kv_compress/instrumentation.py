"""Opt-in instrumentation for q_cap allocation and compression timing."""

from __future__ import annotations

import contextvars
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

# Session state is context-local for async and threaded callers.
_ACTIVE: contextvars.ContextVar[list[dict] | None] = contextvars.ContextVar(
    "kv_compress_instrumentation_active", default=None
)
# Raw mode-2 events, in call order.
_MODE2_LOG: contextvars.ContextVar[list[dict] | None] = contextvars.ContextVar(
    "kv_compress_instrumentation_mode2_log", default=None
)
# Raw allocation-objective events, in group order.
_ALLOC_LOG: contextvars.ContextVar[list[dict] | None] = contextvars.ContextVar(
    "kv_compress_instrumentation_alloc_log", default=None
)


def is_active() -> bool:
    """The single cheap check every instrumentation call site guards on.

    A ``contextvars.ContextVar.get()`` is an O(1) dict-like lookup: no
    allocation, no torch call, no CUDA sync. Safe to call on every
    invocation of a hot path."""
    return _ACTIVE.get() is not None


@contextmanager
def instrumentation_session() -> Iterator[list[dict]]:
    """Activate instrumentation for the duration of the ``with`` block."""
    records: list[dict] = []
    mode2_log: list[dict] = []
    alloc_log: list[dict] = []
    tok_records = _ACTIVE.set(records)
    tok_mode2 = _MODE2_LOG.set(mode2_log)
    tok_alloc = _ALLOC_LOG.set(alloc_log)
    try:
        yield records
    finally:
        _ACTIVE.reset(tok_records)
        _MODE2_LOG.reset(tok_mode2)
        _ALLOC_LOG.reset(tok_alloc)


def record_mode2_call(*, q_cap: int, q_actual: int, niter: int, elapsed_s: float) -> None:
    """Called by ``Tucker2PartialMode2LowRank._lowrank_mode2``, guarded by ``is_active()`` at the call site (this function itself does not re-check, to avoid a second contextvar read on the hot path)."""
    log = _MODE2_LOG.get()
    if log is not None:
        log.append(
            {"q_cap": int(q_cap), "q_actual": int(q_actual), "niter": int(niter),
             "elapsed_s": float(elapsed_s)}
        )


def record_allocation_objective(
    *, lam: float, ranks: tuple[int, ...], bits: int, objective: float
) -> None:
    """Called by ``JointTuckerJLAllocator._run_joint`` once per group at
    convergence, guarded by ``is_active()`` at the call site."""
    log = _ALLOC_LOG.get()
    if log is not None:
        log.append(
            {"lam": float(lam), "ranks": tuple(int(r) for r in ranks),
             "bits": int(bits), "objective": float(objective)}
        )


def pop_mode2_log() -> list[dict]:
    """Drain and return every mode-2 call recorded so far in the active
    session (empty list if inactive or nothing recorded yet)."""
    log = _MODE2_LOG.get()
    if log is None:
        return []
    drained = list(log)
    log.clear()
    return drained


def pop_allocation_log() -> list[dict]:
    """Drain and return every allocation-objective event recorded so far in
    the active session."""
    log = _ALLOC_LOG.get()
    if log is None:
        return []
    drained = list(log)
    log.clear()
    return drained


def emit(record: dict) -> None:
    """Append one finished record to the active session's output list.
    No-op if no session is active."""
    sink = _ACTIVE.get()
    if sink is not None:
        sink.append(record)



_GIT_COMMIT_CACHE: str | None | Any = "_unset"


def git_commit() -> str | None:
    """The current HEAD commit hash, cached after the first call. Returns
    None (never raises) if git is unavailable or this is not a git
    checkout."""
    global _GIT_COMMIT_CACHE
    if _GIT_COMMIT_CACHE == "_unset":
        try:
            _GIT_COMMIT_CACHE = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(Path(__file__).resolve().parent),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            _GIT_COMMIT_CACHE = None
    return _GIT_COMMIT_CACHE


def gpu_and_version_info() -> dict[str, Any]:
    """PyTorch version (always available) plus CUDA version / GPU model
    (None when CUDA is unavailable -- e.g. this Mac -- never raises)."""
    info: dict[str, Any] = {
        "pytorch_version": torch.__version__,
        "cuda_version": None,
        "gpu_model": None,
    }
    if torch.cuda.is_available():
        try:
            info["cuda_version"] = torch.version.cuda
            info["gpu_model"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return info


def peak_memory_bytes(device: "torch.device | str | None") -> tuple[int | None, int | None]:
    """``(peak_allocated_bytes, peak_reserved_bytes)`` for ``device``, or
    ``(None, None)`` off CUDA / when ``device`` is None. Never raises."""
    if device is None:
        return None, None
    try:
        dev = torch.device(device)
    except Exception:
        return None, None
    if dev.type != "cuda" or not torch.cuda.is_available():
        return None, None
    try:
        return (
            int(torch.cuda.max_memory_allocated(dev)),
            int(torch.cuda.max_memory_reserved(dev)),
        )
    except Exception:
        return None, None




def finalize_call(
    *,
    groups_k: list,
    groups_v: list,
    ranks_k: list[tuple[int, ...]],
    ranks_v: list[tuple[int, ...]],
    bits_k: list[int],
    bits_v: list[int],
    recon_k: list["torch.Tensor"] | None = None,
    recon_v: list["torch.Tensor"] | None = None,
    q_cap: int,
    dtype_bytes: int,
    target_ratio: float,
    achieved_ratio: float | None,
    persistent_bytes: int,
    seed: int | None = None,
    niter: int | None = None,
    full_compression_path_runtime_s: float | None = None,
    allocator_runtime_s: float | None = None,
    peak_allocated_bytes: int | None = None,
    peak_reserved_bytes: int | None = None,
    context: dict | None = None,
    recon_k_full: list["torch.Tensor"] | None = None,
    recon_v_full: list["torch.Tensor"] | None = None,
    tucker_only_errors: tuple[float | None, float | None, float | None] | None = None,
    pipeline_errors: tuple[float | None, float | None, float | None] | None = None,
) -> list[dict]:
    """Assemble finished per-group records from the raw mode2/allocation logs plus already-computed call-level quantities, and :func:`emit` them to the active session."""
    if not is_active():
        return []

    mode2_events = pop_mode2_log()
    alloc_events = pop_allocation_log()
    n_groups = len(groups_k) + len(groups_v)
    if len(mode2_events) != n_groups:
        raise RuntimeError(
            f"instrumentation.finalize_call: expected {n_groups} mode-2 SVD "
            f"events (len(groups_k)+len(groups_v)), got {len(mode2_events)} "
            "-- the ordering contract between JointTuckerJLAllocator._run_joint "
            "and this assembly step is broken; refusing to guess which event "
            "belongs to which group (see this function's docstring)."
        )
    if len(alloc_events) != n_groups:
        raise RuntimeError(
            f"instrumentation.finalize_call: expected {n_groups} allocation "
            f"events, got {len(alloc_events)} -- see mode-2 event count error "
            "above for the same ordering-contract concern."
        )

    ctx = context or {}
    # Use streamed Tucker-only errors when available.
    if tucker_only_errors is not None:
        eK_tucker_only, eV_tucker_only, combined_tucker_only = tucker_only_errors
    else:
        eK_tucker_only, eV_tucker_only, combined_tucker_only = _combined_errors(
            groups_k, groups_v, recon_k, recon_v
        )
    # Post-residual fields remain None when post-residual reconstructions are absent.
    if pipeline_errors is not None:
        eK, eV, combined = pipeline_errors
    elif recon_k_full is not None and recon_v_full is not None:
        eK, eV, combined = _combined_errors(groups_k, groups_v, recon_k_full, recon_v_full)
    else:
        eK = eV = combined = None
    # Sum unsynchronized mode-2 host timings.
    mode2_runtime_s = sum(float(e["elapsed_s"]) for e in mode2_events)

    common = {
        "git_commit": git_commit(),
        "model": ctx.get("model"),
        "model_revision": ctx.get("model_revision"),
        "dataset": ctx.get("dataset"),
        "sample_id": ctx.get("sample_id"),
        "target_ratio": float(target_ratio),
        "achieved_ratio": (float(achieved_ratio) if achieved_ratio is not None else None),
        "persistent_bytes": int(persistent_bytes),
        # Post-residual error, when supplied.
        "eK": eK,
        "eV": eV,
        "combined_pipeline_error": combined,
        # Tucker-only diagnostic.
        "eK_tucker_only": eK_tucker_only,
        "eV_tucker_only": eV_tucker_only,
        "combined_tucker_only_error": combined_tucker_only,
        "seed": seed,
        "niter": niter,
        "dtype_bytes": int(dtype_bytes),
        "full_compression_path_runtime_s": full_compression_path_runtime_s,
        "mode2_runtime_s": mode2_runtime_s,
        "allocator_runtime_s": allocator_runtime_s,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        **gpu_and_version_info(),
    }

    records: list[dict] = []
    sides = [("k", groups_k, ranks_k, bits_k), ("v", groups_v, ranks_v, bits_v)]
    idx = 0
    for kind, groups, ranks, bits in sides:
        for g_idx, group in enumerate(groups):
            m2 = mode2_events[idx]
            alloc = alloc_events[idx]
            idx += 1
            r2 = int(ranks[g_idx][2])
            q_actual = int(m2["q_actual"])
            record = dict(common)
            record.update(
                {
                    "kind": kind,
                    "group_index": g_idx,
                    "group_size": int(group.shape[0]),
                    "seq_len": int(group.shape[-2]),
                    "q_cap": int(q_cap),
                    "q_actual": q_actual,
                    "r2": r2,
                    "residual_bits": int(bits[g_idx]),
                    "r2_eq_qcap": bool(r2 == q_cap),
                    "zero_oversampling": bool(q_actual - r2 == 0),
                    "allocation_objective": alloc["objective"],
                    "allocation_lambda": alloc["lam"],
                }
            )
            records.append(record)
            emit(record)
    return records


class StreamingErrorAccumulator:
    """Streaming counterpart to :func:`_combined_errors`: folds each group's contribution into running scalar sums as it becomes available, instead of requiring every group's original + reconstruction tensor to be retained in a list until the whole call is done."""

    def __init__(self) -> None:
        self._num_k = 0.0
        self._sum_k = 0.0
        self._num_v = 0.0
        self._sum_v = 0.0

    def add_k(self, X: "torch.Tensor", X_hat: "torch.Tensor") -> None:
        # Match pipeline.py by computing norms in fp32.
        X = X.to(torch.float32)
        X_hat = X_hat.to(torch.float32)
        self._num_k += float(torch.linalg.vector_norm(X - X_hat) ** 2)
        self._sum_k += float(torch.linalg.vector_norm(X) ** 2)

    def add_v(self, X: "torch.Tensor", X_hat: "torch.Tensor") -> None:
        X = X.to(torch.float32)
        X_hat = X_hat.to(torch.float32)
        self._num_v += float(torch.linalg.vector_norm(X - X_hat) ** 2)
        self._sum_v += float(torch.linalg.vector_norm(X) ** 2)

    def finalize(self) -> tuple[float | None, float | None, float | None]:
        eK = (self._num_k / self._sum_k) ** 0.5 if self._sum_k > 0 else None
        eV = (self._num_v / self._sum_v) ** 0.5 if self._sum_v > 0 else None
        denom = self._sum_k + self._sum_v
        combined = (
            ((self._num_k + self._num_v) / denom) ** 0.5 if denom > 0 else None
        )
        return eK, eV, combined


def _combined_errors(
    groups_k: list, groups_v: list, recon_k: list, recon_v: list
) -> tuple[float | None, float | None, float | None]:
    """Whole-cache (all groups summed) relative Frobenius error between `groups_k`/`groups_v` (originals) and `recon_k`/`recon_v` (reconstructions), for K and V separately (eK, eV) and combined (K+V together, one scalar)."""
    acc = StreamingErrorAccumulator()
    for X, X_hat in zip(groups_k, recon_k):
        acc.add_k(X, X_hat)
    for X, X_hat in zip(groups_v, recon_v):
        acc.add_v(X, X_hat)
    return acc.finalize()
