"""JoLT (Joint Lagrangian Tucker) KV cache for batch-1 forward calls."""

from __future__ import annotations

from kv_compress.integration import CompressedCacheConfig, CompressedKVCache


def make_jolt_cache(model, target_ratio: float) -> CompressedKVCache:
    """Return a JoLT cache with the default method config at `target_ratio`.

    Call pattern: one forward with all prompt tokens except the last stores the
    prompt; the next forward (the last prompt token) compresses it, and every
    later forward attends to reconstructed prompt KV plus a dense tail. After
    that forward the cache exposes last_compression_ratio, last_compressed_bytes
    and last_original_bytes. The cache hooks `model` for pre-RoPE key capture;
    call `cache.detach()` when finished.
    """
    return CompressedKVCache(
        CompressedCacheConfig(target_compression_ratio=float(target_ratio)), model=model
    )
