"""KV cache compression via Tucker decomposition and layer grouping.

Device behavior: Tensors remain on their source device throughout the pipeline.
If input tensors are on GPU, HOSVD and Tucker fitting run on GPU (~6× faster
than CPU). No CPU↔GPU transfers are performed.

Example:
    >>> from kv_extraction import KVExtractor
    >>> from kv_compress import compress, CompressionConfig
    >>> extractor = KVExtractor(model)
    >>> layers = extractor.extract(input_ids)  # tensors on model device
    >>> cfg = CompressionConfig(target_compression_ratio=4.0)
    >>> result = compress(layers, cfg)  # runs on GPU if layers are on GPU
    >>> print(result.summary())
"""

from .config import CompressionConfig
from .pipeline import compress
from .reporting.metrics import CompressionResult

__all__ = [
    "compress",
    "CompressionConfig",
    "CompressionResult",
    "CompressedKVCache",
    "CompressedCacheConfig",
]


def __getattr__(name: str):
    # Lazy import so that consumers who only need offline compression do not
    # pay the transformers import cost at module load.
    if name in ("CompressedKVCache", "CompressedCacheConfig"):
        from .integration import CompressedCacheConfig, CompressedKVCache

        return {"CompressedKVCache": CompressedKVCache, "CompressedCacheConfig": CompressedCacheConfig}[name]
    raise AttributeError(f"module 'kv_compress' has no attribute {name!r}")
