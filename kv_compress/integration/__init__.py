"""HuggingFace Transformers integration for kv_compress.

Plug the Tucker + JL residual compression library into `model.generate()` by
passing a `CompressedKVCache` as `past_key_values=`. The cache subclasses
`transformers.cache_utils.Cache` and compresses K/V between forward calls.

Memory semantics, default dense-reconstruction path (`use_fused_decode=False`,
the default):
  - Persistent KV cache between forwards drops at the configured ratio.
  - Peak transient memory during a forward matches baseline because
    reconstruction materialises the full K/V tensor.
  - Peak savings on this path require the opt-in fused decode path
    (`CompressedCacheConfig.use_fused_decode=True`, see
    `flashjolt_attention.py`), which bypasses dense reconstruction and keeps
    `_recon_cache_{k,v,layer}` empty. It ships in this same package, is
    A100-parity-certified (`results.md` section 13.3), and defaults OFF, so
    every existing consumer of the dense path is byte-for-byte unchanged.

Example:
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer
    >>> from kv_compress.integration import CompressedKVCache, CompressedCacheConfig
    >>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3")
    >>> tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    >>> cache = CompressedKVCache(CompressedCacheConfig(target_compression_ratio=4.0), model=model)
    >>> ids = tok("Hello world", return_tensors="pt").input_ids.to(model.device)
    >>> out = model.generate(ids, past_key_values=cache, max_new_tokens=64)
    >>> cache.detach()
"""

from .config import CompressedCacheConfig
from .compressed_cache import CompressedKVCache

__all__ = ["CompressedKVCache", "CompressedCacheConfig"]
