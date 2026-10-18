# Vendored: kv_compress (JoLT)

Source: the author's JoLT codebase (kv_compress package)
Author's own JoLT code (paper "Joint Lagrangian Tucker", arXiv, same author).
Vendored 2026-10-18. The original is unchanged.

Dropped: `__pycache__`, `flashjolt/.docmesh` (tool state). Everything else kept
(`flashjolt/layout.py` is imported by `compressed_cache.py` at runtime).

Local modifications:
- `integration/pre_rope_capture.py`: `_make_key_norm_hook` accepts the 4-D
  `(B, T, n_kv, head_dim)` output of a per-head k_norm (Qwen3) and captures it
  transposed to `(B, n_kv, T, head_dim)`. The OLMoE 3-D path is unchanged; any
  other shape still raises RuntimeError.
