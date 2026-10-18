"""FlashJoLT fused attention over Tucker and JL-compressed KV.

Submodules define the tensor contract, independent references, bit unpacking,
Triton kernels, and deterministic shared-memory launch selection. Triton is
optional; CPU fallbacks remain importable without a GPU.
"""

from __future__ import annotations
