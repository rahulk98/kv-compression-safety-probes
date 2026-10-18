"""CompressedKVCache: HuggingFace `Cache` that stores K/V via Tucker + JL residual."""

from __future__ import annotations

from typing import NamedTuple

import torch

from ..allocation.base import BudgetSpec
from ..allocation.lagrangian_joint import JointTuckerJLAllocator
from ..decompositions.tucker2_partial import Tucker2Partial
from ..decompositions.tucker2_partial_lowrank import (
    Tucker2PartialMode2LowRank,
    resolve_q_cap,
)
from ..flashjolt.layout import (
    ResidualPayload,
    TuckerGroupFactors,
    adopt_canonical_tucker_factors,
    canonicalize_residual_payload,
)
from ..grouping import parse_grouping_id
from ..residuals.jl_rotated import JLRotatedResidual
from .config import CompressedCacheConfig
from .pre_rope_capture import PreRoPECapture, apply_rope_to_k


class FlashJoltFactorHandle(NamedTuple):
    """FlashJoLT fused-decode-attention factor handle for one layer: the already-fitted Tucker K/V factors and JL residual payloads, canonicalized to the frozen `kv_compress/flashjolt/layout.py` contract, plus the group slot -- exactly what `kv_compress.flashjolt.kernel.fused_decode_attention` needs."""

    tucker_k: TuckerGroupFactors
    residual_k: ResidualPayload | None
    tucker_v: TuckerGroupFactors
    residual_v: ResidualPayload | None
    slot: int


try:
    from transformers.cache_utils import Cache as _HFCache
except Exception:  # pragma: no cover - transformers missing in dev shells
    _HFCache = object  # type: ignore[misc,assignment]


# Avoid persistent dense layer reconstructions at long context.
LAYER_RECON_CACHE_MAX_T = 8192


class CompressedKVCache(_HFCache):
    """Cache that compresses the prefill K/V at first decode and keeps an uncompressed tail buffer of decoded tokens."""

    def __init__(
        self,
        config: CompressedCacheConfig,
        model=None,
        attach: bool = True,
    ):
        # Support both Transformers v4 and v5 Cache constructors.
        try:
            super().__init__(layers=[])
        except TypeError:
            super().__init__()
        self.config = config
        self.model = model

        # Allow manual K/V injection when no model is attached.
        self._capture: PreRoPECapture | None = None
        if model is not None and attach:
            self._capture = PreRoPECapture(model)
            self._capture.attach()

        # Install the optional adapter lazily to avoid an import cycle.
        self._flashjolt_installation = None
        if model is not None and attach and config.use_fused_decode:
            from .flashjolt_attention import install as _fj_install
            try:
                self._flashjolt_installation = _fj_install(model, self)
            except Exception:
                # Roll back capture hooks if installation fails.
                if self._capture is not None:
                    self._capture.detach()
                    self._capture = None
                raise

        self._phase: str = "fresh"
        self._num_layers: int | None = None
        self._prefill_seen: set[int] = set()

        self._prefill_k_pre: dict[int, torch.Tensor] = {}  # (B, n_kv, T, d_h)
        self._prefill_v: dict[int, torch.Tensor] = {}      # (B, n_kv, T, d_h)
        self._prefill_seq_len: int = 0
        self._prefill_cos: torch.Tensor | None = None
        self._prefill_sin: torch.Tensor | None = None

        self._approx_k: list = []  # list[Tucker4DApproximation], per group
        self._approx_v: list = []
        self._residual_k: dict[int, object] = {}  # layer_idx -> EncodedResidual | None
        self._residual_v: dict[int, object] = {}
        self._bits_k: list[int] = []
        self._bits_v: list[int] = []
        self._groups: list[list[int]] = []
        self._layer_to_group_slot: dict[int, tuple[int, int]] = {}
        self._enc_cache: dict[int, JLRotatedResidual] = {}

        # Group-level Tucker reconstructions, released after layer caches are built.
        self._recon_cache_k: dict[int, torch.Tensor] = {}
        self._recon_cache_v: dict[int, torch.Tensor] = {}
        # Final post-RoPE K/V cache in attention dtype, reused during decode.
        self._recon_cache_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # Group index to layers awaiting decode-ready entries.
        self._group_layers_pending: dict[int, set[int]] = {}
        self._cos_sin_cast: (
            tuple[torch.dtype, torch.device, torch.Tensor, torch.Tensor] | None
        ) = None

        # Decode tail: post-RoPE k and v for tokens generated after prefill.
        self._tail_k: dict[int, torch.Tensor] = {}
        self._tail_v: dict[int, torch.Tensor] = {}

        self.last_compression_ratio: float | None = None
        self.last_compressed_bytes: int | None = None
        self.last_original_bytes: int | None = None
        # Resolved q_cap used by the fit, if applicable.
        self.last_q_cap: int | None = None


    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key_states.shape[0] != 1:
            raise NotImplementedError(
                "CompressedKVCache v1 supports batch size 1 only; "
                f"got batch={key_states.shape[0]}"
            )

        # Layer 0 marks each forward and detects the prefill-to-decode transition.
        if layer_idx == 0:
            self.prepare_decode_transition()

        if self._phase in ("fresh", "prefill_stashed"):
            return self._handle_prefill_layer(key_states, value_states, layer_idx)

        return self._handle_decode_layer(key_states, value_states, layer_idx)

    def prepare_decode_transition(self) -> bool:
        """Explicit prefill->decode compression trigger."""
        if self._phase == "prefill_stashed" and 0 in self._prefill_seen:
            self._compress_prefill()
            self._recon_cache_k.clear()
            self._recon_cache_v.clear()
            self._recon_cache_layer.clear()
            self._cos_sin_cast = None
            return True
        return False

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if self._phase == "fresh":
            return 0
        tail_len = self._tail_k[layer_idx].shape[-2] if layer_idx in self._tail_k else 0
        return self._prefill_seq_len + tail_len

    def get_max_length(self) -> int | None:
        return None  # dynamic, no preallocation

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        # Dynamic cache has no fixed maximum.
        return -1

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        # Return full cached length and zero rolling-window offset.
        kv_length = self.get_seq_length(layer_idx) + query_length
        return kv_length, 0

    def early_initialization(self, *args, **kwargs) -> None:
        # State is lazy; nothing is preallocated.
        return

    def reset(self) -> None:
        # Clear cache state while keeping hooks attached.
        if self._capture is not None:
            self._capture.begin_forward()
        # Reset per-prompt fused state without removing the installation.
        if getattr(self, "_flashjolt_installation", None) is not None:
            self._flashjolt_installation.clear_prompt_state()
        self._phase = "fresh"
        self._num_layers = None
        self._prefill_seen.clear()
        self._prefill_k_pre.clear()
        self._prefill_v.clear()
        self._prefill_seq_len = 0
        self._prefill_cos = None
        self._prefill_sin = None
        self._approx_k.clear()
        self._approx_v.clear()
        self._residual_k.clear()
        self._residual_v.clear()
        self._bits_k.clear()
        self._bits_v.clear()
        self._groups.clear()
        self._layer_to_group_slot.clear()
        self._recon_cache_k.clear()
        self._recon_cache_v.clear()
        self._recon_cache_layer.clear()
        self._group_layers_pending.clear()
        self._cos_sin_cast = None
        self._tail_k.clear()
        self._tail_v.clear()
        self.last_compression_ratio = None
        self.last_compressed_bytes = None
        self.last_original_bytes = None
        self.last_q_cap = None

    def __len__(self) -> int:
        return self._num_layers or 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        raise NotImplementedError(
            "CompressedKVCache does not support beam search reordering in v1"
        )


    def detach(self) -> None:
        """Remove pre-RoPE capture hooks and any fused-decode adapter from the model."""
        if self._capture is not None:
            self._capture.detach()
            self._capture = None
        if self._flashjolt_installation is not None:
            self._flashjolt_installation.uninstall()
            self._flashjolt_installation = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.detach()

    def has_persistent_dense_recon(self) -> bool:
        """True iff any persistent dense K/V reconstruction cache is currently populated (`_recon_cache_k` / `_recon_cache_v` / `_recon_cache_layer`)."""
        return bool(self._recon_cache_k or self._recon_cache_v or self._recon_cache_layer)

    def compression_stats(self) -> dict:
        if self._phase != "compressed":
            return {"phase": self._phase}
        return {
            "phase": self._phase,
            "compression_ratio": self.last_compression_ratio,
            "compressed_bytes": self.last_compressed_bytes,
            "original_bytes": self.last_original_bytes,
            "groups": list(self._groups),
            "bits_k": list(self._bits_k),
            "bits_v": list(self._bits_v),
            "prefill_seq_len": self._prefill_seq_len,
        }


    def _handle_prefill_layer(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Prefer captured pre-RoPE K; manual callers provide the fallback.
        if self._capture is not None:
            try:
                k_pre = self._capture.pop_pre_rope_k(layer_idx)
            except KeyError as exc:
                raise RuntimeError(
                    f"pre-RoPE K not captured for layer {layer_idx}; "
                    "make sure CompressedKVCache.attach() ran before model.generate()"
                ) from exc
        else:
            # Manual test path.
            k_pre = key_states  # assumed pre-RoPE in this fallback

        self._prefill_k_pre[layer_idx] = k_pre.detach()
        self._prefill_v[layer_idx] = value_states.detach()
        self._prefill_seen.add(layer_idx)
        self._prefill_seq_len = key_states.shape[-2]
        self._phase = "prefill_stashed"

        if self._num_layers is None or layer_idx + 1 > self._num_layers:
            self._num_layers = layer_idx + 1

        # Capture cos/sin once during prefill.
        if (
            self._capture is not None
            and self._prefill_cos is None
            and layer_idx == 0
        ):
            try:
                cos, sin = self._capture.get_cos_sin()
                self._prefill_cos = cos.detach()
                self._prefill_sin = sin.detach()
            except RuntimeError:
                # Reconstruction reports the missing RoPE tables later.
                pass

        return key_states, value_states


    def _compress_prefill(self) -> None:
        if self._phase != "prefill_stashed":
            return
        if self._prefill_cos is None or self._prefill_sin is None:
            raise RuntimeError(
                "prefill cos/sin not captured; cannot reconstruct pre-RoPE K. "
                "Check the apply_rotary_pos_emb monkey-patch installed for this model."
            )

        layers_for_alloc: dict[int, dict[str, torch.Tensor]] = {}
        for layer_idx in sorted(self._prefill_k_pre):
            k = self._prefill_k_pre[layer_idx]
            v = self._prefill_v[layer_idx]
            layers_for_alloc[layer_idx] = {
                "k": k.squeeze(0).to(torch.float32),
                "v": v.squeeze(0).to(torch.float32),
            }

        # Release the original stash after converting to float32.
        self._prefill_k_pre.clear()
        self._prefill_v.clear()

        cfg = self.config
        grouper = parse_grouping_id(cfg.grouping, alignment_source=cfg.alignment_source)
        groups = grouper.assign(layers_for_alloc)

        k_stacks = [
            torch.stack([layers_for_alloc[i]["k"] for i in g], dim=0) for g in groups
        ]
        v_stacks = [
            torch.stack([layers_for_alloc[i]["v"] for i in g], dim=0) for g in groups
        ]

        # Group stacks now own the data; release the per-layer dictionary.
        from ..allocation.memory import mem_original

        original_bytes = sum(
            mem_original(tuple(layers_for_alloc[i]["k"].shape), cfg.dtype_bytes)
            + mem_original(tuple(layers_for_alloc[i]["v"].shape), cfg.dtype_bytes)
            for i in layers_for_alloc
        )
        layers_for_alloc.clear()

        mode = cfg.budget_mode()
        if mode == "ratio":
            bvalue = float(cfg.target_compression_ratio)
        else:
            bvalue = float(cfg.memory_budget_bytes)

        if cfg.decomposition == "tucker2_partial_lowrank":
            # Resolve auto q_cap from the ratio and stacked token length.
            ratio_for_qcap = bvalue if mode == "ratio" else 4.0
            seq_len = int(k_stacks[0].shape[-2]) if k_stacks else None
            resolved_q_cap = resolve_q_cap(
                ratio_for_qcap, override=cfg.q_cap, seq_len=seq_len
            )
            decomp = Tucker2PartialMode2LowRank(q_cap=resolved_q_cap)
            # Expose the resolved cap separately from the requested override.
            self.last_q_cap = int(resolved_q_cap)
        else:
            decomp = Tucker2Partial()
        allocator = JointTuckerJLAllocator(active_modes=(2, 3), residual_seed=cfg.residual_seed)
        budget = BudgetSpec(mode=mode, value=bvalue, dtype_bytes=cfg.dtype_bytes)

        ranks_k, ranks_v, bits_k, bits_v = allocator.allocate_joint_with_bits(
            k_stacks, v_stacks, budget, decomp
        )

        self._groups = groups
        self._bits_k = list(bits_k)
        self._bits_v = list(bits_v)
        self._approx_k = []
        self._approx_v = []
        self._layer_to_group_slot = {}
        self._group_layers_pending = {
            g_idx: set(group) for g_idx, group in enumerate(groups)
        }

        compressed_bytes = 0

        for g_idx, group in enumerate(groups):
            X_k = k_stacks[g_idx]
            X_v = v_stacks[g_idx]
            r_k = ranks_k[g_idx]
            r_v = ranks_v[g_idx]
            b_k = int(bits_k[g_idx])
            b_v = int(bits_v[g_idx])

            approx_k = decomp.fit(X_k, r_k)
            approx_v = decomp.fit(X_v, r_v)
            # Cast factors before accounting and residual formation so the
            # priced representation matches resident storage.
            if cfg.factor_storage_dtype == "bf16":
                approx_k.cast_storage_bf16_()
                approx_v.cast_storage_bf16_()
            self._approx_k.append(approx_k)
            self._approx_v.append(approx_v)
            compressed_bytes += approx_k.footprint_bytes(cfg.dtype_bytes)
            compressed_bytes += approx_v.footprint_bytes(cfg.dtype_bytes)

            # Reconstruct only when a residual is allocated.
            X_k_recon = approx_k.reconstruct() if b_k > 0 else None  # (L_g, n_kv, T, d_h)
            X_v_recon = approx_v.reconstruct() if b_v > 0 else None

            for slot, layer_idx in enumerate(group):
                self._layer_to_group_slot[layer_idx] = (g_idx, slot)

                if b_k > 0:
                    enc = self._get_encoder(b_k)
                    E_k = X_k[slot] - X_k_recon[slot]
                    encoded = enc.encode(E_k)
                    self._residual_k[layer_idx] = encoded
                    compressed_bytes += encoded.nbytes
                else:
                    self._residual_k[layer_idx] = None

                if b_v > 0:
                    enc = self._get_encoder(b_v)
                    E_v = X_v[slot] - X_v_recon[slot]
                    encoded = enc.encode(E_v)
                    self._residual_v[layer_idx] = encoded
                    compressed_bytes += encoded.nbytes
                else:
                    self._residual_v[layer_idx] = None

            # Release each completed group's transient stacks.
            k_stacks[g_idx] = None
            v_stacks[g_idx] = None

        self.last_original_bytes = int(original_bytes)
        self.last_compressed_bytes = int(compressed_bytes)
        self.last_compression_ratio = (
            float(original_bytes) / float(compressed_bytes) if compressed_bytes > 0 else float("inf")
        )

        self._phase = "compressed"

    def _get_encoder(self, bits: int) -> JLRotatedResidual:
        enc = self._enc_cache.get(bits)
        if enc is None:
            enc = JLRotatedResidual(bits=bits, seed=self.config.residual_seed)
            self._enc_cache[bits] = enc
        return enc


    def _handle_decode_layer(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.append_decode_tail(layer_idx, key_states, value_states)

        k_ready, v_ready = self._decode_ready_layer(
            layer_idx, key_states.dtype, key_states.device
        )
        k_full = torch.cat([k_ready, self._tail_k[layer_idx]], dim=-2)
        v_full = torch.cat([v_ready, self._tail_v[layer_idx]], dim=-2)
        return k_full, v_full

    def append_decode_tail(
        self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        """Append one decode step's already-post-RoPE `key_states` and raw `value_states` to layer `layer_idx`'s decode tail."""
        # Build both tails before assignment so a failed concat cannot skew K/V.
        if layer_idx not in self._tail_k:
            new_k = key_states.detach()
            new_v = value_states.detach()
        else:
            new_k = torch.cat([self._tail_k[layer_idx], key_states.detach()], dim=-2)
            new_v = torch.cat([self._tail_v[layer_idx], value_states.detach()], dim=-2)
        self._tail_k[layer_idx] = new_k
        self._tail_v[layer_idx] = new_v

    def decode_tail(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Current (post-append) decode tail K/V for `layer_idx`, shape
        `(1, n_kv, T_tail, head_dim)`. For callers (the FlashJoLT adapter)
        that need the tail AFTER `append_decode_tail` has folded in this
        step's new token, so self-attention over it is included.
        """
        return self._tail_k[layer_idx], self._tail_v[layer_idx]

    def flashjolt_factor_handle(self, layer_idx: int) -> FlashJoltFactorHandle:
        """Build a `FlashJoltFactorHandle` for `layer_idx`: the already-fitted Tucker K/V factors and JL residual payloads, canonicalized to the frozen `kv_compress/flashjolt/layout.py` contract."""
        if self._phase != "compressed":
            raise RuntimeError(
                f"flashjolt_factor_handle: cache phase is {self._phase!r}, expected "
                "'compressed' -- call after prepare_decode_transition() has triggered "
                "compression"
            )
        g_idx, slot = self._layer_to_group_slot[layer_idx]
        approx_k = self._approx_k[g_idx]
        approx_v = self._approx_v[g_idx]
        # Adopt canonical factors to release SVD views and share fit storage.
        tucker_k = adopt_canonical_tucker_factors(approx_k)
        tucker_v = adopt_canonical_tucker_factors(approx_v)
        orig_shape = (tucker_k.num_kv_heads, tucker_k.seq_len, tucker_k.head_dim)
        residual_k = self._flashjolt_residual_payload(
            self._residual_k.get(layer_idx), self._bits_k[g_idx], orig_shape
        )
        residual_v = self._flashjolt_residual_payload(
            self._residual_v.get(layer_idx), self._bits_v[g_idx], orig_shape
        )
        return FlashJoltFactorHandle(
            tucker_k=tucker_k, residual_k=residual_k, tucker_v=tucker_v, residual_v=residual_v, slot=slot
        )

    def _flashjolt_residual_payload(
        self, encoded, bits: int, orig_shape: tuple[int, int, int]
    ) -> ResidualPayload | None:
        """Mirrors `experiments/flashjolt/validate_model.py::
        _residual_payload_from_state`'s reuse pattern: build a
        contract-conformant `ResidualPayload` from a raw `EncodedResidual`,
        regenerating the materialized JL rotation matrix from the SAME
        deterministic-seed encoder (`_rotation` is reused, not
        reimplemented) rather than storing a redundant copy.
        """
        if encoded is None:
            return None
        enc = self._get_encoder(bits)
        p = encoded.payload
        rotation = enc._rotation(orig_shape[-1], p.packed.device, torch.float32)
        return canonicalize_residual_payload(
            packed=p.packed, scale=p.scale, bits=p.bits, numel=p.numel,
            rotation=rotation, orig_shape=orig_shape,
        )

    def _cos_sin_for(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._cos_sin_cast
        if cached is not None and cached[0] == dtype and cached[1] == device:
            return cached[2], cached[3]
        cos = self._prefill_cos.to(dtype=dtype, device=device)
        sin = self._prefill_sin.to(dtype=dtype, device=device)
        self._cos_sin_cast = (dtype, device, cos, sin)
        return cos, sin

    def _decode_ready_layer(
        self, layer_idx: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the decode-ready (post-RoPE K, V) prefill reconstruction for
        a layer, cached in the attention dtype across decode steps."""
        cached = self._recon_cache_layer.get(layer_idx)
        if (
            cached is not None
            and cached[0].dtype == dtype
            and cached[0].device == device
        ):
            k_ready, v_ready = cached
            return k_ready.unsqueeze(0), v_ready.unsqueeze(0)

        k_pre_recon, v_recon = self._reconstruct_layer(layer_idx)
        # Apply RoPE and cast once to the attention dtype.
        cos, sin = self._cos_sin_for(dtype, device)
        k_ready_b = apply_rope_to_k(
            k_pre_recon.to(dtype=dtype, device=device), cos, sin
        )
        v_ready_b = v_recon.to(dtype=dtype, device=device)

        # Avoid persistent dense layer storage above the long-context limit.
        if k_ready_b.shape[-2] <= LAYER_RECON_CACHE_MAX_T:
            v_store = v_ready_b.squeeze(0)
            if v_ready_b is v_recon:
                # Clone views so releasing group cache frees the backing storage.
                v_store = v_store.clone()
                v_ready_b = v_store.unsqueeze(0)
            self._recon_cache_layer[layer_idx] = (
                k_ready_b.squeeze(0),
                v_store,
            )
            self._release_group_cache_if_done(layer_idx)
        return k_ready_b, v_ready_b

    def _release_group_cache_if_done(self, layer_idx: int) -> None:
        """Free a group's float32 recon cache once every one of its layers has a decode-ready entry; the group cache is only an intermediate."""
        g_idx, _slot = self._layer_to_group_slot[layer_idx]
        pending = self._group_layers_pending.get(g_idx)
        if pending is not None:
            pending.discard(layer_idx)
            if pending:
                return
            self._group_layers_pending.pop(g_idx, None)
        self._recon_cache_k.pop(g_idx, None)
        self._recon_cache_v.pop(g_idx, None)

    def _reconstruct_layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-RoPE float32 reconstruction (Tucker + JL residual) for a layer.

        Used as the builder for `_decode_ready_layer` and directly by
        analysis/tests that want the pre-RoPE reconstruction. Not cached at
        layer level itself; the decode path caches the post-RoPE result.
        """
        g_idx, slot = self._layer_to_group_slot[layer_idx]

        # Reconstruct each group once and cache it until layer entries are ready.
        if g_idx not in self._recon_cache_k:
            self._recon_cache_k[g_idx] = self._approx_k[g_idx].reconstruct()
        if g_idx not in self._recon_cache_v:
            self._recon_cache_v[g_idx] = self._approx_v[g_idx].reconstruct()

        k_layer = self._recon_cache_k[g_idx][slot]  # (n_kv, T, d_h)
        v_layer = self._recon_cache_v[g_idx][slot]

        encoded_k = self._residual_k.get(layer_idx)
        if encoded_k is not None:
            enc = self._get_encoder(self._bits_k[g_idx])
            k_layer = k_layer + enc.decode(encoded_k)
        encoded_v = self._residual_v.get(layer_idx)
        if encoded_v is not None:
            enc = self._get_encoder(self._bits_v[g_idx])
            v_layer = v_layer + enc.decode(encoded_v)

        return k_layer.unsqueeze(0), v_layer.unsqueeze(0)


    def inject_prefill_layer(
        self,
        layer_idx: int,
        k_pre_rope: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Stash pre-RoPE K and V for a layer without going through forward
        hooks. For unit tests that drive the cache manually."""
        if k_pre_rope.dim() != 4 or v.dim() != 4:
            raise ValueError("expected 4D tensors (B, n_kv, T, d_h)")
        self._prefill_k_pre[layer_idx] = k_pre_rope.detach()
        self._prefill_v[layer_idx] = v.detach()
        self._prefill_seen.add(layer_idx)
        self._prefill_seq_len = k_pre_rope.shape[-2]
        self._phase = "prefill_stashed"
        if self._num_layers is None or layer_idx + 1 > self._num_layers:
            self._num_layers = layer_idx + 1

    def inject_prefill_cos_sin(self, cos: torch.Tensor, sin: torch.Tensor) -> None:
        """Set the prefill cos/sin used to re-RoPE reconstructed K."""
        self._prefill_cos = cos.detach()
        self._prefill_sin = sin.detach()
        self._cos_sin_cast = None

    def force_compress(self) -> None:
        """Trigger compression of currently stashed prefill (test helper)."""
        self._compress_prefill()
