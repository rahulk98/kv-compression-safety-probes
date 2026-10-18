"""Instance-scoped HuggingFace attention adapter for FlashJoLT fused decode."""

from __future__ import annotations

import importlib
import inspect
import logging
from dataclasses import dataclass, field
from typing import Callable

import torch

from ..flashjolt import kernel as _kernel
from ..flashjolt.triton_unpack import TRITON_AVAILABLE
from .compressed_cache import CompressedKVCache

logger = logging.getLogger(__name__)

# Validate the forward signature once per installation.
_VALIDATED_PARAM_NAMES = ("hidden_states", "position_embeddings", "attention_mask", "past_key_values")


def _signature_check(forward: Callable) -> tuple[bool, str]:
    try:
        sig = inspect.signature(forward)
    except (TypeError, ValueError) as exc:
        return False, f"could not inspect attention forward signature: {exc}"
    params = list(sig.parameters.values())
    names = tuple(p.name for p in params[: len(_VALIDATED_PARAM_NAMES)])
    if names != _VALIDATED_PARAM_NAMES:
        return False, (
            f"attention forward signature starts with {names!r}, expected the "
            f"validated transformers 5.5.4 order {_VALIDATED_PARAM_NAMES!r}"
        )
    if not params or params[-1].kind != inspect.Parameter.VAR_KEYWORD:
        return False, "attention forward signature has no trailing **kwargs parameter"
    return True, ""


def _mask_is_trivial_causal(mask: torch.Tensor | None) -> bool:
    """True iff `mask` imposes no restriction beyond the pure causal mask the fused kernel's own `CAUSAL=True` + `query_position` already implements (see kernel.py's `_fused_decode_kernel`)."""
    if mask is None:
        return True
    return bool((mask == 0).all())


@dataclass
class _RuntimeState:
    """Shared, per-installation state referenced by every wrapped layer's
    closure. `latched_fused` is the ONE fused-vs-dense decision for the
    current prompt (None = not yet decided this prompt); `factor_cache` is
    populated once per prompt (during preflight) and cleared at the next
    transition, so a fused decode step never rebuilds a layer's Tucker
    factor handle from scratch.
    """

    cache: CompressedKVCache
    apply_rotary_pos_emb: Callable
    signature_ok: bool
    signature_reason: str
    num_layers: int
    original_forwards: dict = field(default_factory=dict)
    latched_fused: bool | None = None
    latch_reason: str = ""
    factor_cache: dict = field(default_factory=dict)


class FlashJoltInstallation:
    """Installation handle that restores each attention module's original `forward`."""

    def __init__(self, attn_modules: list, state: _RuntimeState, model=None):
        self._attn_modules = list(attn_modules)
        self.state = state
        # Model back-reference for idempotent install/uninstall.
        self._model = model

    def uninstall(self) -> None:
        for i, attn in enumerate(self._attn_modules):
            saved = self.state.original_forwards.get(i)
            if saved is not None:
                # Restore a pre-existing instance forward.
                attn.forward = saved
            elif "forward" in vars(attn):
                # Remove the wrapper to restore class-method lookup.
                del attn.forward
        self._attn_modules = []
        # Do not clear a newer installation's model pointer.
        if self._model is not None and getattr(self._model, "_flashjolt_installation", None) is self:
            del self._model._flashjolt_installation
        self._model = None

    def clear_prompt_state(self) -> None:
        """Drop the PER-PROMPT latch decision and factor cache, WITHOUT uninstalling."""
        self.state.latched_fused = None
        self.state.latch_reason = ""
        self.state.factor_cache.clear()


def _run_preflight(
    state: _RuntimeState, hidden_states: torch.Tensor, attention_mask, kwargs: dict
) -> tuple[bool, str]:
    """ONE latch decision for the whole prompt, checked BEFORE the first
    decode token is appended anywhere (see module docstring). Every layer's
    factor handle is built and cached here (not lazily per-step) so a
    fused-latched prompt never has to touch `flashjolt_factor_handle` again.
    """
    if not state.signature_ok:
        return False, state.signature_reason
    if kwargs.get("output_attentions"):
        return False, "output_attentions=True requested; the fused kernel never materializes attention weights"
    if not TRITON_AVAILABLE:
        return False, "triton is not importable in this environment"
    if not torch.cuda.is_available() or not hidden_states.is_cuda:
        return False, "CUDA is not available (or hidden_states is not on a CUDA device)"
    major, minor = torch.cuda.get_device_capability(hidden_states.device)
    if (major, minor) < (8, 0):
        return False, (
            f"fused decode requires compute capability >= 8.0 (Ampere); device is "
            f"sm_{major}{minor}; falling back to dense"
        )
    if not _mask_is_trivial_causal(attention_mask):
        return False, "attention_mask carries non-trivial masking (padding or a custom mask) the fused kernel does not support"

    cache = state.cache
    for layer_idx in range(state.num_layers):
        handle = cache.flashjolt_factor_handle(layer_idx)
        if handle.tucker_k.head_dim % 2 != 0:
            return False, f"layer {layer_idx}: head_dim={handle.tucker_k.head_dim} is odd; RoPE kernel requires even head_dim"
        if not handle.tucker_k.core.is_cuda or not handle.tucker_v.core.is_cuda:
            return False, f"layer {layer_idx}: Tucker factors are not on a CUDA device"
        state.factor_cache[layer_idx] = handle
    return True, ""


def _fused_forward(
    attn_module,
    layer_idx: int,
    state: _RuntimeState,
    hidden_states: torch.Tensor,
    position_embeddings,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    cache = state.cache
    bsz, q_len, _hidden = hidden_states.shape
    if bsz != 1 or q_len != 1:
        raise RuntimeError(
            f"FlashJoLT fused decode requires batch=1, query_len=1 (got bsz={bsz}, "
            f"q_len={q_len}); the latch already committed this prompt to the fused "
            "path, so this is a hard error, not a fallback -- see module docstring."
        )

    position_ids = kwargs.get("position_ids")
    if position_ids is None:
        raise RuntimeError("FlashJoLT fused decode requires a position_ids kwarg; none was supplied")
    query_position = int(position_ids[0, 0])
    expected_position = cache.get_seq_length(layer_idx)
    if query_position != expected_position:
        raise RuntimeError(
            f"FlashJoLT fused decode: layer {layer_idx} query_position={query_position} "
            f"!= cache.get_seq_length()={expected_position}. Cache cropping, reordering, "
            "or a left-padded layout is not supported (the reference assumes compressed "
            "positions 0..T-1 with the tail immediately after) -- refusing to silently "
            "compute a wrong result."
        )

    head_dim = attn_module.head_dim
    hidden_shape = (bsz, q_len, -1, head_dim)
    q = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    k = attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    v = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    q, k = state.apply_rotary_pos_emb(q, k, cos, sin)

    # Append after validation and before attention.
    cache.append_decode_tail(layer_idx, k, v)
    tail_k, tail_v = cache.decode_tail(layer_idx)

    handle = state.factor_cache[layer_idx]
    prefill_cos, prefill_sin = cache._cos_sin_for(hidden_states.dtype, hidden_states.device)

    query_kernel = q[0, :, 0, :].contiguous()  # (n_q, head_dim), consecutive-per-kv_head (HF repeat_kv convention)
    tail_k_kernel = tail_k[0]  # (n_kv, T_tail, head_dim)
    tail_v_kernel = tail_v[0]

    out = _kernel.fused_decode_attention(
        handle.tucker_k, handle.residual_k, handle.tucker_v, handle.residual_v, handle.slot,
        prefill_cos[0], prefill_sin[0], query_kernel, query_position, attn_module.num_key_value_groups,
        tail_k=tail_k_kernel, tail_v=tail_v_kernel,
        sm_scale=attn_module.scaling, causal=True, out_dtype=hidden_states.dtype,
    )
    attn_output = out.reshape(bsz, q_len, -1, head_dim).reshape(bsz, q_len, -1).contiguous()
    attn_output = attn_module.o_proj(attn_output)
    return attn_output, None


def _make_wrapped_forward(attn_module, layer_idx: int, state: _RuntimeState, original_forward: Callable) -> Callable:
    def wrapped_forward(hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
        cache = state.cache
        if layer_idx == 0:
            transitioned = cache.prepare_decode_transition()
            if transitioned:
                # Clear the latch first so failed preflight leaves dense fallback.
                state.latched_fused = None
                state.latch_reason = "preflight did not complete"
                state.factor_cache.clear()
                ok, reason = _run_preflight(state, hidden_states, attention_mask, kwargs)
                state.latched_fused = ok
                state.latch_reason = reason
                if ok:
                    logger.info("FlashJoLT fused decode ENABLED for this prompt (%d layers)", state.num_layers)
                else:
                    logger.info("FlashJoLT fused decode DISABLED for this prompt (dense fallback): %s", reason)

        if cache._phase != "compressed" or not state.latched_fused:
            return original_forward(
                hidden_states, position_embeddings=position_embeddings,
                attention_mask=attention_mask, past_key_values=past_key_values, **kwargs,
            )

        # Reject calls using a cache different from the installation's cache.
        if past_key_values is not cache:
            raise RuntimeError(
                "FlashJoLT fused decode: past_key_values supplied to this call "
                f"({past_key_values!r}) is not the cache this installation is bound "
                "to; refusing to attend over or mutate a different cache while "
                "latched to the fused path for another prompt. This usually means a "
                "previous CompressedKVCache built with use_fused_decode=True was not "
                "detach()ed, so its adapter still owns this module's forward -- call "
                "detach() on that cache (or build the next fused cache, which "
                "uninstalls the stale one via the install() idempotency guard)."
            )

        # Recheck per-step options unsupported by fused attention.
        if kwargs.get("output_attentions") or not _mask_is_trivial_causal(attention_mask):
            raise RuntimeError(
                "FlashJoLT fused decode: this step introduced output_attentions=True "
                "or a non-trivial attention_mask after the prompt already latched to "
                "the fused path; the fused kernel cannot honor it. Refusing rather "
                "than silently ignoring it."
            )

        return _fused_forward(attn_module, layer_idx, state, hidden_states, position_embeddings, **kwargs)

    return wrapped_forward


def install(model, cache: CompressedKVCache) -> FlashJoltInstallation:
    """Wrap every attention module instance in `model.model.layers` with the
    FlashJoLT fused-decode-attention adapter, bound to `cache`. Call BEFORE
    `model.generate(..., past_key_values=cache)`. See module docstring for
    the full state machine; `.uninstall()` on the returned handle restores
    every module's `forward` to its pre-install state.
    """
    # Remove any prior installation before wrapping modules again.
    prior = getattr(model, "_flashjolt_installation", None)
    if prior is not None:
        prior.uninstall()

    layers = model.model.layers
    attn_modules = [layer.self_attn for layer in layers]
    if not attn_modules:
        raise ValueError("install: model.model.layers is empty")

    signature_ok, signature_reason = True, ""
    for i, attn in enumerate(attn_modules):
        ok, reason = _signature_check(attn.forward)
        if not ok:
            signature_ok, signature_reason = False, f"layer {i}: {reason}"
            break

    apply_rotary_pos_emb = None
    if signature_ok:
        module_name = type(attn_modules[0]).__module__
        try:
            modeling_module = importlib.import_module(module_name)
            apply_rotary_pos_emb = getattr(modeling_module, "apply_rotary_pos_emb")
        except (ImportError, AttributeError) as exc:
            signature_ok = False
            signature_reason = f"could not resolve apply_rotary_pos_emb from {module_name!r}: {exc}"

    if not signature_ok:
        logger.warning("FlashJoLT fused decode permanently disabled for this installation: %s", signature_reason)

    state = _RuntimeState(
        cache=cache,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
        signature_ok=signature_ok,
        signature_reason=signature_reason,
        num_layers=len(attn_modules),
    )

    for i, attn in enumerate(attn_modules):
        # `attn.forward` here is whatever the module resolved BEFORE this
        # install: usually the class-lookup bound method, but possibly a
        # pre-existing INSTANCE-level override (accelerate device_map, PEFT,
        # or a previous install of this adapter). The wrapper always
        # delegates to it; `original_forwards[i]` records the instance-level
        # override itself when one existed (else None) so `uninstall()` can
        # restore the exact pre-install state instead of blindly deleting
        # (M7 fix -- see FlashJoltInstallation's docstring).
        original_forward = attn.forward
        state.original_forwards[i] = vars(attn).get("forward")
        attn.forward = _make_wrapped_forward(attn, i, state, original_forward)

    handle = FlashJoltInstallation(attn_modules, state, model=model)
    # Record the back-reference so a subsequent install on the same model can
    # find and uninstall this one first (idempotency guard above), and so
    # `uninstall()` can clear it. Instance-scoped, never a class attribute.
    model._flashjolt_installation = handle
    # Keep the cache's own handle in sync too, so `cache.detach()` tears down
    # THIS installation whether it was created by the `__init__` auto-install
    # or a manual `install()` call. `CompressedKVCache.__init__` always
    # initialises `_flashjolt_installation`; the getattr guard keeps `install()`
    # usable with a non-CompressedKVCache stand-in in isolated unit tests.
    if hasattr(cache, "_flashjolt_installation"):
        cache._flashjolt_installation = handle
    return handle
