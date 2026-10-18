"""Pre-RoPE K and (cos, sin) capture for live model generation."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel


def _get_attention_layers(model) -> list:
    return [layer.self_attn for layer in model.model.layers]


def _key_norm_module(attn) -> torch.nn.Module | None:
    """Return an attention layer's key-norm submodule, if it has one.

    OLMoE-style QK-norm models expose this as `k_norm`, an RMSNorm applied to
    the FULL k_proj output before reshape/RoPE (see this module's docstring).
    Returns None for models without one (e.g. Mistral, LLaMA-2), which keeps
    the plain `k_proj` capture path untouched.
    """
    mod = getattr(attn, "k_norm", None)
    return mod if isinstance(mod, torch.nn.Module) else None


class PreRoPECapture:
    """Captures per-forward pre-RoPE K and cos/sin tensors.

    The buffers are overwritten on every forward (including 1-token decode
    steps), so consumers must `pop_*` what they need before the next forward.
    """

    def __init__(self, model: "PreTrainedModel"):
        self.model = model
        self.attn_layers = _get_attention_layers(model)
        self.num_kv_heads = model.config.num_key_value_heads
        head_dim = getattr(model.config, "head_dim", None)
        if head_dim is None:
            head_dim = model.config.hidden_size // model.config.num_attention_heads
        self.head_dim = head_dim
        # Optional OLMoE post-k_norm clamp.
        self._clip_qkv = getattr(model.config, "clip_qkv", None)

        # Layer 0 represents the shared attention architecture.
        self.has_key_norm = bool(self.attn_layers) and _key_norm_module(self.attn_layers[0]) is not None

        self._k_pre_rope: dict[int, torch.Tensor] = {}
        self._cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None
        self._hooks: list = []
        self._rope_patch: tuple | None = None

    def _make_proj_hook(self, layer_idx: int):
        expected_width = self.num_kv_heads * self.head_dim

        def hook_fn(_module, _inputs, output: torch.Tensor):
            if output.ndim == 3:
                bsz, seq_len, width = output.shape
            elif output.ndim == 2:
                # Newer transformers attention paths flatten (batch, seq) to
                # (batch * seq, hidden) before Linear. In prefill we always
                # extract with batch_size = 1, so seq_len = output.shape[0].
                bsz, width = 1, output.shape[1]
                seq_len = output.shape[0]
            else:
                raise RuntimeError(
                    f"k_proj hook: unexpected output rank {output.ndim} at "
                    f"layer {layer_idx}: {tuple(output.shape)}"
                )
            if width != expected_width:
                raise RuntimeError(
                    f"k_proj hook: output width mismatch at layer {layer_idx}: "
                    f"{tuple(output.shape)} (expected trailing dim "
                    f"{expected_width} = num_kv_heads * head_dim)"
                )
            t = output.reshape(bsz, seq_len, self.num_kv_heads, self.head_dim)
            self._k_pre_rope[layer_idx] = t.transpose(1, 2).detach()
        return hook_fn

    def _make_key_norm_hook(self, layer_idx: int):
        """Hook for models with a k_norm submodule (see module docstring)."""
        expected_width = self.num_kv_heads * self.head_dim
        clip_qkv = self._clip_qkv

        def hook_fn(_module, _inputs, output: torch.Tensor):
            shape = tuple(output.shape)
            if clip_qkv is not None:
                output = output.clamp(min=-clip_qkv, max=clip_qkv)
            if len(shape) == 4 and shape[2:] == (self.num_kv_heads, self.head_dim):
                # Per-head norm on the (B, T, H, D) view (Qwen3).
                self._k_pre_rope[layer_idx] = output.transpose(1, 2).detach()
                return
            if len(shape) != 3 or shape[-1] != expected_width:
                raise RuntimeError(
                    f"k_norm output width mismatch at layer {layer_idx}: got "
                    f"shape {shape}, expected (B, T, {self.num_kv_heads}*"
                    f"{self.head_dim}={expected_width}) for a norm over the "
                    f"full K width (OLMoE) or (B, T, {self.num_kv_heads}, "
                    f"{self.head_dim}) for a per-head norm (Qwen3)."
                )
            t = output.view(shape[0], shape[1], self.num_kv_heads, self.head_dim)
            self._k_pre_rope[layer_idx] = t.transpose(1, 2).detach()
        return hook_fn

    def _install_rope_patch(self) -> None:
        module_name = type(self.model).__module__
        try:
            mod = importlib.import_module(module_name)
            original = getattr(mod, "apply_rotary_pos_emb")
        except (ImportError, AttributeError):
            self._rope_patch = None
            return

        def wrapped(q, k, cos, sin, *args, **kwargs):
            # cos/sin are shared across layers.
            if self._cos_sin is None:
                self._cos_sin = (cos.detach(), sin.detach())
            return original(q, k, cos, sin, *args, **kwargs)

        mod.apply_rotary_pos_emb = wrapped
        self._rope_patch = (mod, original)

    def attach(self) -> None:
        if self._hooks or self._rope_patch is not None:
            return  # idempotent
        for i, attn in enumerate(self.attn_layers):
            key_norm = _key_norm_module(attn)
            if key_norm is not None:
                self._hooks.append(
                    key_norm.register_forward_hook(self._make_key_norm_hook(i))
                )
            else:
                self._hooks.append(
                    attn.k_proj.register_forward_hook(self._make_proj_hook(i))
                )
        self._install_rope_patch()

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        if self._rope_patch is not None:
            mod, original = self._rope_patch
            mod.apply_rotary_pos_emb = original
            self._rope_patch = None
        self._k_pre_rope.clear()
        self._cos_sin = None

    def begin_forward(self) -> None:
        """Reset cos/sin so the next forward captures fresh values."""
        self._cos_sin = None

    def pop_pre_rope_k(self, layer_idx: int) -> torch.Tensor:
        return self._k_pre_rope[layer_idx]

    def get_cos_sin(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cos_sin is None:
            raise RuntimeError(
                "cos/sin not captured. apply_rotary_pos_emb patch may have failed; "
                "check that the model's modeling module exposes apply_rotary_pos_emb."
            )
        return self._cos_sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_k(
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Apply LLaMA/Mistral RoPE to K with `cos` and `sin` broadcast over KV heads."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (k * cos) + (rotate_half(k) * sin)
