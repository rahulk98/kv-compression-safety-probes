from importlib import import_module
import math

import torch
from torch.nn import functional as F
from transformers.cache_utils import Cache, DynamicLayer
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention


class SnapKVCache(Cache):
    """One unpadded prompt per cache; evict once, then retain the dense decode tail.

    Pass the model at construction and this cache as ``past_key_values``. Call
    ``close()`` to detach hooks. Byte statistics cover prompt K/V only, excluding
    the growing decode tail and temporary observation queries before eviction.
    """

    def __init__(self, model, target_ratio=4.0, window_size=32, kernel_size=7):
        if not math.isfinite(target_ratio) or target_ratio < 1:
            raise ValueError("target_ratio must be finite and >= 1")
        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError("window_size must be a positive integer")
        if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        decoder = model.model if hasattr(model, "model") else model
        attentions = [layer.self_attn for layer in decoder.layers]
        if any(not isinstance(attn, (Qwen3Attention, LlamaAttention)) for attn in attentions):
            raise NotImplementedError("Only Qwen3 and Llama attention are supported")
        if any(getattr(attn, "sliding_window", None) is not None for attn in attentions):
            raise NotImplementedError("Sliding-window attention is not supported")
        super().__init__(layers=[DynamicLayer() for _ in attentions])
        self.target_ratio = target_ratio
        self.window_size = window_size
        self.kernel_size = kernel_size
        self._logical_lengths = [0] * len(attentions)
        self._queries = {}
        self._evicted = [False] * len(attentions)
        self.last_original_bytes = 0
        self.last_compressed_bytes = 0
        self.last_compression_ratio = 1.0
        self._hooks = [decoder.register_forward_pre_hook(self._check_inputs, with_kwargs=True)]
        self._hooks.extend(
            attn.register_forward_pre_hook(self._capture_queries, with_kwargs=True)
            for attn in attentions
        )

    def _keep_count(self, length):
        return min(length, max(self.window_size, round(length / self.target_ratio)))

    def _check_inputs(self, module, args, kwargs):
        if kwargs.get("past_key_values") is not self:
            return
        inputs = kwargs.get("input_ids")
        if inputs is None:
            inputs = kwargs.get("inputs_embeds")
        if inputs is None:
            inputs = args[0]
        if inputs.shape[0] != 1:
            raise NotImplementedError("SnapKVCache supports batch size 1 only (no beams)")
        mask = kwargs.get("attention_mask")
        if mask is not None and (
            not isinstance(mask, torch.Tensor) or mask.ndim != 2 or not torch.all(mask == 1)
        ):
            raise NotImplementedError("Only unpadded inputs with no mask or an all-ones 2D mask are supported")

    @torch.no_grad()
    def _capture_queries(self, module, args, kwargs):
        if kwargs.get("past_key_values") is not self or self.get_seq_length(module.layer_idx):
            return
        hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        if hidden.shape[0] != 1:
            raise NotImplementedError("SnapKVCache supports batch size 1 only")
        if self._keep_count(hidden.shape[1]) == hidden.shape[1]:
            self._queries[module.layer_idx] = None
            return
        hidden = hidden[:, -self.window_size:]
        queries = module.q_proj(hidden).view(1, hidden.shape[1], -1, module.head_dim)
        if hasattr(module, "q_norm"):
            queries = module.q_norm(queries)
        queries = queries.transpose(1, 2)
        cos, sin = kwargs["position_embeddings"]
        apply_rope = import_module(type(module).__module__).apply_rotary_pos_emb
        queries, _ = apply_rope(
            queries, queries, cos[:, -self.window_size:], sin[:, -self.window_size:]
        )
        self._queries[module.layer_idx] = queries.detach()

    def _evict(self, layer_idx):
        layer = self.layers[layer_idx]
        queries = self._queries.pop(layer_idx)
        length = layer.keys.shape[-2]
        keep = self._keep_count(length)
        if keep < length:
            prefix_length = length - self.window_size
            prefix_keep = keep - self.window_size
            batch, kv_heads, _, head_dim = layer.keys.shape
            if prefix_keep:
                queries = queries.reshape(batch, kv_heads, -1, self.window_size, head_dim)
                logits = torch.matmul(
                    queries.float(), layer.keys[:, :, None].float().transpose(-1, -2)
                ) * head_dim**-0.5
                window_mask = torch.ones(
                    (self.window_size, self.window_size), dtype=torch.bool, device=logits.device
                ).triu(1)
                logits[..., prefix_length:].masked_fill_(window_mask, float("-inf"))
                scores = logits.softmax(-1, dtype=torch.float32)[..., :prefix_length].sum(-2).mean(2)
                scores = F.max_pool1d(scores, self.kernel_size, stride=1, padding=self.kernel_size // 2)
                prefix = scores.topk(prefix_keep, dim=-1).indices.sort(dim=-1).values
            else:
                prefix = torch.empty((batch, kv_heads, 0), dtype=torch.long, device=layer.keys.device)
            window = torch.arange(prefix_length, length, device=layer.keys.device).expand(batch, kv_heads, -1)
            indices = torch.cat((prefix, window), dim=-1).unsqueeze(-1).expand(-1, -1, -1, head_dim)
            bytes_per_token = batch * kv_heads * head_dim * (layer.keys.element_size() + layer.values.element_size())
            layer.keys = layer.keys.gather(2, indices)
            layer.values = layer.values.gather(2, indices)
            self.last_compressed_bytes -= (length - keep) * bytes_per_token
        self._evicted[layer_idx] = True
        self.last_compression_ratio = self.last_original_bytes / self.last_compressed_bytes

    @torch.no_grad()
    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if key_states.shape[0] != 1 or value_states.shape[0] != 1:
            raise NotImplementedError("SnapKVCache supports batch size 1 only")
        if self._logical_lengths[layer_idx] == 0:
            if layer_idx not in self._queries:
                raise RuntimeError("Prefill queries were not captured; use the model supplied to SnapKVCache before close()")
            original = key_states.numel() * key_states.element_size() + value_states.numel() * value_states.element_size()
            self.last_original_bytes += original
            self.last_compressed_bytes += original
        elif not self._evicted[layer_idx]:
            self._evict(layer_idx)
        result = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        self._logical_lengths[layer_idx] += key_states.shape[-2]
        return result

    def get_seq_length(self, layer_idx=0):
        """Logical positions must not change when prompt keys are removed."""
        return self._logical_lengths[layer_idx]

    def get_mask_sizes(self, query_length, layer_idx):
        logical = self.get_seq_length(layer_idx)
        physical = self.layers[layer_idx].get_seq_length()
        if logical and not self._evicted[layer_idx]:
            physical = self._keep_count(physical)
        # Mask construction precedes update(). Align the final stored key with
        # the logical end, so new queries see all past keys but no future tokens.
        return physical + query_length, logical - physical

    def close(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def reset(self):
        raise NotImplementedError("Create a fresh SnapKVCache for each prompt")

    def crop(self, max_length):
        raise NotImplementedError("Speculative decoding and cache cropping are not supported")

    def reorder_cache(self, beam_idx):
        raise NotImplementedError("Beam search is not supported")

    def batch_repeat_interleave(self, repeats):
        raise NotImplementedError("SnapKVCache supports batch size 1 only")

    def batch_select_indices(self, indices):
        raise NotImplementedError("Batch selection is not supported")


def snapkv_scores(queries, keys, window_size, kernel_size, prefix_only=True):
    """SnapKV importance per KV head, (batch, kv_heads, columns), computed as SnapKVCache._evict does.

    ``queries`` are the post-RoPE queries of the last ``window_size`` prompt positions, ``keys`` the post-RoPE
    prompt keys. ``prefix_only`` scores the columns before the window (what the cache selects from); otherwise
    every column is scored (window columns see only the window queries that causally attend to them).
    """
    batch, kv_heads, length, head_dim = keys.shape
    queries = queries.reshape(batch, kv_heads, -1, window_size, head_dim)
    logits = torch.matmul(queries.float(), keys[:, :, None].float().transpose(-1, -2)) * head_dim**-0.5
    window_mask = torch.ones((window_size, window_size), dtype=torch.bool, device=logits.device).triu(1)
    logits[..., length - window_size:].masked_fill_(window_mask, float("-inf"))
    scores = logits.softmax(-1, dtype=torch.float32)
    if prefix_only:
        scores = scores[..., : length - window_size]
    scores = scores.sum(-2).mean(2)
    return F.max_pool1d(scores, kernel_size, stride=1, padding=kernel_size // 2)
