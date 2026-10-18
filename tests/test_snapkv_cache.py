from contextlib import closing

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

from kvsafety import SnapKVCache


@pytest.fixture(autouse=True)
def cpu_inference():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.inference_mode():
        yield
    torch.set_num_threads(old_threads)


def tiny_model(family, backend="eager"):
    torch.manual_seed(17)
    config_cls, model_cls = {
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
        "llama": (LlamaConfig, LlamaForCausalLM),
    }[family]
    config = config_cls(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
    )
    config._attn_implementation = backend
    return model_cls(config).eval()


def prompt(length):
    return (torch.arange(length).unsqueeze(0) % 127) + 1


@pytest.mark.parametrize("family", ["qwen3", "llama"])
@pytest.mark.parametrize("backend", ["sdpa", "eager"])
@pytest.mark.parametrize("length,ratio", [(45, 1), (12, 4)])
def test_no_eviction_generate_matches_dense(family, backend, length, ratio):
    model = tiny_model(family, backend)
    inputs = prompt(length)
    kwargs = dict(
        input_ids=inputs,
        attention_mask=torch.ones_like(inputs),
        max_new_tokens=4,
        do_sample=False,
        return_dict_in_generate=True,
        output_logits=True,
    )
    dense = model.generate(past_key_values=DynamicCache(config=model.config), **kwargs)
    with closing(SnapKVCache(model, target_ratio=ratio)) as cache:
        actual = model.generate(past_key_values=cache, **kwargs)
        assert torch.equal(actual.sequences, dense.sequences)
        for actual_logits, dense_logits in zip(actual.logits, dense.logits, strict=True):
            torch.testing.assert_close(actual_logits, dense_logits, rtol=1e-5, atol=1e-6)
        assert cache.get_seq_length() == length + 3
        assert cache.last_compression_ratio == 1.0


@pytest.mark.parametrize("family", ["qwen3", "llama"])
@pytest.mark.parametrize("backend", ["sdpa", "eager"])
def test_eviction_shape_bytes_positions_and_dense_tail(family, backend):
    model = tiny_model(family, backend)
    positions = []
    def capture_positions(module, args, kwargs):
        positions.append((kwargs["position_ids"] if "position_ids" in kwargs else args[1]).clone())

    handle = model.model.rotary_emb.register_forward_pre_hook(capture_positions, with_kwargs=True)
    with handle, closing(SnapKVCache(model, target_ratio=4)) as cache:
        model(prompt(200), past_key_values=cache)
        full = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
        assert all(layer.keys.shape == (1, 2, 200, 16) for layer in cache.layers)
        assert cache.get_seq_length() == 200
        # Mask creation anticipates 50 retained prompt keys plus the new input.
        assert cache.get_mask_sizes(1, 0) == (51, 150)
        tails = []
        for step in range(3):
            model(torch.tensor([[3 + step]]), past_key_values=cache)
            assert positions[-1].item() == 200 + step
            assert cache.get_seq_length() == 201 + step
            for idx, layer in enumerate(cache.layers):
                assert layer.keys.shape == layer.values.shape == (1, 2, 51 + step, 16)
                # The observation window is retained unchanged, in original order.
                torch.testing.assert_close(layer.keys[:, :, 18:50], full[idx][0][:, :, -32:], rtol=0, atol=0)
                torch.testing.assert_close(layer.values[:, :, 18:50], full[idx][1][:, :, -32:], rtol=0, atol=0)
                if step:
                    torch.testing.assert_close(layer.keys[:, :, 50:-1], tails[idx][0], rtol=0, atol=0)
                    torch.testing.assert_close(layer.values[:, :, 50:-1], tails[idx][1], rtol=0, atol=0)
            tails = [(layer.keys[:, :, 50:].clone(), layer.values[:, :, 50:].clone()) for layer in cache.layers]
        assert cache.last_original_bytes == 2 * 2 * 2 * 200 * 16 * 4
        assert cache.last_compressed_bytes == 2 * 2 * 2 * 50 * 16 * 4
        assert cache.last_compression_ratio == pytest.approx(4.0)


@pytest.mark.parametrize("family", ["qwen3", "llama"])
def test_selection_matches_independent_eager_attention_reference(family):
    model = tiny_model(family)
    with closing(SnapKVCache(model, target_ratio=4)) as cache:
        output = model(prompt(200), past_key_values=cache, output_attentions=True)
        full_keys = cache.layers[0].keys.clone()
        full_values = cache.layers[0].values.clone()
        attention = output.attentions[0][0, :, -32:, :168]
        scores = attention.sum(-2).reshape(2, 2, 168).mean(1)
        pooled = torch.stack([
            scores[:, max(0, pos - 3):min(168, pos + 4)].max(-1).values
            for pos in range(168)
        ], dim=-1)
        selected = pooled.topk(18, dim=-1).indices.sort(-1).values
        indices = torch.cat((selected, torch.arange(168, 200).expand(2, -1)), dim=-1)
        model(torch.tensor([[5]]), past_key_values=cache)
        # Recover indices from exact stored keys; the cache need not retain them.
        recovered = torch.stack([
            torch.cdist(cache.layers[0].keys[0, head, :50], full_keys[0, head]).argmin(-1)
            for head in range(2)
        ])
        assert torch.equal(recovered, indices)
        gather = indices[None, :, :, None].expand(1, 2, 50, 16)
        torch.testing.assert_close(cache.layers[0].keys[:, :, :50], full_keys.gather(2, gather), rtol=0, atol=0)
        torch.testing.assert_close(cache.layers[0].values[:, :, :50], full_values.gather(2, gather), rtol=0, atol=0)


@pytest.mark.parametrize("family", ["qwen3", "llama"])
@pytest.mark.parametrize("backend", ["sdpa", "eager"])
def test_dense_oracle_and_real_eviction_changes_logits(family, backend):
    model = tiny_model(family, backend)
    dense_cache = DynamicCache(config=model.config)
    dense_prefill = model(prompt(200), past_key_values=dense_cache).logits
    dense_decode = model(torch.tensor([[5]]), past_key_values=dense_cache).logits
    for ratio in (1, 4):
        with closing(SnapKVCache(model, target_ratio=ratio)) as cache:
            prefill = model(prompt(200), past_key_values=cache).logits
            torch.testing.assert_close(prefill, dense_prefill, rtol=0, atol=0)
            decoded = model(torch.tensor([[5]]), past_key_values=cache).logits
            if ratio == 1:
                torch.testing.assert_close(decoded, dense_decode, rtol=0, atol=0)
            else:
                assert not torch.allclose(decoded, dense_decode, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_decode_mask_and_chunked_continuation_are_causal(backend):
    model = tiny_model("qwen3", backend)
    continuation = torch.tensor([[5, 6, 7]])
    with closing(SnapKVCache(model, target_ratio=4)) as chunked:
        model(prompt(200), past_key_values=chunked)
        mask = create_causal_mask(
            model.config, model.model.embed_tokens(continuation),
            attention_mask=torch.ones(1, 203), past_key_values=chunked,
        )
        visible = mask if mask.dtype == torch.bool else mask == 0
        assert visible.shape == (1, 1, 3, 53)
        assert visible[0, 0, :, :50].all()
        assert torch.equal(visible[0, 0, :, 50:], torch.ones(3, 3, dtype=torch.bool).tril())
        actual = model(continuation, past_key_values=chunked).logits
    with closing(SnapKVCache(model, target_ratio=4)) as serial:
        model(prompt(200), past_key_values=serial)
        expected = torch.cat([
            model(token, past_key_values=serial).logits
            for token in continuation.split(1, dim=1)
        ], dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("family", ["qwen3", "llama"])
def test_batch_two_is_rejected(family):
    model = tiny_model(family)
    with closing(SnapKVCache(model)) as cache, pytest.raises(NotImplementedError, match="batch"):
        model(prompt(40).expand(2, -1), past_key_values=cache)


def test_close_removes_hooks():
    model = tiny_model("qwen3")
    modules = [model.model, *(layer.self_attn for layer in model.model.layers)]
    before = [len(module._forward_pre_hooks) for module in modules]
    cache = SnapKVCache(model)
    assert [len(module._forward_pre_hooks) for module in modules] == [n + 1 for n in before]
    cache.close()
    cache.close()
    assert [len(module._forward_pre_hooks) for module in modules] == before


@pytest.mark.parametrize("kwargs", [
    {"target_ratio": 0.9}, {"target_ratio": float("nan")},
    {"window_size": 0}, {"kernel_size": 2},
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        SnapKVCache(tiny_model("qwen3"), **kwargs)


def test_padding_is_rejected():
    model = tiny_model("llama")
    mask = torch.ones(1, 40, dtype=torch.long)
    mask[:, 0] = 0
    with closing(SnapKVCache(model)) as cache, pytest.raises(NotImplementedError, match="mask"):
        model(prompt(40), attention_mask=mask, past_key_values=cache)


@pytest.mark.parametrize("family", ["qwen3", "llama"])
@pytest.mark.parametrize("backend", ["sdpa", "eager"])
def test_generate_with_eviction_keeps_logical_positions(family, backend):
    model = tiny_model(family, backend)
    with closing(SnapKVCache(model, target_ratio=4)) as cache:
        output = model.generate(
            prompt(200), attention_mask=torch.ones(1, 200, dtype=torch.long),
            past_key_values=cache, max_new_tokens=4, do_sample=False,
        )
        assert output.shape == (1, 204)
        assert cache.get_seq_length() == 203
        assert all(layer.keys.shape[-2] == 53 for layer in cache.layers)
        assert cache._queries == {}


@pytest.mark.parametrize("length,ratio,kept", [(33, 100, 32), (201, 4, 50), (206, 4, 52), (1, 4, 1)])
def test_budget_rounding_and_window_only(length, ratio, kept):
    model = tiny_model("qwen3")
    with closing(SnapKVCache(model, target_ratio=ratio)) as cache:
        model(prompt(length), past_key_values=cache)
        window = cache.layers[0].keys[:, :, -32:].clone()
        model(torch.tensor([[5]]), past_key_values=cache)
        assert cache.layers[0].keys.shape[-2] == kept + 1
        if kept <= 32:
            torch.testing.assert_close(cache.layers[0].keys[:, :, :kept], window, rtol=0, atol=0)
        assert cache.last_compression_ratio == pytest.approx(length / kept)


def test_byte_accounting_uses_actual_dtype():
    model = tiny_model("qwen3").to(torch.bfloat16)
    with closing(SnapKVCache(model, target_ratio=4)) as cache:
        model(prompt(200), past_key_values=cache)
        model(torch.tensor([[5]]), past_key_values=cache)
        assert all(layer.keys.dtype == layer.values.dtype == torch.bfloat16 for layer in cache.layers)
        assert cache.last_original_bytes == 2 * 2 * 2 * 200 * 16 * 2
        assert cache.last_compressed_bytes == 2 * 2 * 2 * 50 * 16 * 2
