"""Single-pass driver for the controls: every variant of a prompt comes from one dense prefill and at most one JoLT compression.

A variant is the post-RoPE K/V of the first T-1 prompt tokens per layer (fewer tokens for eviction) held in a plain
DynamicCache. The last prompt token is then fed alone (finish), followed by greedy decoding, with explicit logical
position ids so evicted prefixes keep their RoPE positions. This matches run.py's compress-then-finish protocol.
"""

import argparse
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvsafety import SnapKVCache
from kvsafety.pipeline.data import build_input_ids_with_span, load_filler_ids
from kvsafety.pipeline.run import last_hidden, sync
from kvsafety.snapkv_cache import snapkv_scores

SNAP = ("snapkv", "random", "snapkv_doconly", "snapkv_reqonly")
JOLT = ("jolt", "noise", "jolt_doconly", "jolt_reqonly")
COUNTERFACTUAL = ("jolt_doconly", "jolt_reqonly", "snapkv_doconly", "snapkv_reqonly")
BASE = ("dense",) + SNAP + JOLT
WINDOW, KERNEL = 32, 7  # SnapKVCache defaults


def seed(*parts):
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little") >> 1


def rel_err(approx, ref):
    return ((approx.float() - ref.float()).norm() / ref.float().norm()).item()


def noisy(x, rel, seed_):
    """x plus Gaussian noise scaled so that ||noise|| / ||x|| == rel."""
    g = torch.randn(x.shape, generator=torch.Generator().manual_seed(seed_)).to(x.device)
    xf = x.float()
    return (xf + g * (rel * xf.norm() / g.norm())).to(x.dtype)


def gather(x, idx):
    """x (1, kv_heads, L, d), idx (kv_heads, n): per-head positions."""
    return x.gather(2, idx[None, :, :, None].expand(1, -1, -1, x.shape[-1]))


def pick(score, always, cand, k):
    """Per KV head: the positions in `always` plus the k highest-scoring positions of `cand`, sorted.

    score (kv_heads, L); always and cand are disjoint bool masks over the L positions.
    """
    heads = score.shape[0]
    positions = cand.nonzero().squeeze(-1)
    # top-k over the candidate columns only, so ties (frequent after max-pooling) break as in SnapKVCache._evict
    top = positions[score[:, positions].topk(k, dim=-1).indices]
    return torch.cat((always.nonzero().squeeze(-1).expand(heads, -1), top), dim=-1).sort(-1).values


def kv_bytes(prefix):
    return sum(t.numel() * t.element_size() for kv in prefix for t in kv)


class Prompt:
    """Lazily built K/V prefixes of one prompt, memoised so that variants share their dense prefill and JoLT compression."""

    def __init__(self, model, ids, row_id, ratio, span_start, snap):
        self.model, self.ids, self.id, self.ratio, self.span, self.snap = model, ids, row_id, ratio, span_start, snap
        self.L = ids.shape[1] - 1
        self.memo = {}

    def get(self, key, fn):
        if key not in self.memo:
            try:
                self.memo[key] = fn()
            except Exception as e:
                self.memo[key] = e
        if isinstance(self.memo[key], Exception):
            raise self.memo[key]
        return self.memo[key]

    def prefix(self, variant):
        """(prefix as a list of (K, V) per layer, row extras, seconds spent building it)."""
        return self.get(variant, lambda: getattr(self, "_" + variant)())

    def _prefill(self):
        t0 = sync(self.ids.device)
        cache = SnapKVCache(self.model, self.ratio) if self.snap else DynamicCache()
        try:
            self.model(input_ids=self.ids[:, :-1], past_key_values=cache, use_cache=True, logits_to_keep=1)
        finally:
            if self.snap:
                cache.close()
        prefix = [(layer.keys, layer.values) for layer in cache.layers]
        queries = [cache._queries.get(i) for i in range(len(prefix))] if self.snap else None
        return prefix, queries, sync(self.ids.device) - t0

    def _dense(self):
        prefix, _, secs = self.get("prefill", self._prefill)
        return prefix, {}, secs

    def _jolt(self):
        from kvsafety.jolt_cache import make_jolt_cache

        dense, _, _ = self.get("prefill", self._prefill)
        t0 = sync(self.ids.device)
        torch.manual_seed(seed(self.id, "jolt", f"{self.ratio:g}"))  # the compression draws from the global RNG
        cache = make_jolt_cache(self.model, self.ratio)
        try:
            self.model(input_ids=self.ids[:, :-1], past_key_values=cache, use_cache=True, logits_to_keep=1)
            self.model(input_ids=self.ids[:, -1:], past_key_values=cache, use_cache=True)
            dtype, device = dense[0][0].dtype, dense[0][0].device
            prefix = []
            for layer in range(len(dense)):
                k, v = cache._decode_ready_layer(layer, dtype, device)
                prefix.append((k.clone(), v.clone()))
            extras = {"original_bytes": cache.last_original_bytes, "compressed_bytes": cache.last_compressed_bytes}
        finally:
            cache.detach()
        extras["rel_err_k"] = [rel_err(p[0], d[0]) for p, d in zip(prefix, dense)]
        extras["rel_err_v"] = [rel_err(p[1], d[1]) for p, d in zip(prefix, dense)]
        return prefix, extras, sync(self.ids.device) - t0

    def _noise(self):
        dense, _, s_dense = self.get("prefill", self._prefill)
        _, jolt, s_jolt = self.prefix("jolt")
        t0 = sync(self.ids.device)
        r = f"{self.ratio:g}"
        prefix = [
            (
                noisy(k, jolt["rel_err_k"][i], seed(self.id, i, "k", r)),
                noisy(v, jolt["rel_err_v"][i], seed(self.id, i, "v", r)),
            )
            for i, (k, v) in enumerate(dense)
        ]
        extras = {"rel_err_k": jolt["rel_err_k"], "rel_err_v": jolt["rel_err_v"]}
        return prefix, extras, s_dense + s_jolt + sync(self.ids.device) - t0

    def _splice(self, first, second):
        """Positions [0, span) from `first`, [span, L) from `second`."""
        a, _, sa = self.prefix(first)
        b, _, sb = self.prefix(second)
        prefix = [(torch.cat((ak[:, :, : self.span], bk[:, :, self.span :]), 2),
                   torch.cat((av[:, :, : self.span], bv[:, :, self.span :]), 2)) for (ak, av), (bk, bv) in zip(a, b)]
        extras = {"span_start": self.span, "n_document_tokens": self.span, "n_request_tokens": self.L - self.span}
        return prefix, extras, sa + sb

    def _jolt_doconly(self):
        return self._splice("jolt", "dense")

    def _jolt_reqonly(self):
        return self._splice("dense", "jolt")

    def _evict(self, variant):
        """Kept-token prefix. snapkv: SnapKV selection. random: same kept count and window, uniform choice.

        Counterfactuals: doconly applies SnapKV to the document positions only (the window and the whole request
        span are kept, the document keeps round(n_document / ratio) positions in total, window positions
        included); reqonly keeps every document position and keeps round(n_request / ratio) request positions by
        SnapKV score, scored over all columns (the window rule is ignored).
        """
        dense, queries, s_prefill = self.get("prefill", self._prefill)
        t0 = sync(self.ids.device)
        L, ratio = self.L, self.ratio
        pos = torch.arange(L, device=self.ids.device)
        window, request = pos >= L - WINDOW, pos >= self.span
        if variant in ("snapkv", "random"):
            always, cand, k = window, ~window, min(L, max(WINDOW, round(L / ratio))) - WINDOW
        elif variant == "snapkv_doconly":
            doc = ~request
            always, cand = request | window, doc & ~window
            k = min(int(cand.sum()), max(0, round(int(doc.sum()) / ratio) - int((doc & window).sum())))
        else:
            always, cand, k = ~request, request, round(int(request.sum()) / ratio)
        prefix = []
        for i, (key, value) in enumerate(dense):
            heads = key.shape[1]
            if variant == "random":
                score = torch.rand(heads, L, generator=torch.Generator().manual_seed(seed(self.id, i, "random", f"{ratio:g}")))
                score = score.to(key.device)
            elif queries[i] is None:
                score = torch.zeros(heads, L, device=key.device)
            else:
                score = snapkv_scores(queries[i], key, WINDOW, KERNEL, prefix_only=variant != "snapkv_reqonly")[0]
                score = torch.nn.functional.pad(score, (0, L - score.shape[-1]))
            idx = pick(score, always, cand, k)
            prefix.append((gather(key, idx), gather(value, idx)))
        extras = {"kept_tokens": idx.shape[-1], "prefix_tokens": L,
                  "original_bytes": kv_bytes(dense), "compressed_bytes": kv_bytes(prefix)}
        if variant in COUNTERFACTUAL:
            kept_request = int((idx[0] >= self.span).sum())
            extras |= {"span_start": self.span, "n_document_tokens": self.span, "n_request_tokens": L - self.span,
                       "kept_document_tokens": idx.shape[-1] - kept_request, "kept_request_tokens": kept_request}
        return prefix, extras, s_prefill + sync(self.ids.device) - t0

    def _snapkv(self):
        return self._evict("snapkv")

    def _random(self):
        return self._evict("random")

    def _snapkv_doconly(self):
        return self._evict("snapkv_doconly")

    def _snapkv_reqonly(self):
        return self._evict("snapkv_reqonly")


@torch.no_grad()
def finish_and_decode(model, prefix, ids, max_new_tokens, eos_ids, decode, steer=None):
    """Feed the last prompt token on a copy of `prefix`, then decode greedily; positions are logical.

    `steer` is (decoder layer index, vector) added to that layer's output at every position of every forward.
    """
    T, device = ids.shape[1], ids.device
    cache = DynamicCache()
    for i, (k, v) in enumerate(prefix):
        cache.update(k, v, i)  # later updates concatenate into new tensors, the prefix tensors are never written to
    handle = None
    if steer is not None:
        layer, vec = steer
        handle = model.model.layers[layer].register_forward_hook(lambda module, args, output: output + vec)
    try:
        t0 = sync(device)
        out = model(input_ids=ids[:, -1:], past_key_values=cache, position_ids=torch.tensor([[T - 1]], device=device),
                    use_cache=True, output_hidden_states=True)
        t1 = sync(device)
        last = last_hidden(out)
        generated, early = [], []
        for j in range(max_new_tokens if decode else 0):
            token = out.logits[0, -1].argmax().item()
            if token in eos_ids:
                break
            generated.append(token)
            out = model(input_ids=torch.tensor([[token]], device=device), past_key_values=cache,
                        position_ids=torch.tensor([[T + j]], device=device), use_cache=True, output_hidden_states=True)
            if len(early) < 5:
                early.append(last_hidden(out))
        resp5 = np.mean(early, axis=0) if early else last
        t2 = sync(device)
    finally:
        if handle is not None:
            handle.remove()
    return generated, last, resp5, t1 - t0, t2 - t1


@torch.no_grad()
def compute(model, ids, row_id, variants, ratio, span_start, max_new_tokens, eos_ids, decode=True, steer=None):
    """{variant: result dict, or the exception that variant raised}. `steer` is (decoder layer, direction)."""
    prompt = Prompt(model, ids, row_id, ratio, span_start, any(v.partition("_steer")[0] in SNAP for v in variants))
    results = {}
    for variant in variants:
        try:
            base, _, strength = variant.partition("_steer")
            prefix, extras, seconds_prefix = prompt.prefix(base)
            vec = None
            if strength:
                vec = (steer[0], (float(strength) * steer[1]).to(device=ids.device, dtype=prefix[0][0].dtype))
            generated, last, resp5, seconds_finish, seconds_decode = finish_and_decode(
                model, prefix, ids, max_new_tokens, eos_ids, decode, vec
            )
            extras = dict(extras)
            if strength:
                extras |= {"steer_strength": float(strength), "steer_layer": steer[0]}
            results[variant] = {
                "generated": generated, "last": last, "resp5": resp5, "extras": extras,
                "seconds_prefix": seconds_prefix, "seconds_finish": seconds_finish, "seconds_decode": seconds_decode,
            }
        except Exception as e:
            results[variant] = e
    return results


def read_rows(path):
    """Parsed rows of a jsonl file, skipping unparsable lines."""
    rows = []
    for line in Path(path).read_text().splitlines() if Path(path).exists() else []:
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def done_ids(path):
    return {r["id"] for r in read_rows(path) if r.get("status") == "ok"}


def repair(path):
    """A killed writer can leave a partial last line: drop it, or terminate it if it is a complete row."""
    if not path.exists() or path.stat().st_size == 0 or path.read_bytes().endswith(b"\n"):
        return
    head, _, tail = path.read_bytes().rpartition(b"\n")
    try:
        json.loads(tail)
        path.write_bytes(head + b"\n" + tail + b"\n")
    except ValueError:
        path.write_bytes(head + b"\n" if head else b"")
        print(f"{path.name}: dropped a partial trailing line", flush=True)


def variant_list(args):
    variants = [v for v in args.variants.split(",") if v]
    if args.counterfactual:
        variants += [v for v in COUNTERFACTUAL if v not in variants]
    for strength in args.steer.split(",") if args.steer else []:
        variants += [f"{m}_steer{float(strength):g}" for m in args.steer_methods.split(",")]
    for v in variants:
        if v.partition("_steer")[0] not in BASE:
            raise SystemExit(f"unknown variant {v}")
    if args.steer and not args.steer_direction:
        raise SystemExit("--steer needs --steer-direction")
    return [v for v in variants if v != "dense" or args.ratio == 1.0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--set", required=True, choices=["eval", "train", "val"])
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--variants", default="noise,random")
    ap.add_argument("--counterfactual", action="store_true")
    ap.add_argument("--steer", help="comma separated strengths, harmful prompts only")
    ap.add_argument("--steer-direction")
    ap.add_argument("--steer-methods", default="jolt,snapkv")
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--filler-tokens", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--results-dir")
    ap.add_argument("--sets-dir", default="data/sets")
    ap.add_argument("--filler-file", default="data/raw/wikitext_filler.txt")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args(argv)

    variants = variant_list(args)
    shard, n_shards = map(int, args.shard.split("/"))
    model_name = Path(args.model_dir).name
    results_dir = Path(args.results_dir or f"results/{model_name}")
    outs = {v: results_dir / f"{args.set}_{v}_r{args.ratio:g}_f{args.filler_tokens}.jsonl" for v in variants}
    results_dir.mkdir(parents=True, exist_ok=True)
    for out in outs.values():
        out.with_name(out.stem + "_acts").mkdir(exist_ok=True)
        repair(out)
    finished = {v: done_ids(out) for v, out in outs.items()}
    rows = [json.loads(line) for line in (Path(args.sets_dir) / f"{args.set}.jsonl").read_text().splitlines()]
    rows = rows[shard::n_shards]
    applicable = lambda v, row: "_steer" not in v or row["label"] == 1
    todo = [r for r in rows if any(r["id"] not in finished[v] and applicable(v, r) for v in variants)]
    print(f"{len(rows)} rows in shard {args.shard}, {len(todo)} with missing variants: {','.join(variants)}", flush=True)
    if not todo:
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=getattr(torch, args.dtype), attn_implementation="sdpa"
    ).to(device).eval()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    eos_ids.add(tokenizer.eos_token_id)
    filler_ids = load_filler_ids(tokenizer, args.filler_file) if args.filler_tokens > 0 else None
    steer = None
    if args.steer:
        z = np.load(args.steer_direction)
        steer = (int(z["layer"]), torch.from_numpy(z["direction"]).float())

    meta = {
        "set": args.set, "ratio": args.ratio, "filler_tokens": args.filler_tokens, "model_dir": args.model_dir,
        "torch": torch.__version__, "transformers": transformers.__version__, "hostname": socket.gethostname(),
    }
    files = {v: out.open("a") for v, out in outs.items()}
    n_error, start, seconds = 0, time.time(), {}
    for i, row in enumerate(todo, 1):
        pending = [v for v in variants if row["id"] not in finished[v] and applicable(v, row)]
        try:
            ids, span_start = build_input_ids_with_span(tokenizer, row, filler_ids, args.filler_tokens)
            results = compute(model, ids.to(device), row["id"], pending, args.ratio, span_start, args.max_new_tokens,
                              eos_ids, not args.no_decode, steer)
        except Exception as e:
            results = {v: e for v in pending}
        for v, res in results.items():
            base = {k: row[k] for k in ("id", "prompt", "source", "label")} | meta | {"method": v}
            if isinstance(res, Exception):
                n_error += 1
                record = base | {"status": "error", "error": f"{type(res).__name__}: {res}"}
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                extras = dict(res["extras"])
                original, compressed = extras.pop("original_bytes", None), extras.pop("compressed_bytes", None)
                acts = {"last": res["last"].astype(np.float16)}
                if not args.no_decode:
                    acts["resp5"] = res["resp5"].astype(np.float16)
                np.savez_compressed(outs[v].with_name(outs[v].stem + "_acts") / f"{row['id']}.npz", **acts)
                finish = res["seconds_prefix"] + res["seconds_finish"]
                record = base | {
                    "response": tokenizer.decode(res["generated"], skip_special_tokens=True),
                    "n_new_tokens": len(res["generated"]),
                    "prompt_tokens": ids.shape[1],
                    "compression_ratio_measured": original / compressed if original else None,
                    "compressed_bytes": compressed,
                    "original_bytes": original,
                    "seconds_prefix": res["seconds_prefix"],
                    "seconds_compress_and_finish": finish,
                    "seconds_total": finish + res["seconds_decode"],
                } | extras | {"status": "ok"}
                seconds.setdefault(v, []).append(record["seconds_total"])
            files[v].write(json.dumps(record) + "\n")
            files[v].flush()
        if i % 20 == 0 or i == len(todo):
            print(f"{i}/{len(todo)} {(time.time() - start) / i:.2f} s/prompt errors={n_error}", flush=True)
    print("mean seconds_total per row: " + ", ".join(f"{v} {np.mean(s):.2f}" for v, s in seconds.items()), flush=True)
    return 1 if n_error else 0


if __name__ == "__main__":
    sys.exit(main())
