import argparse
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
from kvsafety.pipeline.data import build_input_ids, load_filler_ids


def make_cache(method, model, ratio):
    if method == "dense":
        return DynamicCache()
    if method == "snapkv":
        return SnapKVCache(model, ratio)
    from kvsafety.jolt_cache import make_jolt_cache

    return make_jolt_cache(model, ratio)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.time()


def last_hidden(out):
    return torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy()


@torch.no_grad()
def process(model, tokenizer, row, args, filler_ids, eos_ids, device):
    ids = build_input_ids(tokenizer, row, filler_ids, args.filler_tokens).to(device)
    cache = make_cache(args.method, model, args.ratio)
    try:
        t0 = sync(device)
        model(input_ids=ids[:, :-1], past_key_values=cache, use_cache=True, logits_to_keep=1)
        out = model(input_ids=ids[:, -1:], past_key_values=cache, use_cache=True, output_hidden_states=True)
        t1 = sync(device)
        last = last_hidden(out)
        generated, early = [], []
        for _ in range(args.max_new_tokens):
            token = out.logits[0, -1].argmax().item()
            if token in eos_ids:
                break
            generated.append(token)
            out = model(
                input_ids=torch.tensor([[token]], device=device),
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
            if len(early) < 5:
                early.append(last_hidden(out))
        resp5 = np.mean(early, axis=0) if early else last
        t2 = sync(device)
        original = getattr(cache, "last_original_bytes", None)
        compressed = getattr(cache, "last_compressed_bytes", None)
    finally:
        if hasattr(cache, "close"):
            cache.close()
        if hasattr(cache, "detach"):
            cache.detach()
    result = {
        "response": tokenizer.decode(generated, skip_special_tokens=True),
        "n_new_tokens": len(generated),
        "prompt_tokens": ids.shape[1],
        "compression_ratio_measured": original / compressed if args.method != "dense" else None,
        "compressed_bytes": compressed if args.method != "dense" else None,
        "original_bytes": original if args.method != "dense" else None,
        "seconds_compress_and_finish": t1 - t0,
        "seconds_total": t2 - t0,
    }
    return result, {"last": last.astype(np.float16), "resp5": resp5.astype(np.float16)}


def done_ids(path):
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["id"] for r in rows if r["status"] == "ok"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--set", required=True, choices=["eval", "train", "val"])
    ap.add_argument("--method", required=True, choices=["dense", "snapkv", "jolt"])
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--filler-tokens", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out")
    ap.add_argument("--sets-dir", default="data/sets")
    ap.add_argument("--filler-file", default="data/raw/wikitext_filler.txt")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args(argv)

    model_name = Path(args.model_dir).name
    out = Path(args.out or f"results/{model_name}/{args.set}_{args.method}_r{args.ratio:g}_f{args.filler_tokens}.jsonl")
    acts_dir = out.with_name(out.stem + "_acts")
    acts_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (Path(args.sets_dir) / f"{args.set}.jsonl").read_text().splitlines()]
    finished = done_ids(out)
    todo = [r for r in rows if r["id"] not in finished]
    print(f"{out.name}: {len(finished)} done, {len(todo)} to do", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=getattr(torch, args.dtype), attn_implementation="sdpa"
    ).to(device).eval()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    eos_ids.add(tokenizer.eos_token_id)
    filler_ids = load_filler_ids(tokenizer, args.filler_file) if args.filler_tokens > 0 else None

    meta = {
        "set": args.set,
        "method": args.method,
        "ratio": args.ratio,
        "filler_tokens": args.filler_tokens,
        "model_dir": args.model_dir,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "hostname": socket.gethostname(),
    }
    n_error, start = 0, time.time()
    with out.open("a") as f:
        for i, row in enumerate(todo, 1):
            base = {k: row[k] for k in ("id", "prompt", "source", "label")} | meta
            try:
                result, acts = process(model, tokenizer, row, args, filler_ids, eos_ids, device)
                np.savez_compressed(acts_dir / f"{row['id']}.npz", **acts)
                record = base | result | {"status": "ok"}
            except Exception as e:
                n_error += 1
                record = base | {"status": "error", "error": f"{type(e).__name__}: {e}"}
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            f.write(json.dumps(record) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} {(time.time() - start) / i:.2f} s/row errors={n_error}", flush=True)
    return 1 if n_error else 0


if __name__ == "__main__":
    sys.exit(main())
