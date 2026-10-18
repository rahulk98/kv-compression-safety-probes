import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse(text):
    safety = re.search(r"Safety: (Safe|Unsafe|Controversial)", text)
    refusal = re.search(r"Refusal: (Yes|No)", text)
    return (safety.group(1) if safety else None), (refusal.group(1) if refusal else None)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


@torch.no_grad()
def judge_run(model, tokenizer, run_path, max_new_tokens=64):
    run_path = Path(run_path)
    out_path = run_path.with_suffix(".judged.jsonl")
    finished = {r["id"] for r in read_jsonl(out_path)}
    rows = [r for r in read_jsonl(run_path) if r["status"] == "ok" and r["id"] not in finished]
    with out_path.open("a") as f:
        for r in rows:
            messages = [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(model.device)
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            raw = tokenizer.decode(gen[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            safety, refusal = parse(raw)
            f.write(json.dumps(r | {"judge_safety": safety, "judge_refusal": refusal, "judge_raw": raw}) + "\n")
            f.flush()
    print(f"{out_path.name}: judged {len(rows)} new rows", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-dir", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("runs", nargs="+", help="run .jsonl files")
    args = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.judge_dir)
    model = AutoModelForCausalLM.from_pretrained(args.judge_dir, dtype=getattr(torch, args.dtype)).to(device).eval()
    for run in args.runs:
        judge_run(model, tokenizer, run)


if __name__ == "__main__":
    main()
