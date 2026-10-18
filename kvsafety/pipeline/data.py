import argparse
import csv
import json
import random
import re
from pathlib import Path

RAW = Path("data/raw")
SETS = Path("data/sets")
SEED = 0


def strip_prompt(text):
    return text.rstrip(" \t\n.!?").strip()


def normalise(text):
    return re.sub(r"\s+", " ", strip_prompt(text).lower().strip())


def load_json(name):
    return [r["instruction"] for r in json.loads((RAW / f"{name}.json").read_text())]


def sample(prompts, n, rng):
    return rng.sample(prompts, n)


def make_rows(prefix, source, label, prompts):
    return [
        {"id": f"{prefix}_{i}", "prompt": strip_prompt(p), "original": p, "label": label, "source": source}
        for i, p in enumerate(prompts)
    ]


def build_sets():
    rng = random.Random(SEED)
    train = make_rows("train_harmful", "harmful_train", 1, sample(load_json("harmful_train"), 128, rng)) + make_rows(
        "train_harmless", "harmless_train", 0, sample(load_json("harmless_train"), 128, rng)
    )
    val = make_rows("val_harmful", "harmful_val", 1, sample(load_json("harmful_val"), 32, rng)) + make_rows(
        "val_harmless", "harmless_val", 0, sample(load_json("harmless_val"), 32, rng)
    )
    xstest = list(csv.DictReader((RAW / "xstest_prompts.csv").open()))
    ev = (
        make_rows("eval_harmbench", "harmbench", 1, load_json("harmbench_test"))
        + make_rows("eval_jbb", "jailbreakbench", 1, load_json("jailbreakbench"))
        + make_rows("eval_harmless", "harmless_test", 0, sample(load_json("harmless_test"), 142, rng))
        + make_rows("eval_xstest_safe", "xstest_safe", 0, sample([r["prompt"] for r in xstest if r["label"] == "safe"], 100, rng))
    )
    seen = {normalise(r["prompt"]) for r in train + val}
    kept = [r for r in ev if normalise(r["prompt"]) not in seen]
    report = {"eval_before_dedup": count_sources(ev), "eval_after_dedup": count_sources(kept)}
    return {"train": train, "val": val, "eval": kept}, report


def count_sources(rows):
    counts = {}
    for r in rows:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    return counts


def load_filler_ids(tokenizer, path="data/raw/wikitext_filler.txt"):
    return tokenizer(Path(path).read_text(), add_special_tokens=False)["input_ids"]


def build_input_ids(tokenizer, row, filler_ids, filler_tokens, seed=SEED):
    """Chat-formatted token ids for one prompt row; the filler chunk depends only on (seed, row id)."""
    request = row["prompt"]
    if filler_tokens > 0:
        start = random.Random(f"{seed}:{row['id']}").randrange(len(filler_ids) - filler_tokens)
        doc = tokenizer.decode(filler_ids[start : start + filler_tokens])
        request = f"Here is a document:\n\n{doc}\n\n{request}"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": request}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]


def build_input_ids_with_span(tokenizer, row, filler_ids, filler_tokens, seed=SEED):
    """build_input_ids plus the index of the first token of the request text (needs a fast tokenizer).

    The request span is [start, T) (chat-template tail included), the document span is [0, start).
    """
    request = row["prompt"]
    if filler_tokens > 0:
        start = random.Random(f"{seed}:{row['id']}").randrange(len(filler_ids) - filler_tokens)
        doc = tokenizer.decode(filler_ids[start : start + filler_tokens])
        request = f"Here is a document:\n\n{doc}\n\n{request}"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": request}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    enc = tokenizer(text, add_special_tokens=False, return_tensors="pt", return_offsets_mapping=True)
    char_start = text.rindex(row["prompt"])
    return enc["input_ids"], int((enc["offset_mapping"][0, :, 1] > char_start).nonzero()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(SETS))
    out_dir = Path(ap.parse_args().out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sets, report = build_sets()
    for name, rows in sets.items():
        (out_dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(name, len(rows), count_sources(rows))
    (out_dir / "dedup_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
