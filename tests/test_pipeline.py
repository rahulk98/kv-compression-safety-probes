import glob
import importlib.util
import json

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from kvsafety.pipeline import analyze, data, judge, run

TOKENIZERS = glob.glob(str(__import__("pathlib").Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-*/snapshots/*"))
needs_tokenizer = pytest.mark.skipif(not TOKENIZERS, reason="no local Qwen3 tokenizer")
FILLER_TOKENS = 64


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZERS[0])
    tokenizer.save_pretrained(d / "Tiny")
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=1024,
    )
    Qwen3ForCausalLM(config).save_pretrained(d / "Tiny")
    filler = d / "filler.txt"
    filler.write_text(" ".join(f"word{i} lorem ipsum dolor sit amet." for i in range(400)))
    rows = [
        {"id": f"p{i}", "prompt": f"Tell me how to do thing number {i}", "original": "x", "label": i % 2,
         "source": ["harmbench", "harmless_test"][i % 2]}
        for i in range(4)
    ]
    (d / "eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def run_args(tiny, method, ratio=1.0, out=None):
    return [
        "--model-dir", str(tiny / "Tiny"), "--set", "eval", "--method", method, "--ratio", str(ratio),
        "--filler-tokens", str(FILLER_TOKENS), "--max-new-tokens", "8", "--sets-dir", str(tiny),
        "--filler-file", str(tiny / "filler.txt"), "--dtype", "float32", "--out", str(out),
    ]


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_strip_and_normalise():
    assert data.strip_prompt("  Do it?!. ") == "Do it"
    assert data.normalise("Do   IT!") == data.normalise("do it?")


def test_build_sets_dedup():
    sets, report = data.build_sets()
    assert {k: len(v) for k, v in sets.items()}["train"] == 256
    assert len(sets["val"]) == 64
    assert sum(report["eval_after_dedup"].values()) == len(sets["eval"])
    norm = lambda rows: {data.normalise(r["prompt"]) for r in rows}
    assert not norm(sets["eval"]) & (norm(sets["train"]) | norm(sets["val"]))
    assert all(r["prompt"] == data.strip_prompt(r["prompt"]) for rows in sets.values() for r in rows)


@needs_tokenizer
def test_filler_deterministic_and_f0(tiny):
    tok = AutoTokenizer.from_pretrained(tiny / "Tiny")
    filler = data.load_filler_ids(tok, tiny / "filler.txt")
    row = {"id": "a", "prompt": "Hello"}
    a = data.build_input_ids(tok, row, filler, 32)
    assert torch.equal(a, data.build_input_ids(tok, row, filler, 32))
    assert not torch.equal(a, data.build_input_ids(tok, {"id": "b", "prompt": "Hello"}, filler, 32))
    assert data.build_input_ids(tok, row, None, 0).shape[1] < 20


@needs_tokenizer
def test_pipeline_end_to_end(tiny):
    layers, d = 3, 32
    rows_by_method = {}
    methods = ["dense", "snapkv"] + (["jolt"] if importlib.util.find_spec("kvsafety.jolt_cache") else [])
    for method in methods:
        out = tiny / f"eval_{method}.jsonl"
        assert run.main(run_args(tiny, method, 1.0 if method == "dense" else 2.0, out)) == 0
        rows = read(out)
        assert [r["status"] for r in rows] == ["ok"] * 4
        assert rows[0]["method"] == method and rows[0]["prompt_tokens"] > FILLER_TOKENS
        rows_by_method[method] = rows
        for r in rows:
            z = np.load(tiny / f"eval_{method}_acts" / f"{r['id']}.npz")
            assert z["last"].shape == (layers, d) and z["resp5"].shape == (layers, d)
            assert z["last"].dtype == np.float16 and np.isfinite(z["last"].astype(np.float32)).all()
        assert run.main(run_args(tiny, method, 1.0 if method == "dense" else 2.0, out)) == 0
        assert len(read(out)) == 4  # resume skipped every id
    assert all(r["compression_ratio_measured"] is None for r in rows_by_method["dense"])
    assert all(r["compression_ratio_measured"] > 1 for r in rows_by_method["snapkv"])
    # compress-then-finish: the SnapKV probe activation differs from the dense one
    dense = np.load(tiny / "eval_dense_acts" / "p0.npz")["last"].astype(np.float32)
    snap = np.load(tiny / "eval_snapkv_acts" / "p0.npz")["last"].astype(np.float32)
    assert not np.allclose(dense[1:], snap[1:])
    assert np.allclose(dense[0], snap[0])  # embedding layer is cache independent


def test_judge_parse():
    assert judge.parse("Safety: Unsafe\nCategories: Violent\nRefusal: No") == ("Unsafe", "No")
    assert judge.parse("garbage") == (None, None)


def synth_run(d, name, rows, rng, shift, judged=True):
    (d / f"{name}_acts").mkdir()
    out, jout = [], []
    for r in rows:
        row = dict(r, status="ok", method=name.split("_")[1], ratio=1.0, compression_ratio_measured=None)
        out.append(row)
        vec = rng.normal(size=(3, 8)) + shift * r["label"]
        np.savez_compressed(d / f"{name}_acts" / f"{r['id']}.npz", last=vec.astype(np.float16), resp5=vec.astype(np.float16))
        jout.append(row | {"judge_safety": rng.choice(["Safe", "Unsafe"]), "judge_refusal": rng.choice(["Yes", "No"])})
    (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    if judged:
        (d / f"{name}.judged.jsonl").write_text("".join(json.dumps(r) + "\n" for r in jout))


def test_analyze_synthetic(tmp_path):
    rng = np.random.default_rng(0)
    mk = lambda pre, src, label, n: [{"id": f"{pre}{src}{i}", "prompt": f"{src} text {i} {label}", "source": src, "label": label} for i in range(n)]
    train = mk("t", "harmful_train", 1, 20) + mk("t", "harmless_train", 0, 20)
    val = mk("v", "harmful_val", 1, 10) + mk("v", "harmless_val", 0, 10)
    ev = (mk("e", "harmbench", 1, 15) + mk("e", "jailbreakbench", 1, 10) + mk("e", "harmless_test", 0, 25)
          + mk("e", "xstest_safe", 0, 10) + mk("e", "xstest_unsafe", 1, 10))
    synth_run(tmp_path, "train_dense_r1_f64", train, rng, 3, judged=False)
    synth_run(tmp_path, "val_dense_r1_f64", val, rng, 3, judged=False)
    synth_run(tmp_path, "eval_dense_r1_f64", ev, rng, 3)
    synth_run(tmp_path, "eval_snapkv_r2_f64", ev, rng, 1)
    analyze.main(["--results-dir", str(tmp_path), "--filler-tokens", "64", "--n-boot", "50"])
    s = json.loads((tmp_path / "summary.json").read_text())
    assert set(s["runs"]) == {"eval_dense_r1_f64", "eval_snapkv_r2_f64"}
    for run_summary in s["runs"].values():
        for m in [run_summary["probe_auroc"]["overall"], run_summary["probe_auroc"]["harmful_vs_harmless_test"],
                  run_summary["probe_auroc"]["xstest_unsafe_vs_safe"], run_summary["behaviour"]["refusal_rate_harmful"],
                  run_summary["behaviour"]["over_refusal_xstest_safe"], run_summary["silent_failure"]]:
            assert np.isfinite([m["value"], m["lo"], m["hi"]]).all()
    assert s["runs"]["eval_dense_r1_f64"]["probe_auroc"]["overall"]["value"] > s["runs"]["eval_snapkv_r2_f64"]["probe_auroc"]["overall"]["value"]
    assert np.isfinite(s["text_baseline"]["overall"]["value"])
    assert "| eval_dense_r1_f64 |" in (tmp_path / "summary.md").read_text()
