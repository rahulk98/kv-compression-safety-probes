"""run_controls on tiny random Qwen3 / Llama models (CPU): equivalence with run.py and the control invariants."""

import glob
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from kvsafety import SnapKVCache
from kvsafety.pipeline import analyze, data, direction, run, run_controls as rc

TOKENIZERS = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-*/snapshots/*"))
needs_tokenizer = pytest.mark.skipif(not TOKENIZERS, reason="no local Qwen3 tokenizer")
COMMON = dict(
    vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=2048,
)
T, STEPS, NO_EOS = 160, 8, {-1}


def tiny_model(kind):
    torch.manual_seed(0)
    return (Qwen3ForCausalLM(Qwen3Config(**COMMON)) if kind == "qwen3" else LlamaForCausalLM(LlamaConfig(**COMMON))).eval()


def make_ids(n=T):
    return torch.randint(0, 128, (1, n), generator=torch.Generator().manual_seed(1))


class Tok:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


def run_py(monkeypatch, model, ids, method, ratio):
    """run.py's path for one prompt: (generated token ids, activations)."""
    monkeypatch.setattr(run, "build_input_ids", lambda *a, **k: ids)
    torch.manual_seed(rc.seed("x", "jolt", f"{ratio:g}"))  # JoLT compression draws from the global RNG; match the seed run_controls sets
    args = SimpleNamespace(method=method, ratio=ratio, filler_tokens=0, max_new_tokens=STEPS)
    result, acts = run.process(model, Tok(), {"id": "x", "prompt": "p"}, args, None, NO_EOS, ids.device)
    return [int(t) for t in result["response"].split()], acts, result


def mine(model, ids, variants, ratio, span=100, **kw):
    out = rc.compute(model, ids, "x", variants, ratio, span, STEPS, NO_EOS, **kw)
    for v in out.values():
        if isinstance(v, Exception):
            raise v
    return out


@pytest.mark.parametrize("kind", ["qwen3", "llama"])
@pytest.mark.parametrize("variant,ratio", [("dense", 1.0), ("snapkv", 2.0), ("jolt", 4.0)])
def test_equivalent_to_run_py(monkeypatch, kind, variant, ratio):
    model, ids = tiny_model(kind), make_ids()
    tokens, acts, result = run_py(monkeypatch, model, ids, variant, ratio)
    res = mine(model, ids, [variant], ratio)[variant]
    mine16 = {k: res[k].astype(np.float16).astype(np.float32) for k in ("last", "resp5")}  # as main() stores them
    diff = {k: np.abs(mine16[k] - acts[k].astype(np.float32)).max() for k in mine16}
    print(f"\n{kind} {variant}: max |last| diff {diff['last']:.2e}, max |resp5| diff {diff['resp5']:.2e}")
    assert res["generated"] == tokens
    assert np.allclose(mine16["last"], acts["last"].astype(np.float32), atol=1e-6, rtol=0)
    assert np.allclose(mine16["resp5"], acts["resp5"].astype(np.float32), atol=1e-6, rtol=0)
    if variant != "dense":
        extras = res["extras"]
        assert extras["original_bytes"] / extras["compressed_bytes"] == pytest.approx(result["compression_ratio_measured"])
        assert extras["compressed_bytes"] == result["compressed_bytes"]


def prompt(kind="qwen3", ratio=4.0, span=100, snap=True):
    return rc.Prompt(tiny_model(kind), make_ids(), "x", ratio, span, snap)


@pytest.mark.parametrize("kind", ["qwen3", "llama"])
def test_noise_matches_jolt_error(kind):
    p = prompt(kind, snap=False)
    dense, _, _ = p.prefix("dense")
    noise, extras, _ = p.prefix("noise")
    assert min(extras["rel_err_k"]) > 0
    for i, ((k, v), (nk, nv)) in enumerate(zip(dense, noise)):
        assert rc.rel_err(nk, k) == pytest.approx(extras["rel_err_k"][i], abs=1e-3)
        assert rc.rel_err(nv, v) == pytest.approx(extras["rel_err_v"][i], abs=1e-3)
    assert not torch.equal(noise[0][0], dense[0][0])
    again = rc.Prompt(p.model, p.ids, "x", 4.0, 100, False).prefix("noise")[0]
    assert all(torch.equal(a[0], b[0]) for a, b in zip(noise, again))  # seeded by (id, layer, k/v, ratio)


def test_random_matches_snapkv_count_and_window():
    ratio = 2.0
    p = prompt(ratio=ratio)
    dense, _, _ = p.prefix("dense")
    snap, snap_extras, _ = p.prefix("snapkv")
    rand, rand_extras, _ = p.prefix("random")
    cache = SnapKVCache(p.model, ratio)
    with torch.no_grad():
        p.model(input_ids=p.ids[:, :-1], past_key_values=cache, use_cache=True)
        p.model(input_ids=p.ids[:, -1:], past_key_values=cache, use_cache=True)
    cache.close()
    kept = cache.layers[0].keys.shape[2] - 1
    assert kept == snap[0][0].shape[2] == rand[0][0].shape[2] == snap_extras["kept_tokens"] == rand_extras["kept_tokens"]
    for (sk, sv), (rk, rv), (dk, dv), layer in zip(snap, rand, dense, cache.layers):
        assert torch.equal(sk, layer.keys[:, :, :-1]) and torch.equal(sv, layer.values[:, :, :-1])
        for kept_k, kept_v, full_k, full_v in ((sk, sv, dk, dv), (rk, rv, dk, dv)):
            assert torch.equal(kept_k[:, :, -rc.WINDOW:], full_k[:, :, -rc.WINDOW:])
            assert torch.equal(kept_v[:, :, -rc.WINDOW:], full_v[:, :, -rc.WINDOW:])
    assert not all(torch.equal(a[0], b[0]) for a, b in zip(snap, rand))


def test_counterfactuals_cover_every_position_once():
    ratio, span = 2.0, 100
    p = prompt(ratio=ratio, span=span)
    L = p.L
    dense, _, _ = p.prefix("dense")
    jolt, _, _ = p.prefix("jolt")
    doc_only, e1, _ = p.prefix("jolt_doconly")
    req_only, e2, _ = p.prefix("jolt_reqonly")
    assert e1["n_document_tokens"] + e1["n_request_tokens"] == L
    for i in range(len(dense)):
        assert torch.equal(doc_only[i][0][:, :, :span], jolt[i][0][:, :, :span])
        assert torch.equal(doc_only[i][0][:, :, span:], dense[i][0][:, :, span:])
        assert torch.equal(req_only[i][0][:, :, :span], dense[i][0][:, :, :span])
        assert torch.equal(req_only[i][1][:, :, span:], jolt[i][1][:, :, span:])
    n_doc = span
    snap_doc, d, _ = p.prefix("snapkv_doconly")
    snap_req, r, _ = p.prefix("snapkv_reqonly")
    assert d["kept_request_tokens"] == L - span and d["kept_document_tokens"] == round(n_doc / ratio)
    assert r["kept_document_tokens"] == n_doc and r["kept_request_tokens"] == round((L - span) / ratio)
    for i in range(len(dense)):  # kept request span / document are exact copies of dense positions, in order
        assert torch.equal(snap_doc[i][0][:, :, -(L - span):], dense[i][0][:, :, span:])
        assert torch.equal(snap_req[i][0][:, :, :span], dense[i][0][:, :, :span])


def test_steering_zero_equals_none_and_nonzero_differs():
    model, ids = tiny_model("qwen3"), make_ids()
    direction_ = torch.randn(COMMON["hidden_size"], generator=torch.Generator().manual_seed(3))
    out = mine(model, ids, ["jolt", "jolt_steer0", "snapkv_steer0", "snapkv", "jolt_steer8"], 4.0, steer=(1, direction_))
    assert out["jolt_steer0"]["generated"] == out["jolt"]["generated"]
    assert np.array_equal(out["jolt_steer0"]["last"], out["jolt"]["last"])
    assert np.array_equal(out["snapkv_steer0"]["last"], out["snapkv"]["last"])
    assert not np.array_equal(out["jolt_steer8"]["last"][2:], out["jolt"]["last"][2:])
    assert np.array_equal(out["jolt_steer8"]["last"][:2], out["jolt"]["last"][:2])  # layers before the hook are untouched
    assert out["jolt_steer8"]["extras"]["steer_layer"] == 1


def test_no_decode_and_prefix_untouched():
    model, ids = tiny_model("qwen3"), make_ids()
    p = rc.Prompt(model, ids, "x", 2.0, 100, True)
    before = [k.clone() for k, _ in p.prefix("snapkv")[0]]
    out = mine(model, ids, ["snapkv"], 2.0, decode=False)["snapkv"]
    assert out["generated"] == [] and np.array_equal(out["resp5"], out["last"])
    full = mine(model, ids, ["snapkv"], 2.0)["snapkv"]
    assert np.array_equal(full["last"], out["last"])
    assert all(torch.equal(a, k) for a, (k, _) in zip(before, p.prefix("snapkv")[0]))


# ---- end to end through main() ----

def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    if not TOKENIZERS:
        pytest.skip("no local Qwen3 tokenizer")
    d = tmp_path_factory.mktemp("tiny")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZERS[0])
    tokenizer.save_pretrained(d / "Tiny")
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=1024,
    )
    Qwen3ForCausalLM(config).save_pretrained(d / "Tiny")
    (d / "filler.txt").write_text(" ".join(f"word{i} lorem ipsum dolor sit amet." for i in range(400)))
    rows = [{"id": f"p{i}", "prompt": f"Tell me how to do thing number {i}", "original": "x", "label": i % 2,
             "source": ["harmbench", "harmless_test"][i % 2]} for i in range(5)]
    (d / "eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    direction_ = np.random.default_rng(0).normal(size=32).astype(np.float32)
    np.savez(d / "direction.npz", layer=1, act_layer=2, direction=direction_)
    return d


def controls_args(d, results, *extra, ratio="2"):
    return ["--model-dir", str(d / "Tiny"), "--set", "eval", "--ratio", ratio, "--filler-tokens", "64",
            "--max-new-tokens", "6", "--sets-dir", str(d), "--filler-file", str(d / "filler.txt"), "--dtype", "float32",
            "--results-dir", str(results), *extra]


@needs_tokenizer
def test_span_and_ids(tiny_dir):
    tok = AutoTokenizer.from_pretrained(tiny_dir / "Tiny")
    filler = data.load_filler_ids(tok, tiny_dir / "filler.txt")
    row = {"id": "p1", "prompt": "Tell me how to do thing number 1"}
    ids, start = data.build_input_ids_with_span(tok, row, filler, 64)
    assert torch.equal(ids, data.build_input_ids(tok, row, filler, 64))
    text_from_span = tok.decode(ids[0, start:])
    assert text_from_span.startswith(row["prompt"][:5]) and row["prompt"] in text_from_span
    assert "Here is a document" in tok.decode(ids[0, :start]) and row["prompt"] not in tok.decode(ids[0, :start])
    # the span partition of the prefix: [0, start) and [start, T-1) cover each position once
    covered = list(range(start)) + list(range(start, ids.shape[1] - 1))
    assert covered == list(range(ids.shape[1] - 1))


@needs_tokenizer
def test_main_resume_shard_partial_and_compat(tiny_dir, tmp_path):
    results = tmp_path / "res"
    # run.py arms first; run_controls must treat their rows as done and write to the same files
    for method in ("snapkv", "jolt"):
        out = results / f"eval_{method}_r2_f64.jsonl"
        assert run.main(["--model-dir", str(tiny_dir / "Tiny"), "--set", "eval", "--method", method, "--ratio", "2",
                         "--filler-tokens", "64", "--max-new-tokens", "6", "--sets-dir", str(tiny_dir),
                         "--filler-file", str(tiny_dir / "filler.txt"), "--dtype", "float32", "--out", str(out)]) == 0
    before = {m: (results / f"eval_{m}_r2_f64.jsonl").read_text() for m in ("snapkv", "jolt")}

    variants = "dense,jolt,noise,snapkv,random"
    assert rc.main(controls_args(tiny_dir, results, "--variants", variants, "--shard", "0/2")) == 0
    assert rc.main(controls_args(tiny_dir, results, "--variants", variants, "--shard", "1/2")) == 0
    for v in ("noise", "random"):
        rows = read(results / f"eval_{v}_r2_f64.jsonl")
        assert sorted(r["id"] for r in rows) == [f"p{i}" for i in range(5)]  # shards cover every row once
        assert {r["status"] for r in rows} == {"ok"} and all(r["method"] == v for r in rows)
        z = np.load(results / f"eval_{v}_r2_f64_acts" / "p0.npz")
        assert z["last"].shape == (3, 32) and z["resp5"].dtype == np.float16
    assert not list(results.glob("eval_dense_*"))  # ratio 2: dense is not written
    for m in ("snapkv", "jolt"):
        assert (results / f"eval_{m}_r2_f64.jsonl").read_text() == before[m]  # run.py rows count as done
    assert all("rel_err_k" in r for r in read(results / "eval_noise_r2_f64.jsonl"))

    # resume: nothing missing, nothing written
    sizes = {p: p.stat().st_size for p in results.glob("*.jsonl")}
    assert rc.main(controls_args(tiny_dir, results, "--variants", variants)) == 0
    assert {p: p.stat().st_size for p in results.glob("*.jsonl")} == sizes

    # drop one row and add a partial trailing line: only the dropped row is recomputed, the file stays strictly parseable
    path = results / "eval_random_r2_f64.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + '\n{"id": "p9", "stat')
    assert rc.done_ids(path) == {json.loads(l)["id"] for l in lines[:-1]}
    assert rc.main(controls_args(tiny_dir, results, "--variants", variants)) == 0
    rows = read(path)  # strict json.loads on every line
    assert sorted(r["id"] for r in rows) == [f"p{i}" for i in range(5)]
    assert len(read(results / "eval_noise_r2_f64.jsonl")) == 5

    # counterfactual and steering, harmful prompts only for steering; --no-decode on a train-like set
    assert rc.main(controls_args(tiny_dir, results, "--variants", "jolt", "--counterfactual", "--steer", "4,8",
                                 "--steer-direction", str(tiny_dir / "direction.npz"), ratio="2")) == 0
    for v in rc.COUNTERFACTUAL:
        assert len(read(results / f"eval_{v}_r2_f64.jsonl")) == 5
    steered = read(results / "eval_jolt_steer4_r2_f64.jsonl")
    assert {r["label"] for r in steered} == {1} and len(steered) == 2 and steered[0]["steer_strength"] == 4.0
    assert (results / "eval_snapkv_steer8_r2_f64.jsonl").exists()
    assert rc.main(controls_args(tiny_dir, results, "--variants", "noise", "--no-decode", ratio="4")) == 0
    row = read(results / "eval_noise_r4_f64.jsonl")[0]
    assert row["n_new_tokens"] == 0 and row["response"] == ""
    assert "resp5" not in np.load(results / "eval_noise_r4_f64_acts" / "p0.npz").files


# ---- analyze / direction ----

def synth(d, name, rows, rng, shift, extra=None):
    (d / f"{name}_acts").mkdir()
    out = []
    for r in rows:
        out.append(dict(r, status="ok", method=name.split("_")[1], ratio=1.0, compression_ratio_measured=None, **(extra or {})))
        vec = (rng.normal(size=(3, 8)) + shift * r["label"]).astype(np.float16)
        np.savez_compressed(d / f"{name}_acts" / f"{r['id']}.npz", last=vec, resp5=vec)
    (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))


def test_analyze_controls_and_direction(tmp_path):
    rng = np.random.default_rng(0)
    mk = lambda pre, src, label, n: [{"id": f"{pre}{src}{i}", "prompt": f"{src} {i}", "source": src, "label": label} for i in range(n)]
    train = mk("t", "harmful_train", 1, 20) + mk("t", "harmless_train", 0, 20)
    val = mk("v", "harmful_val", 1, 10) + mk("v", "harmless_val", 0, 10)
    ev = mk("e", "harmbench", 1, 15) + mk("e", "harmless_test", 0, 25)
    for name, rows, shift in (("train_dense_r1_f64", train, 3), ("val_dense_r1_f64", val, 3), ("eval_dense_r1_f64", ev, 3),
                              ("train_noise_r2_f64", train, 1), ("val_noise_r2_f64", val, 1)):
        synth(tmp_path, name, rows, rng, shift)
    synth(tmp_path, "eval_noise_r2_f64", ev, rng, 1, extra={"rel_err_k": [0.1, 0.2, 0.3], "rel_err_v": [0.2, 0.3, 0.4]})
    synth(tmp_path, "eval_jolt_steer4_r2_f64", ev[:15], rng, 1)  # harmful only: single class, AUROC undefined
    analyze.main(["--results-dir", str(tmp_path), "--filler-tokens", "64", "--n-boot", "50"])
    s = json.loads((tmp_path / "summary.json").read_text())
    noise = s["runs"]["eval_noise_r2_f64"]
    assert noise["rel_err_mean"]["k"] == pytest.approx([0.1, 0.2, 0.3])
    assert noise["retrained_probe"]["n_train"] == 40 and np.isfinite(noise["retrained_probe"]["probe_auroc"]["overall"]["value"])
    assert "retrained_probe" not in s["runs"]["eval_dense_r1_f64"]
    dense_proj = s["runs"]["eval_dense_r1_f64"]["refusal_projection"]
    assert dense_proj["harmful"]["value"] > dense_proj["harmless"]["value"]
    assert s["runs"]["eval_jolt_steer4_r2_f64"]["refusal_projection"]["harmless"] is None
    md = (tmp_path / "summary.md").read_text()
    assert "| eval_noise_r2_f64 |" in md and "unit refusal direction" in md

    direction.main(["--results-dir", str(tmp_path), "--out", str(tmp_path / "dir.npz"), "--filler-tokens", "64"])
    z = np.load(tmp_path / "dir.npz")
    assert z["direction"].dtype == np.float32 and z["direction"].shape == (8,)
    assert int(z["layer"]) == int(z["act_layer"]) - 1 and int(z["act_layer"]) == s["layer"]
    assert float(z["norm"]) == pytest.approx(np.linalg.norm(z["direction"]), rel=1e-5)
