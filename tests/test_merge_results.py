"""scripts/merge_results.py selection rules on a synthetic results_sync (CPU)."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import merge_results as mr

BASE = "eval_dense_r1_f1024"


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def box(sync, name, rows, judged=(), acts=()):
    d = sync / name / "M"
    write(d / f"{BASE}.jsonl", rows)
    if judged:
        write(d / f"{BASE}.judged.jsonl", judged)
    for i in acts:
        (d / f"{BASE}_acts").mkdir(exist_ok=True)
        np.savez(d / f"{BASE}_acts" / f"{i}.npz", last=np.zeros(1))


def row(i, resp, **kw):
    return {"id": i, "status": "ok", "response": resp} | kw


def test_selection(tmp_path):
    sets = tmp_path / "sets"
    ids = ["a", "b", "c"]
    for s in mr.SETS:
        write(sets / f"{s}.jsonl", [{"id": i, "label": 1} for i in ids])
    sync = tmp_path / "sync"
    # a: box1 has a copied_from marker, box2 is real; b: box1 has no acts, box2 has acts; c: judged only on box1 with the same response
    box(sync, "box1", [row("a", "x", copied_from="box2"), row("b", "old"), row("c", "same")], judged=[row("c", "same", judge_refusal="Yes")], acts=["c"])
    box(sync, "box2", [row("a", "x"), row("b", "new"), row("c", "same")], acts=["a", "b", "c"])
    out = tmp_path / "out"
    mr.main(["--sync-dir", str(sync), "--out", str(out), "--sets-dir", str(sets)])
    rows = {r["id"]: r for r in mr.read_jsonl(out / "M" / f"{BASE}.jsonl")}
    assert rows["b"]["response"] == "new" and "copied_from" not in rows["a"]
    assert all((out / "M" / f"{BASE}_acts" / f"{i}.npz").exists() for i in ids)
    # c is chosen from box1 (acts on both, sorted order), so its own judged row is used
    assert [r["id"] for r in mr.read_jsonl(out / "M" / f"{BASE}.judged.jsonl")] == ["c"]
    assert "| 3 | 3 | 0 | 3 | 1 |" in (out / "merge_report.md").read_text()


def test_judged_fallback_needs_identical_response(tmp_path):
    sets = tmp_path / "sets"
    for s in mr.SETS:
        write(sets / f"{s}.jsonl", [{"id": i, "label": 1} for i in "ab"])
    sync = tmp_path / "sync"
    box(sync, "box1", [row("a", "r1"), row("b", "r2")], acts=["a", "b"])
    box(sync, "box2", [row("a", "r1"), row("b", "other")], judged=[row("a", "r1", judge_refusal="No"), row("b", "other", judge_refusal="No")])
    out = tmp_path / "out"
    mr.main(["--sync-dir", str(sync), "--out", str(out), "--sets-dir", str(sets)])
    assert [r["id"] for r in mr.read_jsonl(out / "M" / f"{BASE}.judged.jsonl")] == ["a"]
    assert "| 2 | 2 | 0 | 2 | 1 | 2 | 0 | 1 |" in (out / "merge_report.md").read_text()
