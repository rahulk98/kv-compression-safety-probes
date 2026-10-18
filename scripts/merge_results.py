"""Merge per-box results (results_sync/<alias>/<model>/) into one analyze.py-ready dir per model."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

SETS = ("train", "val", "eval")
COLS = ("base", "rows", "expected", "missing", "with_acts", "judged", "dups", "conflicts", "missing_judged")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def last_by_id(path):
    return {r["id"]: r for r in read_jsonl(path)}


def merge_base(model_dir, sources, base, set_rows, out_dir):
    """Choose one row per id across sources; link acts, write rows and judged rows. Returns the report row."""
    runs = {s: {i: r for i, r in last_by_id(s / model_dir / f"{base}.jsonl").items() if r["status"] == "ok" and "copied_from" not in r} for s in sources}
    judged = {s: last_by_id(s / model_dir / f"{base}.judged.jsonl") for s in sources}
    any_judged = any(judged.values())
    expected = [r["id"] for r in set_rows if "_steer" not in base or r["label"] == 1]
    out = out_dir / model_dir / base
    chosen, acts, judged_out = [], [], []
    n = dict.fromkeys(("dups", "conflicts", "missing_judged"), 0)
    for i in (r["id"] for r in set_rows):
        cands = [s for s in sources if i in runs[s]]
        if not cands:
            continue
        has_acts = [s for s in cands if (s / model_dir / f"{base}_acts" / f"{i}.npz").exists()]
        pool = has_acts or cands
        src = pool[0]
        n["dups"] += len(cands) > 1
        n["conflicts"] += any(runs[s][i]["response"] != runs[src][i]["response"] for s in pool[1:])
        chosen.append(runs[src][i])
        if has_acts:
            acts.append((i, src / model_dir / f"{base}_acts" / f"{i}.npz"))
        if any_judged:
            j = judged[src].get(i) or next((judged[s][i] for s in sources if i in judged[s] and judged[s][i]["response"] == runs[src][i]["response"]), None)
            judged_out.append(j) if j else n.__setitem__("missing_judged", n["missing_judged"] + 1)
    for p in (out.with_suffix(".jsonl"), out.with_suffix(".judged.jsonl")):
        p.unlink(missing_ok=True)
    shutil.rmtree(f"{out}_acts", ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".jsonl").write_text("".join(json.dumps(r) + "\n" for r in chosen))
    if judged_out:
        out.with_suffix(".judged.jsonl").write_text("".join(json.dumps(r) + "\n" for r in judged_out))
    Path(f"{out}_acts").mkdir()
    for i, p in acts:
        (Path(f"{out}_acts") / f"{i}.npz").symlink_to(p.resolve())
    return (base, len(chosen), len(expected), len(set(expected) - {r["id"] for r in chosen}), len(acts), len(judged_out), n["dups"], n["conflicts"], n["missing_judged"] if any_judged else "n/a")


def table(rows):
    return "\n".join(["| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--sets-dir", default="data/sets")
    args = ap.parse_args(argv)
    sync, out = Path(args.sync_dir), Path(args.out)
    sets = {s: read_jsonl(Path(args.sets_dir) / f"{s}.jsonl") for s in SETS}
    assert all(sets.values()), f"missing set files in {args.sets_dir}; build them with python -m kvsafety.pipeline.data"
    set_md5 = {f"{s}.jsonl": md5(Path(args.sets_dir) / f"{s}.jsonl") for s in SETS}
    boxes = sorted(p for p in sync.iterdir() if p.is_dir())
    models = args.models or sorted({m.name for b in boxes for m in b.iterdir() if m.is_dir()})
    lines, checked = [], 0
    for model in models:
        sources = [b for b in boxes if (b / model).is_dir()]
        for s in sources:
            f = s / model / "logs" / "sets.md5"
            checked += f.exists()
            for line in f.read_text().splitlines() if f.exists() else []:
                h, name = line.split()
                if set_md5.get(Path(name).name) != h:
                    lines.append(f"SETS MD5 MISMATCH {s.name}/{model}: {name} {h} vs repo {set_md5.get(Path(name).name)}")
        bases = sorted({p.name[: -len(".jsonl")] for s in sources for p in (s / model).glob("*.jsonl") if not p.name.endswith(".judged.jsonl")})
        report = [merge_base(model, sources, b, sets[b.split("_")[0]], out) for b in bases if b.split("_")[0] in SETS]
        rd = {s.name: md5(s / model / "refusal_direction.npz") for s in sources if (s / model / "refusal_direction.npz").exists()}
        if rd:
            first = min(rd)
            shutil.copy(sync / first / model / "refusal_direction.npz", out / model / "refusal_direction.npz")
        lines += [f"\n## {model} ({len(sources)} source dirs)\n", table(report),
                  f"\nrefusal_direction.npz: {len(rd)} sources, {len(set(rd.values()))} distinct md5" + (f", copied from {first}" if rd else "")]
        if len(set(rd.values())) > 1:
            lines += [f"  {k} {v}" for k, v in sorted(rd.items())]
    sets_check = [l for l in lines if l.startswith("SETS MD5")] or [f"sets.md5: {checked} files checked, all agree with repo sets"]
    text = "# merge_report\n\n" + "\n".join(sets_check) + "\n" + "\n".join(l for l in lines if not l.startswith("SETS MD5")) + "\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "merge_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
