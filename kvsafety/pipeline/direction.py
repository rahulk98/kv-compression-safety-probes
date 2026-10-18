import argparse
from pathlib import Path

import numpy as np

from kvsafety.pipeline.analyze import fit_probe, load_run


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--filler-tokens", type=int, default=1024)
    args = ap.parse_args(argv)
    d, f = Path(args.results_dir), args.filler_tokens

    train_rows, train_acts = load_run(d, f"train_dense_r1_f{f}")
    val_rows, val_acts = load_run(d, f"val_dense_r1_f{f}")
    train_y = np.array([r["label"] for r in train_rows])
    layer, val_aurocs, _ = fit_probe(train_acts, train_y, val_acts, np.array([r["label"] for r in val_rows]))
    assert layer >= 1, "layer 0 is the embedding output; no decoder layer to steer"
    direction = train_acts[train_y == 1, layer].mean(0) - train_acts[train_y == 0, layer].mean(0)
    # `act_layer` indexes the activation stack (0 = embeddings), `layer` is the decoder layer whose output is that activation.
    np.savez(args.out, layer=layer - 1, act_layer=layer, direction=direction.astype(np.float32),
             norm=np.linalg.norm(direction), val_auroc=val_aurocs[layer])
    print(f"act_layer {layer} (decoder layer {layer - 1}), val AUROC {val_aurocs[layer]:.3f}, norm {np.linalg.norm(direction):.3f}")


if __name__ == "__main__":
    main()
