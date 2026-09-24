"""Where does the dominant eigenvalue live? Find the hot spot, remove it, repeat.

    python scripts/hot_spots.py --config configs/full.yaml
    python scripts/hot_spots.py --config configs/full.yaml --wiring degree_preserving

The spectral-radius rescaling divides every weight by the largest eigenvalue. If that eigenvalue
belongs to a small dense cluster, the whole network is turned down to suit that cluster. This
prints the chain of such clusters: after removing one, which one sets the gain next.
Output also saved to results/<name>/hot_spots_<wiring>.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from flyres.config import load_config
from flyres.diagnostics import hot_spot_cascade
from flyres.experiment import build_matrix, prepare_subgraph


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/full.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--wiring", default="connectome")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=8)
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    sub = prepare_subgraph(cfg, verbose=True)
    W, _, _, raw = build_matrix(sub, cfg, args.wiring, args.seed)
    table = hot_spot_cascade(W, sub.neurons, steps=args.steps, seed=args.seed)
    # eigenvalues of the rescaled matrix; multiply back to raw units so they match the summary's raw spectral radius
    table["top_abs_eigenvalue"] *= raw / cfg.reservoir.spectral_radius
    out = Path(cfg.output_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"hot_spots_{args.wiring}.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(table.assign(top_abs_eigenvalue=np.round(table["top_abs_eigenvalue"], 1),
                           spread=np.round(table["spread"]).astype(int)).to_string(index=False))


if __name__ == "__main__":
    main()
