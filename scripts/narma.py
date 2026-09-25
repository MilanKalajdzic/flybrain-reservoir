"""NARMA-10 benchmark: every wiring at many gains, judged at its best valid one.

    python scripts/narma.py --config configs/small.yaml --set n_jobs=4
    python scripts/narma.py --config configs/full.yaml --set reservoir.backend=torch

Same reservoirs as gain_sweep.py (seeds, input weights, readout neurons, latching tests). If gain_sweep.py
has been run for this config, NARMA error is also compared with memory capacity reservoir by reservoir.
Output in results/<name>/narma/: summary.md (start here), runs.csv (every run), grid.csv (mean per wiring
and gain), narma.png.
"""
import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from flyres import plotting  # noqa: E402
from flyres.config import load_config, save_config  # noqa: E402
from flyres.narma import best_narma, memory_link, narma_markdown, run_narma, summarize_narma  # noqa: E402
from flyres.sweep import DEFAULT_GAINS  # noqa: E402

warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given to the executor")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--gains", nargs="+", type=float, default=list(DEFAULT_GAINS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="seeds (default 0 1 2, like the gain sweep)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    cfg.seeds = args.seeds
    run_dir = Path(cfg.output_dir) / cfg.name
    out = run_dir / "narma"
    out.mkdir(parents=True, exist_ok=True)

    runs, baseline = run_narma(cfg, args.gains, verbose=not args.quiet)
    grid = summarize_narma(runs)
    best = best_narma(grid, cfg.reservoir.spectral_radius)
    sweep_path = run_dir / "gain_sweep" / "sweep.csv"
    link = memory_link(runs, pd.read_csv(sweep_path)) if sweep_path.exists() else None
    runs.to_csv(out / "runs.csv", index=False)
    grid.to_csv(out / "grid.csv", index=False)
    baseline.to_csv(out / "linear_baseline.csv", index=False)
    save_config(cfg, out / "config_used.yaml")
    (out / "summary.md").write_text(narma_markdown(cfg, grid, best, baseline, link), encoding="utf-8")
    plt.close(plotting.plot_narma(grid, baseline["nrmse"].mean(), path=out / "narma.png"))
    print(best.round(3).to_string(index=False))
    print(f"linear baseline: {baseline['nrmse'].mean():.3f}")
    print(f"open {out / 'summary.md'}")


if __name__ == "__main__":
    main()
