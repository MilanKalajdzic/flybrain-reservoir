"""Robustness checks for the memory result: rerun the gain sweep with the input fed into randomly chosen
neurons, and with leaky neurons (leak rate 0.5 and 0.2), and compare every wiring at its best valid gain.

    python scripts/robustness.py --config configs/small.yaml --set n_jobs=4
    python scripts/robustness.py --config configs/full.yaml --set reservoir.backend=torch
    python scripts/robustness.py --config configs/small.yaml --variants standard leak_0.2

Same reservoirs, seeds, gains and latching tests as gain_sweep.py (the "standard" variant reproduces it).
Output in results/<name>/robustness/: summary.md (start here), runs.csv (every run), grid.csv (mean per
variant, wiring and gain), best.csv (each wiring's best valid gain per variant), robustness.png.
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
from flyres.robustness import (VARIANTS, fly_gap, robustness_markdown, run_robustness,  # noqa: E402
                               summarize_robustness)
from flyres.sweep import DEFAULT_GAINS  # noqa: E402

warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given to the executor")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--gains", nargs="+", type=float, default=list(DEFAULT_GAINS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="seeds (default 0 1 2, like the gain sweep)")
    p.add_argument("--replot", action="store_true", help="redraw summary and figure from runs.csv")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    cfg.seeds = args.seeds
    out = Path(cfg.output_dir) / cfg.name / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    if args.replot:
        df = pd.read_csv(out / "runs.csv")
    else:
        df = run_robustness(cfg, args.variants, args.gains, verbose=not args.quiet)
        df.to_csv(out / "runs.csv", index=False)
        save_config(cfg, out / "config_used.yaml")
    grid, best = summarize_robustness(df)
    grid.to_csv(out / "grid.csv", index=False)
    best.to_csv(out / "best.csv", index=False)
    (out / "summary.md").write_text(robustness_markdown(cfg, best), encoding="utf-8")
    labels = {v: label for v, (label, _) in VARIANTS.items()}
    plt.close(plotting.plot_robustness(best, labels, noise=cfg.memory.readout_noise, path=out / "robustness.png"))
    metric = "memory_noisy" if "memory_noisy" in best.columns else "memory"
    print(best.pivot(index="variant", columns="wiring", values=metric).round(1).to_string())
    print(fly_gap(best, metric).round(2).to_string(index=False))
    print(f"open {out / 'summary.md'}")


if __name__ == "__main__":
    main()
