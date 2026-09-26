"""Compare every wiring at its own best gain (memory capacity only).

    python scripts/gain_sweep.py --config configs/full.yaml --set reservoir.backend=torch
    python scripts/gain_sweep.py --config configs/small.yaml --gains 0.5 0.9 1.0 1.25 1.5 --seeds 0 1

Uses the config's subgraph, wirings, reservoir and memory settings, with the seeds from --seeds (default 0 1 2,
whatever the config says); only the gain changes.
Output in results/<name>/gain_sweep/: summary.md (start here), sweep.csv (every run),
grid.csv (mean per wiring and gain), gain_sweep.png. --replot redraws them from sweep.csv, with the settings
the sweep ran with (config_used.yaml).
"""
import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from flyres import plotting  # noqa: E402
from flyres.config import load_config, run_label, save_config  # noqa: E402
from flyres.sweep import DEFAULT_GAINS, best_valid, run_gain_sweep, summarize_sweep, sweep_markdown  # noqa: E402

# joblib restarts a worker whose memory grew a lot between jobs; harmless, results aren't lost.
warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given to the executor")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--gains", nargs="+", type=float, default=list(DEFAULT_GAINS),
                   help=f"targets for the spectral radius (default: {' '.join(f'{g:g}' for g in DEFAULT_GAINS)})")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                   help="seeds (default 0 1 2; fewer than the main experiment because it multiplies by the gains)")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--replot", action="store_true", help="redraw summary and figure from sweep.csv")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    cfg.seeds = args.seeds
    out = Path(cfg.output_dir) / cfg.name / "gain_sweep"
    out.mkdir(parents=True, exist_ok=True)

    if args.replot:
        df = pd.read_csv(out / "sweep.csv")
        if (out / "config_used.yaml").exists():  # the settings the sweep ran with, not today's command line
            cfg = load_config(out / "config_used.yaml")
    else:
        df = run_gain_sweep(cfg, args.gains, verbose=not args.quiet)
        df.to_csv(out / "sweep.csv", index=False)
        save_config(cfg, out / "config_used.yaml")
    summary = summarize_sweep(df)
    best = best_valid(summary)
    summary.to_csv(out / "grid.csv", index=False)
    (out / "summary.md").write_text(sweep_markdown(cfg, summary, best), encoding="utf-8")
    label = "spectral radius" if cfg.reservoir.normalize == "spectral" else "bulk scale (frobenius)"
    plt.close(plotting.plot_gain_sweep(summary, label, noise=cfg.memory.readout_noise, path=out / "gain_sweep.png",
                                       title=f"Memory capacity at each gain, {run_label(cfg)}"))
    print(best.to_string(index=False))
    print(f"open {out / 'summary.md'}")


if __name__ == "__main__":
    main()
