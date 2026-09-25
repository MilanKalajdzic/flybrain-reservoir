"""Per-neuron gains from a homeostatic rule: every neuron scales its inputs toward the same input level.

    python scripts/homeostasis.py --config configs/small.yaml --set n_jobs=4
    python scripts/homeostasis.py --config configs/full.yaml --set reservoir.backend=torch

For every wiring and seed: the best single gain (for comparison), then synaptic scaling toward each of a
few targets for the size (RMS) of a neuron's recurrent input, keeping the best valid target. Reported numbers come from fresh white-noise inputs.
Output in results/<name>/homeostasis/: summary.md (start here), results.csv (one row per wiring and
seed), regions.csv (mean gain per region), trials.csv (every target), homeostasis.png.
Redraw summary and figure from the CSVs without rerunning: add --replot.
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
from flyres.homeostasis import TARGETS, homeostasis_markdown, run_homeostasis  # noqa: E402
from flyres.sweep import DEFAULT_GAINS  # noqa: E402

warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given to the executor")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--targets", nargs="+", type=float, default=list(TARGETS),
                   help=f"targets for the RMS of each neuron's recurrent input (default: "
                        f"{' '.join(f'{t:g}' for t in TARGETS)})")
    p.add_argument("--gains", nargs="+", type=float, default=list(DEFAULT_GAINS),
                   help="single gains tried for the comparison (default: the gain sweep's)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="seeds (default 0 1 2)")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--replot", action="store_true", help="don't run; redraw summary and figure from earlier CSVs")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    cfg.seeds = args.seeds
    out = Path(cfg.output_dir) / cfg.name / "homeostasis"
    out.mkdir(parents=True, exist_ok=True)
    if args.replot:
        results, regions, trials = (pd.read_csv(out / f) for f in ("results.csv", "regions.csv", "trials.csv"))
        cfg = load_config(out / "config_used.yaml")
    else:
        results, regions, trials = run_homeostasis(cfg, args.gains, args.targets, verbose=not args.quiet)
        results.to_csv(out / "results.csv", index=False)
        regions.to_csv(out / "regions.csv", index=False)
        trials.to_csv(out / "trials.csv", index=False)
        save_config(cfg, out / "config_used.yaml")
    (out / "summary.md").write_text(homeostasis_markdown(cfg, results, regions, trials), encoding="utf-8")
    plt.close(plotting.plot_region_gains(
        results, regions, noise=cfg.memory.readout_noise, path=out / "homeostasis.png", after="homeostatic",
        after_label="homeostatic gains", title=f"Homeostatic gains, {run_label(cfg)}",
        heat_title="Mean gain of each region's neurons after the rule (neurons)",
        subtitle="Hollow = each wiring's best single gain. Filled = after every neuron scales its inputs toward the "
                 "same input level; red = region turned up, blue = down. Mean over seeds, small dots = seeds."))
    col = "memory_noisy" if "memory_noisy_single" in results.columns else "memory"
    print(results.groupby("wiring", sort=False)[[f"{col}_single", f"{col}_homeostatic", "target"]].mean()
          .round(3).to_string())
    print(f"open {out / 'summary.md'}")


if __name__ == "__main__":
    main()
