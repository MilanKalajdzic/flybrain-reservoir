"""Give every brain region its own gain and search for the best combination (memory capacity only).

    python scripts/region_gains.py --config configs/small.yaml --set n_jobs=4
    python scripts/region_gains.py --config configs/full.yaml --set reservoir.backend=torch

Each wiring starts from the two best peaks of its single-gain curve; a coordinate search then moves one
region's factor at a time (x2, then x1.41) while memory with readout noise improves and the reservoir
stays valid (4 latching tests). Reported numbers come from fresh white-noise inputs the search never saw.
Output in results/<name>/region_gains/: summary.md (start here), results.csv (one row per wiring and
seed), regions.csv (chosen factor per region), evaluations.csv (every step of the search),
region_gains.png. Redraw summary and figure from those CSVs without searching again: add --replot.
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
from flyres.regions import region_markdown, run_region_search  # noqa: E402
from flyres.sweep import DEFAULT_GAINS  # noqa: E402

# joblib restarts a worker whose memory grew a lot between jobs; harmless, results aren't lost.
warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given to the executor")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--gains", nargs="+", type=float, default=list(DEFAULT_GAINS),
                   help="single gains tried before the per-region search (default: the gain sweep's)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="seeds (default 0 1 2)")
    p.add_argument("--min-region", type=int, default=30, help="regions smaller than this fold into 'other'")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--replot", action="store_true",
                   help="don't search; redraw summary.md and the figure from the CSVs of an earlier run")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    cfg.seeds = args.seeds
    out = Path(cfg.output_dir) / cfg.name / "region_gains"
    out.mkdir(parents=True, exist_ok=True)

    if args.replot:
        results, regions = pd.read_csv(out / "results.csv"), pd.read_csv(out / "regions.csv")
        cfg = load_config(out / "config_used.yaml")
    else:
        results, regions, trace = run_region_search(cfg, args.gains, args.min_region, verbose=not args.quiet)
        results.to_csv(out / "results.csv", index=False)
        regions.to_csv(out / "regions.csv", index=False)
        trace.to_csv(out / "evaluations.csv", index=False)
        save_config(cfg, out / "config_used.yaml")
    (out / "summary.md").write_text(region_markdown(cfg, results, regions), encoding="utf-8")
    plt.close(plotting.plot_region_gains(results, regions, noise=cfg.memory.readout_noise,
                                         path=out / "region_gains.png"))
    col = "memory_noisy" if "memory_noisy_single" in results.columns else "memory"
    table = results.groupby("wiring", sort=False)[[f"{col}_single", f"{col}_regions"]].mean().round(1)
    print(table.to_string())
    print(f"open {out / 'summary.md'}")


if __name__ == "__main__":
    main()
