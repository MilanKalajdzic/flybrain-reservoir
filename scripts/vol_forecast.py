"""Forecast realized volatility with the same reservoirs as the returns experiment.

    python scripts/vol_forecast.py --config configs/small.yaml
    python scripts/vol_forecast.py --config configs/full.yaml --set reservoir.backend=torch
    python scripts/vol_forecast.py --config configs/full.yaml --set reservoir.backend=torch --best-gains

Same reservoirs (wiring, inputs, input weights, readout neurons) as run_experiment.py; a second readout
predicts log realized variance over the next 5 and 22 days (`volatility.horizons`). Benchmarks: HAR,
HAR fitted on variance, EWMA, and HAR + the reservoir's inputs (linear). If run_experiment.py has
already been run for this config, each reservoir's memory capacity is read from its graph_stats.csv
to check whether more memory means better forecasts. With --best-gains, every wiring runs at its best
valid gain from gain_sweep.py instead (memory then comes from the sweep, for the seeds it ran).
Output in results/<name>/volatility/ (or volatility_best_gain/): summary.md (start here), metrics.csv,
comparisons.csv, predictions.parquet, figures/.
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
from flyres.sweep import best_valid  # noqa: E402
from flyres.volatility import run_vol_experiment, vol_markdown  # noqa: E402

warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given to the executor")
WINDOWS = {"2008 crash": ("2008-08-01", "2009-06-30"), "COVID": ("2020-01-15", "2020-07-31")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--best-gains", action="store_true",
                   help="run every wiring at its best valid gain from gain_sweep.py (needs its results)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    run_dir = Path(cfg.output_dir) / cfg.name
    gains = None
    if args.best_gains:
        sweep_dir = run_dir / "gain_sweep"
        if not (sweep_dir / "grid.csv").exists():
            raise SystemExit(f"no {sweep_dir / 'grid.csv'}: run gain_sweep.py with this config first")
        best = best_valid(pd.read_csv(sweep_dir / "grid.csv")).dropna(subset=["best_gain"])
        gains = {w: float(g) for w, g in zip(best["wiring"], best["best_gain"]) if w in cfg.wirings}
        sweep = pd.read_csv(sweep_dir / "sweep.csv")
        graph = pd.concat([sweep[(sweep["wiring"] == w) & (sweep["gain"] == g)] for w, g in gains.items()])
        out = run_dir / "volatility_best_gain"
    else:
        graph_path = run_dir / "graph_stats.csv"
        graph = pd.read_csv(graph_path) if graph_path.exists() else None
        if graph is None and not args.quiet:
            print(f"no {graph_path} (run run_experiment.py first to relate memory to forecasting skill)")
        out = run_dir / "volatility"
    (out / "figures").mkdir(parents=True, exist_ok=True)

    res = run_vol_experiment(cfg, graph, verbose=not args.quiet, gains=gains)
    res.metrics.to_csv(out / "metrics.csv", index=False)
    res.comparisons.to_csv(out / "comparisons.csv", index=False)
    res.predictions.to_parquet(out / "predictions.parquet", index=False)
    if res.link is not None:
        res.link.to_csv(out / "memory_vs_skill.csv", index=False)
    save_config(cfg, out / "config_used.yaml")
    (out / "summary.md").write_text(vol_markdown(cfg, res), encoding="utf-8")
    for ticker in res.metrics["ticker"].unique():
        plt.close(plotting.plot_vol_models(res.metrics, ticker, path=out / "figures" / f"vol_models_{ticker}.png"))
        if res.link is not None:
            plt.close(plotting.plot_vol_memory(res.link, ticker, path=out / "figures" / f"vol_memory_{ticker}.png"))
        h = min(cfg.volatility.horizons)
        plt.close(plotting.plot_vol_forecast(res.predictions, ticker, h, WINDOWS,
                                             path=out / "figures" / f"vol_forecast_{ticker}.png"))
    m = res.metrics
    har = m[m["model"] == "har"].set_index(["ticker", "horizon"])["mse_log"]
    m = m.assign(vs_har=100 * (m["mse_log"] / har.reindex(pd.MultiIndex.from_frame(m[["ticker", "horizon"]])).to_numpy()
                               - 1))
    print(m.groupby(["ticker", "horizon", "model"], sort=False)[["r2_log", "vs_har", "qlike"]].mean().round(3)
          .to_string())
    print(f"open {out / 'summary.md'}")


if __name__ == "__main__":
    main()
