"""Make the "brain lighting up" figures: a heatmap of every pathway stage over the whole sample and
an animated brain for a crash window.

    python scripts/brain_activity.py --config configs/small.yaml
    python scripts/brain_activity.py --config configs/small.yaml --window 2020-01-01 2020-08-31 --name covid

Uses the connectome wiring with the experiment's own seed, so it's the same reservoir the forecasts
used. The animation needs the raw annotation file (for soma positions); run download_data.py first.
Output: results/<config name>/brain/ (or --out).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from flyres import activity, plotting  # noqa: E402
from flyres.config import load_config  # noqa: E402
from flyres.connectome import FILES  # noqa: E402
from flyres.experiment import build_matrix, load_closes, market_reservoir, prepare_subgraph  # noqa: E402
from flyres.market import make_dataset  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--ticker", default=None, help="which ticker to use (default: the first one)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", nargs=2, default=["2008-06-01", "2009-06-30"], metavar=("START", "END"))
    p.add_argument("--name", default="2008", help="label for the animation file and title")
    p.add_argument("--title", default=None)
    p.add_argument("--out", default=None, help="output folder (default results/<name>/brain)")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    out = Path(args.out or Path(cfg.output_dir) / cfg.name / "brain")
    out.mkdir(parents=True, exist_ok=True)

    sub = prepare_subgraph(cfg, verbose=False)
    closes = load_closes(cfg)
    ticker = args.ticker or next(iter(closes))
    close = closes[ticker]
    mc = cfg.market
    ds = make_dataset(close, mc.features, mc.horizon, mc.z_window, mc.ar_lags, ticker=ticker)
    W, *_ = build_matrix(sub, cfg, "connectome", args.seed)
    states = market_reservoir(sub, cfg, W, ds.X.shape[1], args.seed).run(ds.X, washout=cfg.reservoir.washout)
    dates = ds.dates[cfg.reservoir.washout:]
    z = activity.zscore(states)
    groups = activity.neuron_groups(sub.neurons, sub.input_idx)

    dev = activity.group_deviation(z, groups, dates, freq="MS")
    dev.to_csv(out / "group_deviation_monthly.csv")
    plt.close(plotting.plot_activity_heatmap(dev, close, activity.EPISODES, ticker=ticker,
                                             path=out / "activity_heatmap.png"))
    print(f"wrote {out / 'activity_heatmap.png'}")

    raw = Path(cfg.data.raw_dir)
    if cfg.connectome.source != "malecns" or not (raw / FILES["annotations"]).exists():
        print("skipping the animation: needs the real annotation file (scripts/download_data.py)")
        return
    xyz = activity.soma_positions(raw, sub.neurons["bodyId"])
    background = activity.all_soma_positions(raw)
    glow = activity.weekly(np.abs(z), dates)
    title = args.title or f"A fly brain watching the {args.name} crash"
    gif = plotting.animate_brain(glow, xyz[:, :2], sub.input_idx, close, background[:, :2], tuple(args.window),
                                 title, out / f"brain_{args.name}.gif", ticker=ticker)
    print(f"wrote {gif}")


if __name__ == "__main__":
    main()
