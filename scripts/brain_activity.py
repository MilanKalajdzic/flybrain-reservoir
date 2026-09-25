"""Make the "brain lighting up" figures: a heatmap of every pathway stage over the whole sample and
animated brains for crash windows (2008 and COVID by default).

    python scripts/brain_activity.py --config configs/small.yaml
    python scripts/brain_activity.py --config configs/full.yaml --set reservoir.backend=torch
    python scripts/brain_activity.py --config configs/full.yaml --set reservoir.backend=torch --compare degree_preserving
    python scripts/brain_activity.py --config configs/small.yaml --window 2022-01-01 2022-10-31 --name 2022

Uses the experiment's own reservoir (same wiring seed, input weights, bias). Only neurons that move by at
least 0.1% of their maximum glow; the rest stay dark, since z-scoring would blow fluctuations of a millionth
up to full size. The whole CNS is simulated once per block of neurons (--chunk), so memory stays small.
The animation needs the raw annotation file (for soma positions); run download_data.py first.
--compare adds a second wiring and animates the two brains side by side (brain_<window>_vs_<other>.gif).
Output: results/<config name>/brain/ (or --out). Files for another wiring get its name as a suffix.
The simulated activity is saved there too (activity*.npz): add --replot to redraw the last run's figures
without simulating.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from flyres import activity, plotting  # noqa: E402
from flyres.config import load_config  # noqa: E402
from flyres.connectome import SOURCES  # noqa: E402
from flyres.controls import WIRINGS  # noqa: E402
from flyres.experiment import build_matrix, load_closes, market_reservoir, prepare_subgraph  # noqa: E402
from flyres.market import make_dataset  # noqa: E402

WINDOWS = {"2008": ("2008-06-01", "2009-06-30"), "COVID": ("2020-01-01", "2020-08-31")}
WIRING_TEXT = {  # (title start, how the neurons are wired, short name, how, for a side-by-side panel)
    "connectome": ("A fly brain", "wired as in the male CNS connectome", "The fly's wiring",
                   "as in the male CNS connectome"),
    "degree_preserving": ("The same neurons, rewired at random,", "rewired, each keeping its number of partners",
                          "Rewired at random", "each neuron keeps its number of partners"),
    "weight_shuffle": ("The fly's wiring with shuffled synapse counts,", "real connections, synapse counts shuffled",
                       "Synapse counts shuffled", "the fly's connections, synapse counts shuffled"),
    "sign_shuffle": ("The fly's wiring with shuffled signs,", "real connections, excitatory/inhibitory shuffled",
                     "Signs shuffled", "the fly's connections, excitatory/inhibitory shuffled"),
    "erdos_renyi": ("The same neurons, wired completely at random,", "wired completely at random",
                    "Wired completely at random", "same number of connections, nothing else kept"),
}
DRAWN = "Only neurons whose activity varies by more than 0.1% of its maximum are drawn"
CONNECTOMES = {"malecns": "the male CNS connectome", "flywire": "the FlyWire connectome (female brain)",
               "synthetic": "a synthetic test graph"}


def save_activity(path: Path, dev, glow: dict, moving, windows: dict) -> None:
    """Everything the figures need, so they can be redrawn without simulating again (--replot)."""
    arrays = {"moving": moving, "dev": dev.to_numpy(), "dev_index": dev.index.values.astype("datetime64[ns]"),
              "dev_columns": np.array(dev.columns, dtype=str), "windows": np.array(list(glow), dtype=str)}
    for name, g in glow.items():
        arrays[f"glow_{name}"] = g.to_numpy(np.float32)
        arrays[f"weeks_{name}"] = g.index.values.astype("datetime64[ns]")
        arrays[f"bounds_{name}"] = np.array(windows[name], dtype=str)
    np.savez_compressed(path, **arrays)


def load_activity(path: Path):
    """The saved (dev, glow, moving, windows); the windows are the ones that run simulated."""
    if not path.exists():
        raise SystemExit(f"no {path}: run once without --replot first")
    z = np.load(path)
    dev = pd.DataFrame(z["dev"], index=pd.DatetimeIndex(z["dev_index"]), columns=list(z["dev_columns"]))
    names = [str(n) for n in z["windows"]]
    glow = {n: pd.DataFrame(z[f"glow_{n}"], index=pd.DatetimeIndex(z[f"weeks_{n}"])) for n in names}
    windows = {n: tuple(str(b) for b in z[f"bounds_{n}"]) for n in names}
    return dev, glow, z["moving"], windows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--wiring", default="connectome", choices=WIRINGS)
    p.add_argument("--compare", default=None, choices=WIRINGS,
                   help="a second wiring: also simulate it and animate both brains side by side")
    p.add_argument("--ticker", default=None, help="which ticker to use (default: the first one)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", nargs=2, default=None, metavar=("START", "END"),
                   help="one custom window instead of the default 2008 and COVID ones (use with --name)")
    p.add_argument("--name", default="custom", help="label for a custom --window")
    p.add_argument("--chunk", type=int, default=20_000, help="neurons recorded per simulation run")
    p.add_argument("--out", default=None, help="output folder (default results/<name>/brain)")
    p.add_argument("--replot", action="store_true",
                   help="don't simulate; redraw from the activity saved by an earlier run (activity*.npz)")
    args = p.parse_args()
    if args.compare == args.wiring:
        p.error("--compare needs a different wiring than --wiring")

    cfg = load_config(args.config, args.set)
    connectome = CONNECTOMES.get(cfg.connectome.source, "the connectome")
    text = {w: tuple(t.replace("the male CNS connectome", connectome) for t in v) for w, v in WIRING_TEXT.items()}
    out = Path(args.out or Path(cfg.output_dir) / cfg.name / "brain")
    out.mkdir(parents=True, exist_ok=True)
    windows = {args.name: tuple(args.window)} if args.window else WINDOWS

    # verbose: the first run builds the connectome cache (a few minutes, once) and prints its progress
    sub = prepare_subgraph(cfg, verbose=True)
    closes = load_closes(cfg)
    ticker = args.ticker or next(iter(closes))
    close = closes[ticker]
    mc = cfg.market
    ds = make_dataset(close, mc.features, mc.horizon, mc.z_window, mc.ar_lags, ticker=ticker)
    dates = ds.dates[cfg.reservoir.washout:]
    groups = activity.neuron_groups(sub.neurons, sub.input_idx)

    def suffix(wiring):
        return "" if wiring == "connectome" else f"_{wiring}"

    runs = {}  # wiring -> (dev, glow, moving, windows)
    for wiring in [args.wiring] + ([args.compare] if args.compare else []):
        saved = out / f"activity{suffix(wiring)}.npz"
        if args.replot:  # redraw whatever that run simulated; --window is ignored here
            runs[wiring] = load_activity(saved)
        else:
            W, *_ = build_matrix(sub, cfg, wiring, args.seed)
            res = market_reservoir(sub, cfg, W, ds.X.shape[1], args.seed)
            n_runs = -(-sub.n // args.chunk)
            print(f"simulating {sub.n:,} neurons ({wiring}) over {len(ds):,} days"
                  + (f", {n_runs} runs of {args.chunk:,} recorded neurons" if n_runs > 1 else ""), flush=True)
            dev, glow, moving = activity.stream_activity(res, ds.X, cfg.reservoir.washout, dates, groups, windows,
                                                         chunk=args.chunk)
            save_activity(saved, dev, glow, moving, windows)
            runs[wiring] = (dev, glow, moving, windows)
        dev, glow, moving, _ = runs[wiring]
        print(f"{wiring}: {moving.sum():,} of {sub.n:,} neurons ({moving.mean():.0%}) move by at least 0.1% "
              "of their maximum")
        dev.to_csv(out / f"group_deviation_monthly{suffix(wiring)}.csv")
        heat = out / f"activity_heatmap{suffix(wiring)}.png"
        title = ("How stirred up each part of the fly circuit is" if wiring == "connectome"
                 else f"How stirred up each part of the circuit is ({text[wiring][2].lower()})")
        plt.close(plotting.plot_activity_heatmap(dev, close, activity.EPISODES, ticker=ticker, path=heat,
                                                 share=activity.moving_share(groups, moving), title=title))
        print(f"wrote {heat}")

    raw, source = Path(cfg.data.raw_dir), cfg.connectome.source
    if source not in SOURCES or not activity.annotation_file(raw, source).exists():
        print("skipping the animation: needs the real annotation file (scripts/download_data.py)")
        return
    xyz = activity.soma_positions(raw, sub.neurons["bodyId"], source)
    background = activity.all_soma_positions(raw, source)

    def crash(name, window):
        return f"the {name} crash" if name in WINDOWS and tuple(window) == WINDOWS[name] else name

    for wiring, (_, glow, moving, wins) in runs.items():
        start, what, *_ = text[wiring]
        subtitle = (f"{sub.n:,} neurons {what}, driven by {ticker} returns and volatility. "
                    f"Brighter = further from normal.\n{DRAWN}: {moving.mean():.0%} of them.")
        for name, window in wins.items():
            if glow[name].empty:
                print(f"skipping {name}: no data in {window[0]} to {window[1]}")
                continue
            gif = plotting.animate_brain(glow[name], xyz[:, :2], sub.input_idx, close, background[:, :2], window,
                                         f"{start} watching {crash(name, window)}",
                                         out / f"brain_{name}{suffix(wiring)}.gif", ticker=ticker, moving=moving,
                                         subtitle=subtitle)
            print(f"wrote {gif}")

    if args.compare:
        (_, glow_a, moving_a, wins_a), (_, glow_b, moving_b, wins_b) = runs[args.wiring], runs[args.compare]
        labels = [(text[w][2], f"{text[w][3][0].upper() + text[w][3][1:]}. "
                      f"{m.sum():,} neurons move ({m.mean():.0%}).")
                  for w, m in ((args.wiring, moving_a), (args.compare, moving_b))]
        subtitle = (f"Driven by {ticker} returns and volatility. Brighter = further from normal.\n"
                    f"{DRAWN}.")
        for name, window in wins_a.items():
            if name not in wins_b or glow_a[name].empty or glow_b[name].empty:
                continue
            gif = plotting.animate_brain_pair(
                [glow_a[name], glow_b[name]], xyz[:, :2], sub.input_idx, close, background[:, :2], window,
                f"The same {sub.n:,} neurons, two wirings, watching {crash(name, window)}",
                out / f"brain_{name}{suffix(args.wiring)}_vs_{args.compare}.gif", labels, [moving_a, moving_b],
                subtitle=subtitle, ticker=ticker)
            print(f"wrote {gif}")


if __name__ == "__main__":
    main()
