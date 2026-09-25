"""Collect the headline numbers from finished runs into one short markdown report.

    python scripts/report.py                       # results/small and results/full
    python scripts/report.py results/small results/rho05

For each run folder: market metrics per model, memory per wiring (noise-free and with readout noise),
the dynamics diagnostics, and, for whichever of gain_sweep.py, region_gains.py, narma.py, homeostasis.py,
robustness.py and vol_forecast.py were run for it, their headline tables.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from flyres.controls import WIRINGS
from flyres.sweep import best_valid


def pm(s: pd.Series, digits: int = 1) -> str:
    s = s.dropna()
    if s.empty:
        return "-"
    return f"{s.mean():.{digits}f}" + (f" ± {s.std():.{digits}f}" if len(s) > 1 else "")


def order(names):
    rank = {w: i for i, w in enumerate(WIRINGS)}
    return sorted(names, key=lambda w: rank.get(w, len(rank)))


def run_report(folder: Path) -> list[str]:
    lines = [f"## {folder.name}", ""]
    metrics, graph = folder / "metrics.csv", folder / "graph_stats.csv"
    if metrics.exists():
        m = pd.read_csv(metrics)
        lines += ["| model | IC | hit rate | Sharpe (net) |", "|---|---|---|---|"]
        for model in order(m["model"].unique()):
            g = m[m["model"] == model]
            lines.append(f"| {model} | {pm(g['ic'], 3)} | {pm(g['hit_rate'], 3)} | {pm(g['sharpe_net'], 2)} |")
        lines.append("")
    if graph.exists():
        g = pd.read_csv(graph)
        cols = [("memory_capacity", "memory (noise-free)", 1), ("memory_capacity_noisy", "memory (readout noise)", 1),
                ("raw_gain", "raw spectral radius", 1), ("top_mode_spread", "top mode spread", 0),
                ("active_readouts", "active readouts", 2), ("unstable_readouts", "unstable readouts", 2)]
        cols = [c for c in cols if c[0] in g.columns]
        lines += ["| wiring | " + " | ".join(c[1] for c in cols) + " |", "|---|" + "---|" * len(cols)]
        for w in order(g["wiring"].unique()):
            s = g[g["wiring"] == w]
            lines.append(f"| {w} | " + " | ".join(pm(s[c], d) for c, _, d in cols) + " |")
        lines += ["", f"({s['n'].iloc[0]:,} neurons, {g['seed'].nunique()} seeds)", ""]
    grid = folder / "gain_sweep" / "grid.csv"
    if grid.exists():
        summary = pd.read_csv(grid)
        best = best_valid(summary)
        noisy = "memory_noisy" in best.columns
        lines += ["Gain sweep, best valid gain per wiring" + (" (picked on memory with readout noise)" if noisy else ""),
                  "", "| wiring | best gain | " + ("memory (readout noise) | " if noisy else "")
                  + "memory (noise-free) | active | valid gains |", "|---|---|" + ("---|" if noisy else "") + "---|---|---|"]
        for _, r in best.iterrows():
            def f(m, s):
                return "-" if not np.isfinite(m) else f"{m:.1f}" + (f" ± {s:.1f}" if np.isfinite(s) else "")
            lines.append(f"| {r['wiring']} | {r['best_gain']:g} | "
                         + (f"{f(r['memory_noisy'], r['memory_noisy_sd'])} | " if noisy else "")
                         + f"{f(r['memory'], r['memory_sd'])} | "
                         + ("-" if not np.isfinite(r["active"]) else f"{r['active']:.0%}")
                         + f" | {r['valid_gains']}/{r['gains_tried']} |")
        at = summary[summary["gain"].isin([0.9, 1.0])]
        if len(at) and noisy:
            lines += ["", "At the standard gains (memory with readout noise / noise-free):", ""]
            for w in order(at["wiring"].unique()):
                parts = [f"{r['gain']:g}: {r['memory_noisy']:.1f} / {r['memory']:.1f}"
                         for _, r in at[at["wiring"] == w].sort_values("gain").iterrows()]
                lines.append(f"- {w}: " + ", ".join(parts))
        lines.append("")
    regions = folder / "region_gains" / "results.csv"
    if regions.exists():
        r = pd.read_csv(regions)
        col = "memory_noisy" if "memory_noisy_single" in r.columns else "memory"
        lines += ["Per-region gains, memory " + ("with readout noise " if col == "memory_noisy" else "")
                  + "on a fresh input", "",
                  "| wiring | best single gain | per-region gains | change | active readouts | valid on fresh input |",
                  "|---|---|---|---|---|---|"]
        for w in order(r["wiring"].unique()):
            s = r[r["wiring"] == w]
            d = (s[f"{col}_regions"] - s[f"{col}_single"]).mean()
            lines.append(f"| {w} | {pm(s[f'{col}_single'])} | {pm(s[f'{col}_regions'])} | {d:+.1f} | "
                         f"{s['active_readouts_single'].mean():.0%} → {s['active_readouts_regions'].mean():.0%} | "
                         f"{int(s['valid_single'].sum())}/{len(s)} → {int(s['valid_regions'].sum())}/{len(s)} |")
        lines.append("")
    narma = folder / "narma" / "grid.csv"
    if narma.exists():
        from flyres.narma import best_narma

        grid = pd.read_csv(narma)
        best = best_narma(grid, 0.9)
        noisy = "nrmse_noisy" in grid.columns
        base = folder / "narma" / "linear_baseline.csv"
        lines += ["NARMA-10 error (NRMSE, lower is better" + (", with readout noise" if noisy else "") + ")"
                  + (f"; linear baseline {pd.read_csv(base)['nrmse'].mean():.3f}" if base.exists() else ""), "",
                  "| wiring | at gain 0.9 | best valid gain | NRMSE there | noise-free there | valid gains |",
                  "|---|---|---|---|---|---|"]
        for _, r in best.iterrows():
            main = r["nrmse_noisy"] if noisy else r["nrmse"]
            there = (f"{r['best_gain']:g} | {main:.3f} | {r['nrmse']:.3f}" if np.isfinite(main) else "none | - | -")
            lines.append(f"| {r['wiring']} | {r['standard']:.3f} | {there} | {r['valid_gains']}/{r['gains_tried']} |")
        lines.append("")
    homeo = folder / "homeostasis" / "results.csv"
    if homeo.exists():
        r = pd.read_csv(homeo)
        col = "memory_noisy" if "memory_noisy_single" in r.columns else "memory"
        lines += ["Homeostatic gains, memory " + ("with readout noise " if col == "memory_noisy" else "")
                  + "on a fresh input", "",
                  "| wiring | best single gain | homeostatic | change | targets | valid on fresh input |",
                  "|---|---|---|---|---|---|"]
        for w in order(r["wiring"].unique()):
            s = r[r["wiring"] == w]
            d = (s[f"{col}_homeostatic"] - s[f"{col}_single"]).mean()
            lines.append(f"| {w} | {pm(s[f'{col}_single'])} | {pm(s[f'{col}_homeostatic'])} | {d:+.1f} | "
                         f"{', '.join(f'{t:g}' for t in s['target'])} | "
                         f"{int(s['valid_single'].sum())}/{len(s)} → {int(s['valid_homeostatic'].sum())}/{len(s)} |")
        lines.append("")
    rob = folder / "robustness" / "best.csv"
    if rob.exists():
        from flyres.robustness import VARIANTS, fly_gap

        b = pd.read_csv(rob)
        metric = "memory_noisy" if "memory_noisy" in b.columns else "memory"
        wirings = order(b["wiring"].unique())
        lines += ["Robustness, memory " + ("with readout noise " if metric == "memory_noisy" else "")
                  + "at each wiring's best valid gain", "",
                  "| variant | " + " | ".join(wirings) + " | best control / fly |",
                  "|---|" + "---|" * (len(wirings) + 1)]
        gap = fly_gap(b, metric).set_index("variant")
        for v in [v for v in VARIANTS if v in set(b["variant"])]:
            s = b[b["variant"] == v].set_index("wiring")
            cells = ["-" if w not in s.index or not np.isfinite(s.loc[w, metric])
                     else f"{s.loc[w, metric]:.1f} ± {s.loc[w, metric + '_sd']:.1f}" for w in wirings]
            lines.append(f"| {VARIANTS[v][0]} | " + " | ".join(cells) + f" | {gap.loc[v, 'ratio']:.1f}× |")
        lines.append("")
    for sub, label in (("volatility", "standard gain"), ("volatility_best_gain", "each wiring at its best valid gain")):
        vm = folder / sub / "metrics.csv"
        if not vm.exists():
            continue
        v = pd.read_csv(vm)
        horizons = sorted(v["horizon"].unique())
        har = v[v["model"] == "har"].set_index(["ticker", "horizon"])["mse_log"]
        for ticker in v["ticker"].unique():
            lines += [f"Volatility forecasts, {ticker}, {label}: log MSE vs HAR (R² log)", "",
                      "| model | " + " | ".join(f"next {h} days" for h in horizons) + " |",
                      "|---|" + "---|" * len(horizons)]
            names = order(v["model"].unique())
            for model in [n for n in names if n in WIRINGS] + [n for n in ("har_inputs", "har", "har_levels", "ewma")
                                                              if n in names]:
                cells = []
                for h in horizons:
                    s = v[(v["model"] == model) & (v["ticker"] == ticker) & (v["horizon"] == h)]
                    cells.append(f"{100 * (s['mse_log'].mean() / har[(ticker, h)] - 1):+.1f}% "
                                 f"({s['r2_log'].mean():.3f})")
                lines.append(f"| {model} | " + " | ".join(cells) + " |")
            lines.append("")
    if len(lines) == 2:
        lines += ["(no results found)", ""]
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="*", default=["results/small", "results/full"])
    args = p.parse_args()
    out = ["# flybrain-reservoir report", ""]
    for folder in args.folders:
        out += run_report(Path(folder))
    print("\n".join(out))


if __name__ == "__main__":
    main()
