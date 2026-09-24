"""Collect the headline numbers from finished runs into one short markdown report.

    python scripts/report.py                       # results/small and results/full
    python scripts/report.py results/small results/rho05

For each run folder: market metrics per model, memory per wiring (noise-free and with readout noise),
the dynamics diagnostics, and, if gain_sweep.py / region_gains.py were run for it, each wiring's best
valid gain and what per-region gains add.
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
