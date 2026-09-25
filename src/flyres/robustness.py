"""Robustness checks for the memory result: does the fly still lose when the setup changes?

Two obvious objections to the main result, each rerun as a full gain sweep (every wiring at every
gain, validity by the latching tests, each wiring judged at its best valid gain, exactly as in sweep.py):

- Where the input enters. The inputs are the sensory neurons with the most synapses, and the fly's
  biggest knot sits right behind the smell sensors, in the antennal lobe. "random_inputs" feeds the
  input into the same number of neurons picked at random among all neurons that have outgoing synapses
  (chosen per seed, the same for every wiring of that seed).
- Neuron speed. The standard reservoir has leak rate 1: each neuron's state is replaced every step.
  "leak_0.5" and "leak_0.2" make neurons leaky integrators, x <- (1 - a) x + a tanh(...): inputs linger
  for roughly 1/a steps, but each step's input moves the state less. Real neurons are slow like this.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import ExperimentConfig
from .controls import WIRINGS
from .experiment import prepare_subgraph
from .subgraph import Subgraph
from .sweep import DEFAULT_GAINS, best_valid, summarize_sweep, sweep_job

VARIANTS = {  # name: (label, change)
    "standard": ("standard", {}),
    "random_inputs": ("random input neurons", {"inputs": "random"}),
    "leak_0.5": ("leak rate 0.5", {"leak_rate": 0.5}),
    "leak_0.2": ("leak rate 0.2", {"leak_rate": 0.2}),
}


def random_inputs(sub: Subgraph, seed: int) -> Subgraph:
    """The same circuit with the input fed into randomly chosen neurons (as many as before) instead.
    Only neurons with outgoing synapses qualify, so the input can spread."""
    sends = np.flatnonzero(np.asarray(sub.W.sum(axis=0)).ravel() > 0)  # W[post, pre]: column sums = output
    idx = np.random.default_rng([seed, 13]).choice(sends, min(len(sub.input_idx), len(sends)), replace=False)
    return Subgraph(sub.W, sub.sign, sub.neurons, np.sort(idx))


def variant_setup(sub: Subgraph, cfg: ExperimentConfig, variant: str, seed: int):
    """(subgraph, config) for one variant and seed."""
    change = VARIANTS[variant][1]
    cfg = copy.deepcopy(cfg)
    if "leak_rate" in change:
        cfg.reservoir.leak_rate = change["leak_rate"]
    if change.get("inputs") == "random":
        sub = random_inputs(sub, seed)
    return sub, cfg


def robustness_job(sub: Subgraph, cfg: ExperimentConfig, variant: str, wiring: str, seed: int, gains) -> list[dict]:
    t0 = time.time()
    sub_v, cfg_v = variant_setup(sub, cfg, variant, seed)
    rows = [{"variant": variant, **r} for r in sweep_job(sub_v, cfg_v, wiring, seed, gains)]
    rows[-1]["seconds"] = time.time() - t0
    return rows


def run_robustness(cfg: ExperimentConfig, variants=tuple(VARIANTS), gains=None, verbose: bool = True) -> pd.DataFrame:
    """Every variant x wiring x seed x gain: memory (noise-free and with readout noise) and stability."""
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choose from {list(VARIANTS)}")
    gains = sorted(float(g) for g in (gains or DEFAULT_GAINS))
    sub = prepare_subgraph(cfg, verbose)
    jobs = [(v, w, s) for v in variants for w in cfg.wirings for s in cfg.seeds]
    if verbose:
        print(f"robustness: {len(jobs)} jobs ({len(variants)} variants x {len(cfg.wirings)} wirings x "
              f"{len(cfg.seeds)} seeds), {len(gains)} gains each", flush=True)
    if cfg.n_jobs == 1:
        out = []
        for i, (v, w, s) in enumerate(jobs, 1):
            out.append(robustness_job(sub, cfg, v, w, s, gains))
            if verbose:
                print(f"  [{i}/{len(jobs)}] {v:<14} {w:<18} seed {s}: {out[-1][-1]['seconds']:.0f}s", flush=True)
    else:
        out = Parallel(n_jobs=cfg.n_jobs, verbose=10 if verbose else 0)(
            delayed(robustness_job)(sub, cfg, v, w, s, gains) for v, w, s in jobs)
    df = pd.DataFrame([r for rows in out for r in rows]).drop(columns="seconds", errors="ignore")
    return df.sort_values(["variant", "wiring", "gain", "seed"]).reset_index(drop=True)


def summarize_robustness(df: pd.DataFrame):
    """(grid, best): the gain-sweep summary and each wiring's best valid gain, per variant, in VARIANTS order."""
    grids, bests = [], []
    for v in [v for v in VARIANTS if v in set(df["variant"])]:
        grid = summarize_sweep(df[df["variant"] == v].drop(columns="variant"))
        grids.append(grid.assign(variant=v))
        bests.append(best_valid(grid).assign(variant=v))
    return pd.concat(grids, ignore_index=True), pd.concat(bests, ignore_index=True)


def fly_gap(best: pd.DataFrame, metric: str = "memory_noisy") -> pd.DataFrame:
    """Per variant: the fly's memory, the best control's, and how many times more the best control has."""
    rows = []
    for v, b in best.groupby("variant", sort=False):
        b = b.set_index("wiring")[metric]
        fly = b.get("connectome", np.nan)
        controls = b.drop("connectome", errors="ignore").dropna()
        top = controls.idxmax() if len(controls) else None
        rows.append({"variant": v, "connectome": fly, "best_control": top,
                     "best_control_memory": controls.max() if len(controls) else np.nan,
                     "ratio": controls.max() / fly if len(controls) and fly > 0 else np.nan,
                     "rank": int((controls > fly).sum()) + 1 if np.isfinite(fly) else np.nan})
    return pd.DataFrame(rows)


def robustness_markdown(cfg: ExperimentConfig, best: pd.DataFrame) -> str:
    noisy = "memory_noisy" in best.columns
    metric = "memory_noisy" if noisy else "memory"
    wirings = [w for w in WIRINGS if w in set(best["wiring"])]
    lines = [f"# Robustness: {cfg.name}", "",
             f"Memory capacity{' with readout noise' if noisy else ''} of each wiring at its best valid gain, "
             f"{len(cfg.seeds)} seed(s), for each variant of the setup (gain sweep as in gain_sweep.py). "
             "Cells: mean ± sd over seeds (best gain).", "",
             "| variant | " + " | ".join(wirings) + " |", "|---|" + "---|" * len(wirings)]
    for v in [v for v in VARIANTS if v in set(best["variant"])]:
        b = best[best["variant"] == v].set_index("wiring")
        cells = []
        for w in wirings:
            r = b.loc[w]
            cells.append("none valid" if not np.isfinite(r[metric]) else
                         f"{r[metric]:.1f}" + (f" ± {r[metric + '_sd']:.1f}" if np.isfinite(r[metric + "_sd"]) else "")
                         + f" ({r['best_gain']:g})")
        lines.append(f"| {VARIANTS[v][0]} | " + " | ".join(cells) + " |")
    gap = fly_gap(best, metric)
    lines += ["", "## The fly against the best control", "",
              "| variant | connectome | best control | its memory | times the fly's | fly's rank |",
              "|---|---|---|---|---|---|"]
    for _, r in gap.iterrows():
        lines.append(f"| {VARIANTS[r['variant']][0]} | {r['connectome']:.1f} | {r['best_control'] or '-'} | "
                     f"{r['best_control_memory']:.1f} | {r['ratio']:.1f}× | "
                     + ("-" if not np.isfinite(r["rank"]) else f"{int(r['rank'])} of {len(wirings)}") + " |")
    lines += ["", "Random input neurons: the input enters through as many randomly chosen neurons (with outgoing "
                  "synapses) instead of the sensory neurons. Leak rate a: x ← (1 − a)·x + a·tanh(...).", ""]
    return "\n".join(lines)
