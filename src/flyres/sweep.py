"""Gain sweep: every wiring at many gains, so each can be compared at its own best setting.

Forcing one spectral radius on every wiring is arbitrary: a wiring whose largest eigenvalue sits on a
small hot spot gets turned down much harder than one whose eigenvalues are spread out. The fair
comparison is the one reservoir-computing papers use for different architectures: tune the gain
per wiring and compare the best results. Here "best" means the highest memory capacity among the
gains where the reservoir is still valid, i.e. (almost) no readout remembers where it started.

Market forecasts aren't swept: they showed no skill at any setting, and memory capacity is the
cleaner measure of what the wiring itself contributes.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .benchmarks import memory_capacity, readout_stability
from .config import ExperimentConfig
from .controls import WIRINGS
from .experiment import build_matrix, prepare_subgraph, readout_neurons, reservoir_kwargs
from .reservoir import Reservoir
from .subgraph import Subgraph

DEFAULT_GAINS = (0.5, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0)
STABLE_MAX_UNSTABLE = 0.01  # a gain counts as valid if at most 1% of readouts are unstable, for every seed


def sweep_job(sub: Subgraph, cfg: ExperimentConfig, wiring: str, seed: int, gains) -> list[dict]:
    """One wiring and seed: build the matrix once, then measure memory and stability at every gain."""
    t0 = time.time()
    W_base, *_ = build_matrix(sub, cfg, wiring, seed)  # scaled to cfg.reservoir.spectral_radius
    base = cfg.reservoir.spectral_radius
    readout_idx = readout_neurons(sub, cfg, seed)
    mcfg = cfg.memory
    rows = []
    for gain in gains:
        W = (W_base * np.float32(gain / base)).tocsr()
        res = Reservoir(W, sub.input_idx, 1, rng=np.random.default_rng([seed, 4]), **reservoir_kwargs(cfg))
        mc, _ = memory_capacity(res, readout_idx, mcfg.n_steps, mcfg.max_delay, mcfg.washout,
                                rng=np.random.default_rng([seed, 5]))
        rows.append({"wiring": wiring, "seed": seed, "gain": float(gain), "memory_capacity": mc,
                     **readout_stability(res, readout_idx, rng=np.random.default_rng([seed, 6]))})
    rows[-1]["seconds"] = time.time() - t0
    return rows


def run_gain_sweep(cfg: ExperimentConfig, gains=DEFAULT_GAINS, verbose: bool = True) -> pd.DataFrame:
    unknown = [w for w in cfg.wirings if w not in WIRINGS]
    if unknown:
        raise ValueError(f"unknown wirings {unknown}; choose from {WIRINGS}")
    sub = prepare_subgraph(cfg, verbose)
    gains = sorted(float(g) for g in gains)
    jobs = [(w, s) for w in cfg.wirings for s in cfg.seeds]
    if verbose:
        print(f"gain sweep: {len(jobs)} jobs ({len(cfg.wirings)} wirings x {len(cfg.seeds)} seeds), "
              f"{len(gains)} gains each: {gains}")
    if cfg.n_jobs == 1:
        out = []
        for i, (w, s) in enumerate(jobs, 1):
            out.append(sweep_job(sub, cfg, w, s, gains))
            if verbose:
                print(f"  [{i}/{len(jobs)}] {w:<18} seed {s}: {out[-1][-1]['seconds']:.0f}s", flush=True)
    else:
        out = Parallel(n_jobs=cfg.n_jobs, verbose=10 if verbose else 0)(
            delayed(sweep_job)(sub, cfg, w, s, gains) for w, s in jobs)
    df = pd.DataFrame([r for rows in out for r in rows]).drop(columns="seconds", errors="ignore")
    return df.sort_values(["wiring", "gain", "seed"]).reset_index(drop=True)


def summarize_sweep(df: pd.DataFrame, max_unstable: float = STABLE_MAX_UNSTABLE) -> pd.DataFrame:
    """Per wiring and gain: mean and sd of memory over seeds, active share, worst-seed unstable share.
    `valid` marks gains where every seed stays at or below `max_unstable`."""
    g = df.groupby(["wiring", "gain"])
    out = pd.DataFrame({
        "memory": g["memory_capacity"].mean(),
        "memory_sd": g["memory_capacity"].std(),
        "active": g["active_readouts"].mean(),
        "unstable_worst": g["unstable_readouts"].max(),
        "seeds": g.size(),
    }).reset_index()
    out["valid"] = out["unstable_worst"] <= max_unstable
    return out


def best_valid(summary: pd.DataFrame) -> pd.DataFrame:
    """Each wiring's best valid gain (highest mean memory). Wirings with no valid gain get NaN."""
    rows = []
    for w, g in summary.groupby("wiring", sort=False):
        ok = g[g["valid"]]
        if len(ok):
            r = ok.loc[ok["memory"].idxmax()]
            rows.append({"wiring": w, "best_gain": r["gain"], "memory": r["memory"], "memory_sd": r["memory_sd"],
                         "active": r["active"], "valid_gains": len(ok), "gains_tried": len(g)})
        else:
            rows.append({"wiring": w, "best_gain": np.nan, "memory": np.nan, "memory_sd": np.nan,
                         "active": np.nan, "valid_gains": 0, "gains_tried": len(g)})
    order = {w: i for i, w in enumerate(WIRINGS)}
    return pd.DataFrame(rows).sort_values("wiring", key=lambda s: s.map(order)).reset_index(drop=True)


def sweep_markdown(cfg: ExperimentConfig, summary: pd.DataFrame, best: pd.DataFrame,
                   max_unstable: float = STABLE_MAX_UNSTABLE) -> str:
    rc = cfg.reservoir
    what = "spectral radius" if rc.normalize == "spectral" else "bulk scale (frobenius)"
    lines = [f"# Gain sweep: {cfg.name}", "",
             f"Memory capacity (delays 1..{cfg.memory.max_delay}) of every wiring at each {what}, "
             f"{len(cfg.seeds)} seed(s). A gain is valid when at most {max_unstable:.0%} of readouts are unstable "
             "for every seed.", "",
             "## Each wiring at its best valid gain", "",
             "| wiring | best gain | memory capacity | active readouts | valid gains |", "|---|---|---|---|---|"]
    for _, r in best.iterrows():
        if np.isfinite(r["memory"]):
            sd = f" ± {r['memory_sd']:.1f}" if np.isfinite(r["memory_sd"]) else ""
            lines.append(f"| {r['wiring']} | {r['best_gain']:g} | {r['memory']:.1f}{sd} | {r['active']:.0%} | "
                         f"{r['valid_gains']}/{r['gains_tried']} |")
        else:
            lines.append(f"| {r['wiring']} | none valid | - | - | 0/{r['gains_tried']} |")
    lines += ["", "## Full grid", "", "| wiring | gain | memory capacity | active readouts | unstable (worst seed) |",
              "|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        sd = f" ± {r['memory_sd']:.1f}" if np.isfinite(r["memory_sd"]) else ""
        flag = "" if r["valid"] else " (invalid)"
        lines.append(f"| {r['wiring']} | {r['gain']:g} | {r['memory']:.1f}{sd}{flag} | {r['active']:.0%} | "
                     f"{r['unstable_worst']:.0%} |")
    lines += ["", "Active readouts: share of readout neurons that move at all under white-noise input. Unstable: "
                  "share whose state still depends on inputs from hundreds of steps ago (latched or chaotic).", ""]
    return "\n".join(lines)
