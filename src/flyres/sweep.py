"""Gain sweep: every wiring at many gains, so each can be compared at its own best setting.

Forcing one spectral radius on every wiring is arbitrary: a wiring whose largest eigenvalue sits on a
small hot spot gets turned down much harder than one whose eigenvalues are spread out. The fair
comparison is the one reservoir-computing papers use for different architectures: tune the gain
per wiring and compare the best results. Here "best" means the highest memory capacity among the
gains where the reservoir is still valid, i.e. (almost) no readout remembers where it started. It is
judged on memory *with readout noise* (see benchmarks.memory_capacities), since noise-free memory can
come from fluctuations of a millionth; the noise-free numbers are reported too.

Market forecasts aren't swept: they showed no skill at any setting, and memory capacity is the
cleaner measure of what the wiring itself contributes.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .benchmarks import memory_capacities, readout_stability
from .config import ExperimentConfig
from .controls import WIRINGS
from .experiment import build_matrix, memory_noise_levels, prepare_subgraph, readout_neurons, reservoir_kwargs
from .reservoir import Reservoir
from .subgraph import Subgraph

DEFAULT_GAINS = (0.5, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0)
STABLE_MAX_UNSTABLE = 0.01  # a gain counts as valid if at most 1% of readouts are unstable, in every latching
                            # test (memory.stability_tests per seed) and for every seed


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
        caps = memory_capacities(res, readout_idx, memory_noise_levels(cfg), mcfg.n_steps, mcfg.max_delay,
                                 mcfg.washout, rng=np.random.default_rng([seed, 5]))
        row = {"wiring": wiring, "seed": seed, "gain": float(gain), "memory_capacity": caps[0.0][0]}
        if mcfg.readout_noise > 0:
            row["memory_capacity_noisy"] = caps[mcfg.readout_noise][0]
        rows.append({**row, **readout_stability(res, readout_idx, rng=np.random.default_rng([seed, 6]),
                                                n_tests=mcfg.stability_tests)})
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
    cols = {
        "memory": g["memory_capacity"].mean(),
        "memory_sd": g["memory_capacity"].std(),
    }
    if "memory_capacity_noisy" in df.columns:
        cols["memory_noisy"] = g["memory_capacity_noisy"].mean()
        cols["memory_noisy_sd"] = g["memory_capacity_noisy"].std()
    out = pd.DataFrame({
        **cols,
        "active": g["active_readouts"].mean(),
        "unstable_worst": g["unstable_readouts"].max(),
        "seeds": g.size(),
    }).reset_index()
    out["valid"] = out["unstable_worst"] <= max_unstable
    return out


def best_valid(summary: pd.DataFrame, metric: str | None = None) -> pd.DataFrame:
    """Each wiring's best valid gain by `metric` ("memory_noisy" when available, else "memory"), with both
    memory numbers at that gain. Wirings with no valid gain get NaN."""
    metric = metric or ("memory_noisy" if "memory_noisy" in summary.columns else "memory")
    extra = [c for c in ("memory_noisy", "memory_noisy_sd") if c in summary.columns]
    rows = []
    for w, g in summary.groupby("wiring", sort=False):
        ok = g[g["valid"]]
        base = {"wiring": w, "valid_gains": len(ok), "gains_tried": len(g)}
        if len(ok):
            r = ok.loc[ok[metric].idxmax()]
            rows.append({**base, "best_gain": r["gain"], "memory": r["memory"], "memory_sd": r["memory_sd"],
                         **{c: r[c] for c in extra}, "active": r["active"]})
        else:
            rows.append({**base, "best_gain": np.nan, "memory": np.nan, "memory_sd": np.nan,
                         **{c: np.nan for c in extra}, "active": np.nan})
    order = {w: i for i, w in enumerate(WIRINGS)}
    cols = ["wiring", "best_gain", "memory", "memory_sd", *extra, "active", "valid_gains", "gains_tried"]
    return pd.DataFrame(rows)[cols].sort_values("wiring", key=lambda s: s.map(order)).reset_index(drop=True)


def _pm(mean, sd, digits: int = 1) -> str:
    if not np.isfinite(mean):
        return "-"
    return f"{mean:.{digits}f}" + (f" ± {sd:.{digits}f}" if np.isfinite(sd) else "")


def sweep_markdown(cfg: ExperimentConfig, summary: pd.DataFrame, best: pd.DataFrame,
                   max_unstable: float = STABLE_MAX_UNSTABLE) -> str:
    rc = cfg.reservoir
    what = "spectral radius" if rc.normalize == "spectral" else "bulk scale (frobenius)"
    noisy = "memory_noisy" in summary.columns
    noise = cfg.memory.readout_noise
    lines = [f"# Gain sweep: {cfg.name}", "",
             f"Memory capacity (delays 1..{cfg.memory.max_delay}) of every wiring at each {what}, "
             f"{len(cfg.seeds)} seed(s). A gain is valid when at most {max_unstable:.0%} of readouts are unstable "
             f"in every latching test ({cfg.memory.stability_tests} per seed) and for every seed.", ""]
    if noisy:
        lines += [f"*With readout noise*: noise of std {noise:g} ({noise:.1%} of a neuron's maximum activity) is added to every "
                  "readout before fitting, so only memory that survives a little noise counts. Best gains are picked "
                  "on this number. The noise-free number is the standard benchmark but can come from fluctuations of "
                  "a millionth.", ""]
    lines += ["## Each wiring at its best valid gain", "",
              "| wiring | best gain | " + ("memory with readout noise | " if noisy else "")
              + "memory (noise-free) | active readouts | valid gains |",
              "|---|---|" + ("---|" if noisy else "") + "---|---|---|"]
    for _, r in best.iterrows():
        if np.isfinite(r["memory"]):
            lines.append(f"| {r['wiring']} | {r['best_gain']:g} | "
                         + (f"{_pm(r['memory_noisy'], r['memory_noisy_sd'])} | " if noisy else "")
                         + f"{_pm(r['memory'], r['memory_sd'])} | {r['active']:.0%} | "
                         f"{r['valid_gains']}/{r['gains_tried']} |")
        else:
            lines.append(f"| {r['wiring']} | none valid | " + ("- | " if noisy else "")
                         + f"- | - | 0/{r['gains_tried']} |")
    lines += ["", "## Full grid", "",
              "| wiring | gain | " + ("memory with readout noise | " if noisy else "")
              + "memory (noise-free) | active readouts | unstable (worst seed) |",
              "|---|---|" + ("---|" if noisy else "") + "---|---|---|"]
    for _, r in summary.iterrows():
        flag = "" if r["valid"] else " (invalid)"
        lines.append(f"| {r['wiring']} | {r['gain']:g} | "
                     + (f"{_pm(r['memory_noisy'], r['memory_noisy_sd'])}{flag} | " if noisy else "")
                     + f"{_pm(r['memory'], r['memory_sd'])}{flag} | {r['active']:.0%} | {r['unstable_worst']:.0%} |")
    lines += ["", "Active readouts: share of readout neurons that move by more than 0.001 under white-noise input. "
                  "Unstable: share whose state still depends on inputs from hundreds of steps ago (latched or "
                  "chaotic).", ""]
    return "\n".join(lines)
