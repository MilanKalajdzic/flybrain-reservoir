"""NARMA-10: a second market-free benchmark, one that needs memory *and* nonlinearity.

Memory capacity asks a reservoir to replay its input; NARMA-10 (Atiya & Parlos 2000, the standard
reservoir-computing benchmark since Jaeger 2003) asks it to compute something from it:

    y(t+1) = 0.3 y(t) + 0.05 y(t) * sum_{i=0..9} y(t-i) + 1.5 u(t-9) u(t) + 0.1,   u(t) ~ U[0, 0.5]

The readout sees the reservoir state after u(t) (plus u(t) itself) and predicts y(t+1). The product
u(t-9) u(t) needs a memory of 10 steps and a multiplication, so a linear model of the recent inputs
can't do it; a good reservoir gets the error well below that. Scored by NRMSE = RMSE / std(y) on
held-out steps (lower is better; 1 = no better than predicting the mean).

Like memory capacity, the main score adds readout noise (memory.readout_noise, 0.1% of a neuron's range)
to the recorded states before fitting. Noise-free, a readout decodes fluctuations of a millionth: on the
whole brain the fly's NARMA error was 0.41 noise-free and 0.83 with the noise (the degree-preserving
shuffle: 0.39 and 0.42). The noise-free score is kept for reference.

Like the gain sweep, every wiring runs at many gains and is judged at its best valid one, with the
same reservoirs (input weights, bias, readout neurons, latching tests) as the memory benchmark.
"""
from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from .benchmarks import readout_stability
from .config import ExperimentConfig
from .controls import WIRINGS
from .experiment import build_matrix, prepare_subgraph, readout_neurons, reservoir_kwargs
from .readout import fit_ridge
from .reservoir import Reservoir
from .sweep import DEFAULT_GAINS, STABLE_MAX_UNSTABLE

N_STEPS, WASHOUT, N_TEST = 4000, 200, 1000
LAGS = 10


@lru_cache(maxsize=16)
def narma10(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Input u (n,) and target y (n,) with y[t] = the NARMA-10 value at t+1, i.e. what a model that has
    seen u[0..t] should predict. Redrawn with the next sub-seed if the recursion blows up (it
    occasionally does for unlucky inputs)."""
    for attempt in range(100):
        u = np.random.default_rng([seed, 11, attempt]).uniform(0.0, 0.5, n + 1)
        y = np.zeros(n + 1)
        for t in range(9, n):
            y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * y[t - 9:t + 1].sum() + 1.5 * u[t - 9] * u[t] + 0.1
        if np.isfinite(y).all() and np.abs(y).max() < 1.0:
            u, y = u[:n], y[1:]
            u.flags.writeable = y.flags.writeable = False  # cached: shared between calls, so read-only
            return u, y
    raise RuntimeError("NARMA-10 kept diverging")


def nrmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)) / np.std(y))


def _fit_score(X: np.ndarray, y: np.ndarray, penalty=None) -> float:
    """Ridge readout on rows WASHOUT..-N_TEST (penalty picked on the end of that window), NRMSE on the
    last N_TEST rows."""
    tr, te = slice(WASHOUT, len(y) - N_TEST), slice(len(y) - N_TEST, len(y))
    model = fit_ridge(X[tr], y[tr], val_frac=0.2, gap=1, penalty_factor=penalty)
    return nrmse(model.predict(X[te]), y[te])


def linear_baseline(seed: int, n_steps: int = N_STEPS) -> float:
    """Ridge on the last LAGS inputs u(t), ..., u(t-9): memory, but no nonlinearity."""
    u, y = narma10(n_steps, seed)
    X = np.stack([np.r_[np.zeros(k), u[:len(u) - k]] for k in range(LAGS)], axis=1)
    return _fit_score(X, y)


def narma_score(reservoir: Reservoir, record_idx, seed: int, input_penalty: float = 0.001,
                readout_noise: float = 0.0, n_steps: int = N_STEPS) -> dict:
    """NRMSE of a readout on [u(t), reservoir state] for the NARMA-10 series of `seed`: noise-free
    ("nrmse") and, if readout_noise > 0, with that much Gaussian noise on every state ("nrmse_noisy")."""
    u, y = narma10(n_steps, seed)
    states = reservoir.run((4.0 * u - 1.0)[:, None].astype(np.float32), record_idx=record_idx)  # u in [-1, 1]
    penalty = np.r_[input_penalty, np.ones(states.shape[1])]
    out = {"nrmse": _fit_score(np.hstack([u[:, None], states]), y, penalty)}
    if readout_noise > 0:
        noisy = states + np.random.default_rng([seed, 12]).normal(0.0, readout_noise, states.shape)
        out["nrmse_noisy"] = _fit_score(np.hstack([u[:, None], noisy]), y, penalty)
    return out


def narma_job(sub, cfg: ExperimentConfig, wiring: str, seed: int, gains) -> list[dict]:
    """One wiring and seed at every gain: NARMA-10 error and the same latching tests as the gain sweep."""
    t0 = time.time()
    W_base, *_ = build_matrix(sub, cfg, wiring, seed)
    base = cfg.reservoir.spectral_radius
    readout_idx = readout_neurons(sub, cfg, seed)
    rows = []
    for g in gains:
        res = Reservoir((W_base * np.float32(g / base)).tocsr(), sub.input_idx, 1,
                        rng=np.random.default_rng([seed, 4]), **reservoir_kwargs(cfg))
        stab = readout_stability(res, readout_idx, rng=np.random.default_rng([seed, 6]),
                                 n_tests=cfg.memory.stability_tests)
        rows.append({"wiring": wiring, "seed": seed, "gain": float(g), **stab,
                     **narma_score(res, readout_idx, seed, cfg.reservoir.input_penalty, cfg.memory.readout_noise)})
    rows[-1]["seconds"] = time.time() - t0
    return rows


def run_narma(cfg: ExperimentConfig, gains=DEFAULT_GAINS, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every wiring x seed x gain. Returns (runs, linear baseline per seed)."""
    unknown = [w for w in cfg.wirings if w not in WIRINGS]
    if unknown:
        raise ValueError(f"unknown wirings {unknown}; choose from {WIRINGS}")
    sub = prepare_subgraph(cfg, verbose)
    gains = sorted(float(g) for g in gains)
    jobs = [(w, s) for w in cfg.wirings for s in cfg.seeds]
    if verbose:
        print(f"NARMA-10: {len(jobs)} jobs ({len(cfg.wirings)} wirings x {len(cfg.seeds)} seeds), "
              f"{len(gains)} gains each")
    if cfg.n_jobs == 1:
        out = []
        for i, (w, s) in enumerate(jobs, 1):
            out.append(narma_job(sub, cfg, w, s, gains))
            if verbose:
                print(f"  [{i}/{len(jobs)}] {w:<18} seed {s}: {out[-1][-1]['seconds']:.0f}s", flush=True)
    else:
        out = Parallel(n_jobs=cfg.n_jobs, verbose=10 if verbose else 0)(
            delayed(narma_job)(sub, cfg, w, s, gains) for w, s in jobs)
    runs = pd.DataFrame([r for rows in out for r in rows]).drop(columns="seconds", errors="ignore")
    baseline = pd.DataFrame({"seed": cfg.seeds, "nrmse": [linear_baseline(s) for s in cfg.seeds]})
    return runs.sort_values(["wiring", "gain", "seed"]).reset_index(drop=True), baseline


def summarize_narma(runs: pd.DataFrame, max_unstable: float = STABLE_MAX_UNSTABLE) -> pd.DataFrame:
    """Per wiring and gain: mean and sd of NRMSE over seeds, and whether every seed passes the latching tests."""
    g = runs.groupby(["wiring", "gain"])
    cols = {"nrmse": g["nrmse"].mean(), "nrmse_sd": g["nrmse"].std()}
    if "nrmse_noisy" in runs.columns:
        cols.update(nrmse_noisy=g["nrmse_noisy"].mean(), nrmse_noisy_sd=g["nrmse_noisy"].std())
    out = pd.DataFrame({**cols, "active": g["active_readouts"].mean(), "unstable_worst": g["unstable_readouts"].max(),
                        "seeds": g.size()}).reset_index()
    out["valid"] = out["unstable_worst"] <= max_unstable
    return out


def metric_of(frame: pd.DataFrame) -> str:
    """The main score: with readout noise when it was measured, noise-free otherwise."""
    return "nrmse_noisy" if "nrmse_noisy" in frame.columns else "nrmse"


def best_narma(grid: pd.DataFrame, standard_gain: float) -> pd.DataFrame:
    """Each wiring's best valid gain by the main score (lowest error), with both scores there."""
    metric = metric_of(grid)
    scores = [c for c in ("nrmse_noisy", "nrmse_noisy_sd", "nrmse", "nrmse_sd") if c in grid.columns]
    rows = []
    for w, g in grid.groupby("wiring", sort=False):
        ok = g[g["valid"]]
        std = g[np.isclose(g["gain"], standard_gain)]
        row = {"wiring": w, "standard": std[metric].iloc[0] if len(std) else np.nan,
               "standard_nrmse": std["nrmse"].iloc[0] if len(std) else np.nan,
               "valid_gains": len(ok), "gains_tried": len(g)}
        if len(ok):
            b = ok.loc[ok[metric].idxmin()]
            row.update(best_gain=b["gain"], active=b["active"], **{c: b[c] for c in scores})
        else:
            row.update(best_gain=np.nan, active=np.nan, **{c: np.nan for c in scores})
        rows.append(row)
    rank = {w: i for i, w in enumerate(WIRINGS)}
    return pd.DataFrame(rows).sort_values("wiring", key=lambda s: s.map(rank)).reset_index(drop=True)


def memory_link(runs: pd.DataFrame, sweep: pd.DataFrame | None, max_unstable: float = STABLE_MAX_UNSTABLE):
    """NARMA error next to memory capacity for the same reservoir (wiring, seed, gain), valid reservoirs only.
    Memory comes from the gain sweep's sweep.csv (same seeds and gains)."""
    if sweep is None or not len(sweep):
        return None
    col = "memory_capacity_noisy" if "memory_capacity_noisy" in sweep.columns else "memory_capacity"
    m = runs.merge(sweep[["wiring", "seed", "gain", col]].rename(columns={col: "memory"}),
                   on=["wiring", "seed", "gain"], how="inner")
    m["error"] = m[metric_of(runs)]
    m = m[m["unstable_readouts"] <= max_unstable]
    return m if len(m) > 2 else None


def narma_markdown(cfg: ExperimentConfig, grid: pd.DataFrame, best: pd.DataFrame, baseline: pd.DataFrame,
                   link: pd.DataFrame | None) -> str:
    noisy = "nrmse_noisy" in grid.columns
    noise = cfg.memory.readout_noise
    lines = [f"# NARMA-10: {cfg.name}", "",
             f"NRMSE on the last {N_TEST} of {N_STEPS} steps (lower is better; 1 = predicting the mean). Readout on "
             f"u(t) and the reservoir state, ridge penalty picked on the end of the training window. "
             f"{len(cfg.seeds)} seed(s); a gain is valid when at most {STABLE_MAX_UNSTABLE:.0%} of readouts are unstable "
             f"in each of {cfg.memory.stability_tests} latching tests, for every seed.", ""]
    if noisy:
        lines += [f"*With readout noise*: noise of std {noise:g} ({noise:.1%} of a neuron's range) on every recorded "
                  "state before fitting, as for memory capacity. Best gains are picked on this score; the noise-free "
                  "score can come from fluctuations of a millionth.", ""]
    lines += [f"Linear baseline (ridge on the last {LAGS} inputs, no reservoir): NRMSE "
              f"{baseline['nrmse'].mean():.3f} ± {baseline['nrmse'].std():.3f}.", "",
              "## Each wiring", "",
              f"| wiring | at the standard gain ({cfg.reservoir.spectral_radius:g}) | best valid gain | "
              + ("with readout noise | " if noisy else "") + "noise-free | active readouts | valid gains |",
              "|---|---|---|" + ("---|" if noisy else "") + "---|---|---|"]
    for _, r in best.iterrows():
        def pm(m, s):
            return f"{m:.3f} ± {s:.3f}" if np.isfinite(m) else "-"
        gain = f"{r['best_gain']:g}" if np.isfinite(r["best_gain"]) else "none valid"
        lines.append(f"| {r['wiring']} | {r['standard']:.3f} | {gain} | "
                     + (f"{pm(r['nrmse_noisy'], r['nrmse_noisy_sd'])} | " if noisy else "")
                     + f"{pm(r['nrmse'], r['nrmse_sd'])} | "
                     + (f"{r['active']:.0%}" if np.isfinite(r["active"]) else "-")
                     + f" | {r['valid_gains']}/{r['gains_tried']} |")
    if link is not None:
        rho, p = stats.spearmanr(link["memory"], link["error"])
        lines += ["", "## Does memory capacity predict NARMA?", "",
                  f"Over every valid reservoir (wiring x seed x gain, n = {len(link)}), Spearman correlation between "
                  f"memory capacity (from the gain sweep) and NARMA-10 error{' with readout noise' if noisy else ''}: "
                  f"ρ = {rho:+.2f} (p = {p:.3f}). Negative = more memory, lower error."]
    lines += ["", "## Full grid", "",
              "| wiring | gain | " + ("NRMSE with readout noise | " if noisy else "") + "NRMSE noise-free | "
              "active readouts | unstable (worst seed) |", "|---|---|" + ("---|" if noisy else "") + "---|---|---|"]
    for _, r in grid.iterrows():
        flag = "" if r["valid"] else " (invalid)"
        lines.append(f"| {r['wiring']} | {r['gain']:g} | "
                     + (f"{r['nrmse_noisy']:.3f} ± {r['nrmse_noisy_sd']:.3f}{flag} | " if noisy else "")
                     + f"{r['nrmse']:.3f} ± {r['nrmse_sd']:.3f}{flag} | {r['active']:.0%} | {r['unstable_worst']:.0%} |")
    lines.append("")
    return "\n".join(lines)
