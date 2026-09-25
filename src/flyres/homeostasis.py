"""Per-neuron gains set by a homeostatic rule: every neuron tunes itself toward the same activity level.

Per-region gains (regions.py) still can't separate a knot from the rest of its region. Real neurons
don't need a search for this: synaptic scaling (Turrigiano 1998) multiplies all of a neuron's incoming
synapses up when it's too quiet and down when it's too busy. This module does the same:

    x[t+1] = tanh(g ⊙ (W x[t]) + W_in u[t] + b)       g_i: neuron i's gain on its incoming synapses

Drive the reservoir with white noise, measure how big each neuron's recurrent input is (root mean
square over time of g_i (W x)_i), and move g_i halfway (in log terms) toward target / rms_i, at most
x2 or /2 per round, for a fixed number of rounds. Knots that are too loud get turned down, silent
stretches of the brain get turned up, with no knowledge of anatomy.

Two variants were tried first and collapsed. A rule on fluctuations (std of the output or of the input)
reads a saturated network, whose neurons sit near ±1 and barely move, as "too quiet" and turns it up
even further. Adding threshold adaptation (intrinsic plasticity: each neuron shifts its bias to cancel
its mean input) made the random wirings go silent. The RMS sees the big constant input of a saturated
neuron and turns it down.

The rule doesn't always settle. In these mostly excitatory networks, a neuron's input can have no
level near the target: a little more gain tips its neighborhood into a self-sustained active state, a
little less drops it back. So the rule runs for a fixed number of rounds and the result is judged as
it is; the summary reports how many neurons ended within x2 of the target. The one free parameter is
the target; like the gain sweep, every wiring
gets the same targets and keeps its best valid one (picked on one set of white-noise inputs,
reported on fresh ones). Neurons without recurrent inputs keep g = 1 (nothing to scale).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

from .config import ExperimentConfig
from .controls import WIRINGS
from .diagnostics import top_mode
from .experiment import build_matrix, prepare_subgraph, readout_neurons, reservoir_kwargs
from .regions import evaluate, neuron_regions, scale_rows, single_gain_peaks
from .reservoir import Reservoir
from .sweep import DEFAULT_GAINS, STABLE_MAX_UNSTABLE

TARGETS = (0.03, 0.1, 0.3, 1.0)    # RMS of each neuron's recurrent input (tanh bends from ~0.5 on)
N_ROUNDS = 30
ADAPT_STEPS, ADAPT_WASHOUT = 200, 100
DAMPING = 0.5                       # move halfway (in log terms) toward the correcting factor each round
STEP_CLIP = (0.5, 2.0)              # per round, a gain at most halves or doubles
GAIN_BOUNDS = (1e-3, 1e3)


def homeostatic_gains(sub, cfg: ExperimentConfig, W0, seed: int, target: float, rounds: int = N_ROUNDS,
                      steps: int = ADAPT_STEPS, washout: int = ADAPT_WASHOUT):
    """Synaptic scaling of W0's rows toward recurrent input of RMS `target`. Returns (gains, per-round
    history). The white-noise drive has its own seed, separate from the inputs used to judge the result."""
    W0 = sp.csr_matrix(W0, dtype=np.float32)
    u = np.random.default_rng([seed, 9]).uniform(-1.0, 1.0, size=(steps + washout, 1)).astype(np.float32)
    has_input = np.diff(W0.indptr) > 0
    g = np.ones(W0.shape[0])
    history = []
    for r in range(rounds):
        res = Reservoir(scale_rows(W0, g), sub.input_idx, 1, rng=np.random.default_rng([seed, 4]),
                        **reservoir_kwargs(cfg))
        X = res.run(u, washout=washout - 1)                       # states from step washout-1 on
        rec = np.asarray(W0 @ X[:-1].T, dtype=np.float64)         # (W x)_i over the next `steps` steps
        drive = np.sqrt((rec ** 2).mean(axis=1)) * g                 # RMS of g_i (W x)_i
        s = drive[has_input]
        history.append({"round": r, "median_drive": float(np.median(s)),
                        "near_target": float(np.mean(np.abs(np.log(np.maximum(s, 1e-12) / target)) < np.log(2))),
                        "silent": float(np.mean(s < 1e-6))})
        step = np.clip((target / np.maximum(drive, 1e-12)) ** DAMPING, *STEP_CLIP)
        g = np.where(has_input, np.clip(g * step, *GAIN_BOUNDS), 1.0)
    return g, pd.DataFrame(history)


def homeostasis_job(sub, cfg: ExperimentConfig, wiring: str, seed: int, codes: np.ndarray, regions: list[str],
                    gains=DEFAULT_GAINS, targets=TARGETS) -> dict:
    """One wiring and seed: its best single gain (for comparison), then the homeostatic rule at every
    target, keeping the best valid one."""
    t0 = time.time()
    metric = "memory_noisy" if cfg.memory.readout_noise > 0 else "memory"
    W_base, *_ = build_matrix(sub, cfg, wiring, seed)
    base = cfg.reservoir.spectral_radius
    readout_idx = readout_neurons(sub, cfg, seed)

    def at_gain(g):
        return (W_base * np.float32(g / base)).tocsr()

    singles = [{**evaluate(sub, cfg, at_gain(float(g)), readout_idx, seed), "gain": float(g)} for g in sorted(gains)]
    peaks = single_gain_peaks(singles, metric, 1)
    single = peaks[0] if peaks else singles[0]

    trials, adapted = [], {}
    for target in targets:
        g, hist = homeostatic_gains(sub, cfg, W_base, seed, target)
        r = evaluate(sub, cfg, scale_rows(W_base, g), readout_idx, seed)
        adapted[target] = g
        last = hist.iloc[-1]
        trials.append({"wiring": wiring, "seed": seed, "target": target, **r,
                       "median_drive": last["median_drive"], "near_target": last["near_target"],
                       "silent": last["silent"]})
    ok = [t for t in trials if t["valid"]]
    chosen = max(ok, key=lambda t: t[metric]) if ok else min(trials, key=lambda t: t["unstable_readouts"])
    g = adapted[chosen["target"]]
    W1 = scale_rows(W_base, g)

    fresh0 = evaluate(sub, cfg, at_gain(single["gain"]), readout_idx, seed, fresh=True)
    fresh1 = evaluate(sub, cfg, W1, readout_idx, seed, fresh=True)
    _, spread0, _ = top_mode(at_gain(single["gain"]), seed=seed)
    _, spread1, _ = top_mode(W1, seed=seed)
    has_input = np.diff(W_base.tocsr().indptr) > 0
    result = {"wiring": wiring, "seed": seed, "metric": metric, "single_gain": single["gain"],
              "target": chosen["target"], "valid_targets": len(ok), "picked_single": single[metric],
              "picked_homeostatic": chosen[metric], "spread_single": spread0, "spread_homeostatic": spread1,
              "gain_capped": float(np.mean(g[has_input] >= GAIN_BOUNDS[1])), "seconds": time.time() - t0}
    for tag, r in (("single", fresh0), ("homeostatic", fresh1)):
        result.update({f"{k}_{tag}": r[k] for k in ("memory", "memory_noisy", "active_readouts",
                                                    "unstable_readouts", "valid") if k in r})
    logg = np.log2(g)
    region_rows = pd.DataFrame({
        "wiring": wiring, "seed": seed, "region": regions,
        "neurons": np.bincount(codes, minlength=len(regions)),
        # geometric mean gain over the region's neurons that have recurrent inputs (1 if none)
        "factor": [float(2.0 ** logg[(codes == i) & has_input].mean()) if ((codes == i) & has_input).any() else 1.0
                   for i in range(len(regions))],
    })
    return {"result": result, "regions": region_rows, "trials": pd.DataFrame(trials)}


def run_homeostasis(cfg: ExperimentConfig, gains=DEFAULT_GAINS, targets=TARGETS, min_size: int = 30,
                    verbose: bool = True):
    """Every wiring x seed. Returns (results, regions, trials) DataFrames."""
    unknown = [w for w in cfg.wirings if w not in WIRINGS]
    if unknown:
        raise ValueError(f"unknown wirings {unknown}; choose from {WIRINGS}")
    sub = prepare_subgraph(cfg, verbose)
    codes, regions = neuron_regions(sub.neurons, min_size)
    jobs = [(w, s) for w in cfg.wirings for s in cfg.seeds]
    if verbose:
        print(f"homeostasis: {len(jobs)} jobs ({len(cfg.wirings)} wirings x {len(cfg.seeds)} seeds), "
              f"targets {list(targets)}")
    if cfg.n_jobs == 1:
        out = []
        for i, (w, s) in enumerate(jobs, 1):
            out.append(homeostasis_job(sub, cfg, w, s, codes, regions, gains, targets))
            if verbose:
                print(f"  [{i}/{len(jobs)}] {w:<18} seed {s}: {out[-1]['result']['seconds']:.0f}s", flush=True)
    else:
        out = Parallel(n_jobs=cfg.n_jobs, verbose=10 if verbose else 0)(
            delayed(homeostasis_job)(sub, cfg, w, s, codes, regions, gains, targets) for w, s in jobs)
    return (pd.DataFrame([o["result"] for o in out]), pd.concat([o["regions"] for o in out], ignore_index=True),
            pd.concat([o["trials"] for o in out], ignore_index=True))


# --------------------------------------------------------------------------- summary

def _pm(s: pd.Series, digits: int = 1) -> str:
    s = s.dropna()
    if s.empty:
        return "-"
    return f"{s.mean():.{digits}f}" + (f" ± {s.std():.{digits}f}" if len(s) > 1 else "")


def homeostasis_markdown(cfg: ExperimentConfig, results: pd.DataFrame, region_table: pd.DataFrame,
                         trials: pd.DataFrame) -> str:
    from .regions import geo_mean_factors

    noisy = "memory_noisy_single" in results.columns
    col = "memory_noisy" if noisy else "memory"
    rank = {w: i for i, w in enumerate(WIRINGS)}
    wirings = sorted(results["wiring"].unique(), key=lambda w: rank.get(w, 99))
    targets = sorted(trials["target"].unique())
    lines = [f"# Homeostatic gains: {cfg.name}", "",
             f"Every neuron scales its incoming synapses until its recurrent input has a target size (RMS under white "
             f"noise), {N_ROUNDS} damped rounds, at most x2 per round. Targets tried: "
             f"{', '.join(f'{t:g}' for t in targets)}; "
             f"each wiring keeps its best valid one (at most {STABLE_MAX_UNSTABLE:.0%} of readouts unstable in each of "
             f"{cfg.memory.stability_tests} latching tests). {len(cfg.seeds)} seed(s); targets are picked on one set of "
             "white-noise inputs and every number below comes from fresh ones. The rule doesn't always settle "
             "(see *neurons within x2 of target*): the gains are whatever it reached after the last round.", "",
             "## Memory on a fresh input", "",
             f"| wiring | best single gain | target | memory{' with noise' if noisy else ''}: single gain | "
             "homeostatic | change | noise-free: single gain | homeostatic | active readouts | valid on fresh input |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for w in wirings:
        r = results[results["wiring"] == w]
        d = (r[f"{col}_homeostatic"] - r[f"{col}_single"]).mean()
        lines.append(f"| {w} | {', '.join(f'{g:g}' for g in r['single_gain'])} | "
                     f"{', '.join(f'{t:g}' for t in r['target'])} | {_pm(r[f'{col}_single'])} | "
                     f"{_pm(r[f'{col}_homeostatic'])} | {d:+.1f} | {_pm(r['memory_single'])} | "
                     f"{_pm(r['memory_homeostatic'])} | "
                     f"{r['active_readouts_single'].mean():.0%} → {r['active_readouts_homeostatic'].mean():.0%} | "
                     f"{int(r['valid_single'].sum())}/{len(r)} → {int(r['valid_homeostatic'].sum())}/{len(r)} |")
    lines += ["", "## Every target (search inputs, mean over seeds)", "",
              "| wiring | target | " + ("memory with noise | " if noisy else "") + "memory (noise-free) | "
              "valid seeds | median input RMS after | neurons within x2 of target |", "|---|---|---|---|---|---|"
              + ("---|" if noisy else "")]
    for w in wirings:
        for t in targets:
            g = trials[(trials["wiring"] == w) & (trials["target"] == t)]
            lines.append(f"| {w} | {t:g} | " + (f"{_pm(g['memory_noisy'])} | " if noisy else "")
                         + f"{_pm(g['memory'])} | {int(g['valid'].sum())}/{len(g)} | "
                         f"{g['median_drive'].mean():.3g} | {g['near_target'].mean():.0%} |")
    factors = geo_mean_factors(region_table)
    sizes = region_table.drop_duplicates("region").set_index("region")["neurons"]
    lines += ["", "## Gains by region", "",
              "Geometric mean gain of the region's neurons after the rule (geometric mean over seeds); below 1 = the "
              "rule turned the region down.", "",
              "| region | neurons | " + " | ".join(wirings) + " |", "|---|---|" + "---|" * len(wirings)]
    for reg in factors.index:
        lines.append(f"| {reg} | {int(sizes[reg]):,} | " + " | ".join(f"{factors.loc[reg, w]:.2g}" for w in wirings)
                     + " |")
    lines += ["", "## Dominant mode", "", "Top mode spread (roughly how many neurons carry the largest eigenvalue).", "",
              "| wiring | best single gain | homeostatic | neurons at the gain cap |", "|---|---|---|---|"]
    for w in wirings:
        r = results[results["wiring"] == w]
        lines.append(f"| {w} | {r['spread_single'].mean():,.0f} | {r['spread_homeostatic'].mean():,.0f} | "
                     f"{r['gain_capped'].mean():.1%} |")
    lines.append("")
    return "\n".join(lines)
