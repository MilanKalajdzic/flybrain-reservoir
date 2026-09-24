"""Per-region gains: every brain region gets its own gain, and a search finds the best combination.

One global gain can only suit the loudest hot spot (diagnostics.py). A real brain isn't stuck with
that: neuromodulators and local inhibition set how excitable each region is. This module gives the
model the same freedom. The neurons are split into the anatomical regions of the activity figures
(antennal lobe, mushroom body, central complex, visual system, nerve cord, ...), every neuron's
incoming synapses are multiplied by its region's factor, and a coordinate search looks for the
factors with the most memory while the reservoir stays valid.

Fair to every wiring: the same neurons carry the same region labels in every wiring, and every
wiring gets the same search. The search starts from the wiring's best single gain, so any
improvement comes from the per-region freedom. Factors are picked on one white-noise input and the
reported numbers come from a fresh one, so the search can't just fit that input's quirks.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

from .activity import neuron_group
from .benchmarks import memory_capacities, readout_stability
from .config import ExperimentConfig
from .controls import WIRINGS
from .diagnostics import top_mode
from .experiment import build_matrix, memory_noise_levels, prepare_subgraph, readout_neurons, reservoir_kwargs
from .reservoir import Reservoir, spectral_radius
from .subgraph import Subgraph
from .sweep import DEFAULT_GAINS, STABLE_MAX_UNSTABLE

# short names for tables; all sensor types share one region (their incoming synapses barely matter)
REGION_NAMES = {
    "Antennal lobe (smell processing)": "antennal lobe", "Mushroom body (learning)": "mushroom body",
    "Central complex (navigation)": "central complex", "Other central brain": "other central brain",
    "Descending (brain to body)": "descending", "Ascending (body to brain)": "ascending",
    "Motor neurons": "motor", "Nerve cord": "nerve cord", "Visual": "visual", "Other": "other",
}
SEARCH_STEPS = (2.0, 2 ** 0.5)  # first pass doubles/halves a region's factor, second refines by sqrt(2)
MAX_FACTOR = 64.0
MIN_IMPROVEMENT = 0.05          # memory units; smaller gains aren't worth a move


def neuron_regions(neurons: pd.DataFrame, min_size: int = 30) -> tuple[np.ndarray, list[str]]:
    """Region code per neuron and region names, biggest region first. Regions with fewer than
    `min_size` neurons are folded into "other"."""
    cls = neurons["class"] if "class" in neurons.columns else pd.Series([None] * len(neurons))
    groups = [neuron_group(s, c) for s, c in zip(neurons["superclass"], cls)]
    names = pd.Series(["sensory" if g.endswith("sensors") else REGION_NAMES.get(g, g) for g in groups])
    counts = names.value_counts()
    names = names.where(~names.isin(counts.index[counts < min_size]), "other")
    counts = names.value_counts()
    order = list(counts.index)
    if "other" in order:  # the catch-all goes last whatever its size
        order.remove("other")
        order.append("other")
    codes = names.map({r: i for i, r in enumerate(order)}).to_numpy(dtype=np.int64)
    return codes, order


def scale_rows(W: sp.spmatrix, factors: np.ndarray) -> sp.csr_matrix:
    """Multiply every neuron's incoming synapses (row of W[post, pre]) by its factor."""
    return (sp.diags(np.asarray(factors, dtype=np.float32)) @ sp.csr_matrix(W, dtype=np.float32)).tocsr()


def internal_radius(W: sp.spmatrix, codes: np.ndarray, n_regions: int) -> np.ndarray:
    """Largest |eigenvalue| of each region's own block (connections inside the region only): how loud
    that region is on its own."""
    W = sp.csr_matrix(W)
    out = np.zeros(n_regions)
    for r in range(n_regions):
        idx = np.flatnonzero(codes == r)
        block = W[idx][:, idx]
        out[r] = spectral_radius(block) if block.nnz else 0.0
    return out


def evaluate(sub: Subgraph, cfg: ExperimentConfig, W, readout_idx, seed: int, fresh: bool = False) -> dict:
    """Memory (noise-free and with readout noise) and readout stability of one matrix. The reservoir's
    input weights and bias are the experiment's; the white-noise test input is the gain sweep's
    (`fresh=False`, used to pick gains) or a new one never used for picking (`fresh=True`)."""
    mc = cfg.memory
    k = 7 if fresh else 5
    res = Reservoir(W, sub.input_idx, 1, rng=np.random.default_rng([seed, 4]), **reservoir_kwargs(cfg))
    caps = memory_capacities(res, readout_idx, memory_noise_levels(cfg), mc.n_steps, mc.max_delay, mc.washout,
                             rng=np.random.default_rng([seed, k]))
    out = {"memory": caps[0.0][0]}
    if mc.readout_noise > 0:
        out["memory_noisy"] = caps[mc.readout_noise][0]
    out.update(readout_stability(res, readout_idx, rng=np.random.default_rng([seed, k + 1])))
    out["valid"] = out["unstable_readouts"] <= STABLE_MAX_UNSTABLE
    return out


def region_search_job(sub: Subgraph, cfg: ExperimentConfig, wiring: str, seed: int, codes: np.ndarray,
                      regions: list[str], gains=DEFAULT_GAINS, steps=SEARCH_STEPS) -> dict:
    """One wiring and seed. Stage 1: best valid single gain (like the gain sweep, for this seed).
    Stage 2: from there, go through the regions biggest first and move each one's factor up (or down)
    while memory improves and the reservoir stays valid; then repeat with smaller steps."""
    t0 = time.time()
    metric = "memory_noisy" if cfg.memory.readout_noise > 0 else "memory"
    W_base, _, _, raw_gain = build_matrix(sub, cfg, wiring, seed)
    base = cfg.reservoir.spectral_radius
    readout_idx = readout_neurons(sub, cfg, seed)
    trace = []

    def score(W, stage, gain, region=None, factor=1.0):
        r = evaluate(sub, cfg, W, readout_idx, seed)
        trace.append({"wiring": wiring, "seed": seed, "eval": len(trace) + 1, "stage": stage, "gain": gain,
                      "region": region, "factor": factor, **r})
        return r

    best, single_gain = None, None
    for g in sorted(float(x) for x in gains):
        r = score((W_base * np.float32(g / base)).tocsr(), "single gain", g)
        if r["valid"] and (best is None or r[metric] > best[metric]):
            best, single_gain = r, g
    if best is None:  # nothing valid: fall back to the lowest gain and report it as invalid
        single_gain = min(gains)
        best = trace[0]
    W0 = (W_base * np.float32(single_gain / base)).tocsr()
    start = best

    factors = np.ones(len(regions))
    for step in steps:
        for r in range(len(regions)):
            for direction in (step, 1.0 / step):
                moved = False
                while True:
                    trial = factors.copy()
                    trial[r] *= direction
                    if not 1.0 / MAX_FACTOR <= trial[r] <= MAX_FACTOR:
                        break
                    res = score(scale_rows(W0, trial[codes]), "regions", single_gain, regions[r], float(trial[r]))
                    if not (res["valid"] and res[metric] > best[metric] + MIN_IMPROVEMENT):
                        break
                    factors, best, moved = trial, res, True
                if moved:
                    break  # found the right direction for this region; don't try the other one
    W1 = scale_rows(W0, factors[codes])

    fresh0 = evaluate(sub, cfg, W0, readout_idx, seed, fresh=True)
    fresh1 = evaluate(sub, cfg, W1, readout_idx, seed, fresh=True)
    rho1, spread1, _ = top_mode(W1, seed=seed)
    _, spread0, _ = top_mode(W0, seed=seed)
    raw_scale = raw_gain / base  # W_base * raw_scale = the raw signed weights
    result = {"wiring": wiring, "seed": seed, "metric": metric, "single_gain": single_gain,
              "evals": len(trace), "picked_single": start[metric], "picked_regions": best[metric],
              "spread_single": spread0, "spread_regions": spread1, "radius_regions": rho1,
              "seconds": time.time() - t0}
    for tag, r in (("single", fresh0), ("regions", fresh1)):
        result.update({f"{k}_{tag}": r[k] for k in ("memory", "memory_noisy", "active_readouts",
                                                    "unstable_readouts", "valid") if k in r})
    region_rows = pd.DataFrame({
        "wiring": wiring, "seed": seed, "region": regions,
        "neurons": np.bincount(codes, minlength=len(regions)),
        "readouts": np.bincount(codes[readout_idx], minlength=len(regions)),
        "internal_radius": internal_radius(W_base * np.float32(raw_scale), codes, len(regions)),
        "factor": factors,
    })
    return {"result": result, "regions": region_rows, "trace": pd.DataFrame(trace)}


def run_region_search(cfg: ExperimentConfig, gains=DEFAULT_GAINS, min_size: int = 30, verbose: bool = True):
    """Every wiring x seed. Returns (results, regions, trace) DataFrames."""
    unknown = [w for w in cfg.wirings if w not in WIRINGS]
    if unknown:
        raise ValueError(f"unknown wirings {unknown}; choose from {WIRINGS}")
    sub = prepare_subgraph(cfg, verbose)
    codes, regions = neuron_regions(sub.neurons, min_size)
    jobs = [(w, s) for w in cfg.wirings for s in cfg.seeds]
    if verbose:
        sizes = np.bincount(codes, minlength=len(regions))
        print(f"{len(regions)} regions: " + ", ".join(f"{r} {n:,}" for r, n in zip(regions, sizes)))
        print(f"region search: {len(jobs)} jobs ({len(cfg.wirings)} wirings x {len(cfg.seeds)} seeds)")
    if cfg.n_jobs == 1:
        out = []
        for i, (w, s) in enumerate(jobs, 1):
            out.append(region_search_job(sub, cfg, w, s, codes, regions, gains))
            r = out[-1]["result"]
            if verbose:
                print(f"  [{i}/{len(jobs)}] {w:<18} seed {s}: {r['evals']} evals, {r['seconds']:.0f}s", flush=True)
    else:
        out = Parallel(n_jobs=cfg.n_jobs, verbose=10 if verbose else 0)(
            delayed(region_search_job)(sub, cfg, w, s, codes, regions, gains) for w, s in jobs)
    results = pd.DataFrame([o["result"] for o in out])
    region_table = pd.concat([o["regions"] for o in out], ignore_index=True)
    trace = pd.concat([o["trace"] for o in out], ignore_index=True)
    return results, region_table, trace


# --------------------------------------------------------------------------- summary

def _order(names) -> list:
    rank = {w: i for i, w in enumerate(WIRINGS)}
    return sorted(set(names), key=lambda w: rank.get(w, len(rank)))


def _pm(s: pd.Series, digits: int = 1) -> str:
    s = s.dropna()
    if s.empty:
        return "-"
    return f"{s.mean():.{digits}f}" + (f" ± {s.std():.{digits}f}" if len(s) > 1 else "")


def geo_mean_factors(region_table: pd.DataFrame) -> pd.DataFrame:
    """Region x wiring table of the chosen factor, geometric mean over seeds."""
    t = region_table.assign(log2=np.log2(region_table["factor"]))
    wide = t.pivot_table(index="region", columns="wiring", values="log2", aggfunc="mean", sort=False)
    return (2.0 ** wide)[_order(wide.columns)]


def region_markdown(cfg: ExperimentConfig, results: pd.DataFrame, region_table: pd.DataFrame) -> str:
    noisy = "memory_noisy_single" in results.columns
    noise = cfg.memory.readout_noise
    wirings = _order(results["wiring"])
    lines = [f"# Per-region gains: {cfg.name}", "",
             f"Every region's incoming synapses get their own factor on top of the wiring's best single gain. "
             f"{len(cfg.seeds)} seed(s); factors are searched per seed on one white-noise input, and every number "
             "below comes from a fresh input the search never saw. Valid = at most "
             f"{STABLE_MAX_UNSTABLE:.0%} of readouts unstable.", ""]
    if noisy:
        lines += [f"Memory *with readout noise* (std {noise:g}, {noise:.1%} of a neuron's range) is what the search "
                  "maximizes; noise-free memory is shown for reference.", ""]
    head = "| wiring | best single gain | "
    head += ("memory with noise: single gain | with per-region gains | change | " if noisy else "")
    head += "noise-free: single gain | with per-region gains | active readouts | valid seeds |"
    lines += ["## Memory on a fresh input", "", head, "|---|" + "---|" * (head.count("|") - 2)]
    for w in wirings:
        r = results[results["wiring"] == w]
        row = f"| {w} | {', '.join(f'{g:g}' for g in r['single_gain'])} | "
        if noisy:
            d = r["memory_noisy_regions"] - r["memory_noisy_single"]
            row += f"{_pm(r['memory_noisy_single'])} | {_pm(r['memory_noisy_regions'])} | {d.mean():+.1f} | "
        row += (f"{_pm(r['memory_single'])} | {_pm(r['memory_regions'])} | "
                f"{r['active_readouts_single'].mean():.0%} → {r['active_readouts_regions'].mean():.0%} | "
                f"{int(r['valid_regions'].sum())}/{len(r)} |")
        lines.append(row)

    regions = list(dict.fromkeys(region_table["region"]))
    first = region_table.drop_duplicates("region").set_index("region")
    readouts = region_table.groupby("region")["readouts"].mean()
    rad = region_table.pivot_table(index="region", columns="wiring", values="internal_radius", aggfunc="mean")
    factors = geo_mean_factors(region_table)
    lines += ["", "## Regions", "",
              "Internal radius: largest |eigenvalue| of the region's own connections, in raw synapse units "
              "(how loud the region is on its own; the whole wiring's radius sets the single gain). "
              "Factor: what the search multiplied the region's incoming synapses by (geometric mean over seeds; "
              "above 1 = turned up).", "",
              "| region | neurons | readouts | " + " | ".join(f"radius: {w}" for w in wirings) + " | "
              + " | ".join(f"factor: {w}" for w in wirings) + " |",
              "|---|---|---|" + "---|" * (2 * len(wirings))]
    for reg in regions:
        lines.append(f"| {reg} | {int(first.loc[reg, 'neurons']):,} | {readouts[reg]:.0f} | "
                     + " | ".join(f"{rad.loc[reg, w]:.1f}" for w in wirings) + " | "
                     + " | ".join(f"{factors.loc[reg, w]:.2g}" for w in wirings) + " |")
    lines += ["", "## Dominant mode", "",
              "Top mode spread (roughly how many neurons carry the largest eigenvalue) before and after the "
              "per-region factors.", "", "| wiring | single gain | per-region gains |", "|---|---|---|"]
    for w in wirings:
        r = results[results["wiring"] == w]
        lines.append(f"| {w} | {r['spread_single'].mean():,.0f} | {r['spread_regions'].mean():,.0f} |")
    lines.append("")
    return "\n".join(lines)
