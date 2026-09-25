"""Market-free reservoir benchmark: short-term memory capacity (Jaeger 2001).

Feed i.i.d. noise u(t) and train one linear readout per delay k to reconstruct u(t - k).
MC_k = squared correlation on held-out data, total MC = sum over k. It measures how much of its
input history the wiring keeps around, which is cleaner than market results (no noise floor
problem) and tells you *why* one wiring might forecast better than another.
"""
from __future__ import annotations

import numpy as np

from .readout import ridge_multi
from .reservoir import Reservoir


def memory_capacities(reservoir: Reservoir, record_idx, noise_levels=(0.0,), n_steps: int = 3000,
                      max_delay: int = 100, washout: int = 200, rng: np.random.Generator | None = None,
                      alpha: float = 1e-4) -> dict:
    """Memory capacity at several readout-noise levels from one reservoir run.

    Noise level s adds Gaussian noise with std s to every recorded state (states live in [-1, 1], so
    s = 0.001 is 0.1% of a neuron's maximum activity) before the readout is fitted. Without noise, the readout
    standardizes every neuron and can exploit fluctuations of a millionth that no physical system could
    carry; with a little noise, only memory that is actually usable counts.

    Returns {noise level: (total MC, array of MC_k for k = 1..max_delay)}. Needs n_features=1.
    """
    rng = rng if rng is not None else np.random.default_rng()
    u = rng.uniform(-1.0, 1.0, size=(n_steps, 1)).astype(np.float32)  # drawn first: noise-free result unchanged
    S = reservoir.run(u, record_idx=record_idx)
    t = np.arange(washout + max_delay, n_steps)  # rows where every delay is available
    Y = np.stack([u[t - k, 0] for k in range(1, max_delay + 1)], axis=1).astype(np.float64)
    split = len(t) * 2 // 3
    Yt = Y[split:]
    yc = Yt - Yt.mean(axis=0)
    out = {}
    for s in noise_levels:
        X = S[t].astype(np.float64)
        if s > 0:
            X = X + rng.normal(0.0, s, size=X.shape)
        pred = ridge_multi(X[:split], Y[:split], X[split:], alpha)
        pc = pred - pred.mean(axis=0)
        denom = np.sqrt((pc ** 2).sum(axis=0) * (yc ** 2).sum(axis=0))
        r = np.where(denom > 0, (pc * yc).sum(axis=0) / np.where(denom > 0, denom, 1.0), 0.0)
        mc = r ** 2
        out[s] = (float(mc.sum()), mc)
    return out


def memory_capacity(reservoir: Reservoir, record_idx, n_steps: int = 3000, max_delay: int = 100,
                    washout: int = 200, rng: np.random.Generator | None = None, alpha: float = 1e-4):
    """Noise-free memory capacity: (total MC, array of MC_k for k = 1..max_delay). See memory_capacities."""
    return memory_capacities(reservoir, record_idx, (0.0,), n_steps, max_delay, washout, rng, alpha)[0.0]


ACTIVE_STD = 1e-3    # a readout whose state moves less than this under white-noise input carries ~nothing
UNSTABLE_DIFF = 1e-3  # a readout whose final state still depends on inputs from long ago is "unstable"


def readout_stability(reservoir: Reservoir, record_idx=None, n_steps: int = 800, perturb: int = 300,
                      rng: np.random.Generator | None = None, n_tests: int = 1) -> dict:
    """Are the readout neurons alive, and do they forget where they started? (the echo state property)

    A test = two runs with identical white-noise inputs except for the first `perturb` steps.
      echo_gap          largest state difference at the end (~0 = the distant past washed out)
      unstable_readouts share of readouts whose end state still differs: they latched onto the distant
                        past or went chaotic (typically a gain well above 1), so the reservoir is invalid
      active_readouts   share of readouts that actually move after the first `perturb` steps; the rest
                        sit at a constant and give the readout nothing to work with

    With `n_tests` > 1, several independent tests run in one batched pass and the worst one counts
    (unstable_readouts, echo_gap = max over tests; active_readouts = mean). Near the edge a reservoir
    can have several stable states and only fall into one of them for some inputs, so one test passes
    by luck surprisingly often: on the 3,000-neuron circuit, one wiring latched in 2 of 6 tests at a
    gain where the other 4 found nothing. n_tests=1 reproduces the single-test numbers exactly.
    """
    rng = rng if rng is not None else np.random.default_rng()
    streams = []
    for _ in range(n_tests):  # same draw order as a single test, repeated
        u1 = rng.uniform(-1.0, 1.0, size=(n_steps, 1)).astype(np.float32)
        u2 = u1.copy()
        u2[:perturb] = rng.uniform(-1.0, 1.0, size=(perturb, 1))
        streams += [u1, u2]
    S = reservoir.run(np.stack(streams), record_idx=record_idx)  # (2 * n_tests, T, n_recorded)
    base, other = S[0::2], S[1::2]
    diff = np.abs(base[:, -1] - other[:, -1])  # (n_tests, n_recorded)
    return {"echo_gap": float(diff.max()),
            "unstable_readouts": float((diff > UNSTABLE_DIFF).mean(axis=1).max()),
            "active_readouts": float((base[:, perturb:].std(axis=1) >= ACTIVE_STD).mean())}


def echo_state_gap(reservoir: Reservoir, record_idx=None, n_steps: int = 800, perturb: int = 300,
                   rng: np.random.Generator | None = None) -> float:  # single test
    """Largest end-state difference between two runs whose inputs differ only in the distant past.
    ~0 means the reservoir forgets its starting point, as it should (see readout_stability)."""
    return readout_stability(reservoir, record_idx, n_steps, perturb, rng)["echo_gap"]
