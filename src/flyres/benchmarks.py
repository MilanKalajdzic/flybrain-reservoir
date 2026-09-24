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


def memory_capacity(reservoir: Reservoir, record_idx, n_steps: int = 3000, max_delay: int = 100,
                    washout: int = 200, rng: np.random.Generator | None = None, alpha: float = 1e-4):
    """Returns (total MC, array of MC_k for k = 1..max_delay). Needs a reservoir built with n_features=1."""
    rng = rng if rng is not None else np.random.default_rng()
    u = rng.uniform(-1.0, 1.0, size=(n_steps, 1)).astype(np.float32)
    S = reservoir.run(u, record_idx=record_idx)
    t = np.arange(washout + max_delay, n_steps)  # rows where every delay is available
    X = S[t]
    Y = np.stack([u[t - k, 0] for k in range(1, max_delay + 1)], axis=1).astype(np.float64)
    split = len(t) * 2 // 3
    pred = ridge_multi(X[:split], Y[:split], X[split:], alpha)
    Yt = Y[split:]
    pc = pred - pred.mean(axis=0)
    yc = Yt - Yt.mean(axis=0)
    denom = np.sqrt((pc ** 2).sum(axis=0) * (yc ** 2).sum(axis=0))
    r = np.where(denom > 0, (pc * yc).sum(axis=0) / np.where(denom > 0, denom, 1.0), 0.0)
    mc = r ** 2
    return float(mc.sum()), mc


ACTIVE_STD = 1e-3    # a readout whose state moves less than this under white-noise input carries ~nothing
UNSTABLE_DIFF = 1e-3  # a readout whose final state still depends on inputs from long ago is "unstable"


def readout_stability(reservoir: Reservoir, record_idx=None, n_steps: int = 800, perturb: int = 300,
                      rng: np.random.Generator | None = None) -> dict:
    """Are the readout neurons alive, and do they forget where they started? (the echo state property)

    Two runs get identical white-noise inputs except for the first `perturb` steps.
      echo_gap          largest state difference at the end (~0 = the distant past washed out)
      unstable_readouts share of readouts whose end state still differs: they latched onto the distant
                        past or went chaotic (typically a gain well above 1), so the reservoir is invalid
      active_readouts   share of readouts that actually move after the first `perturb` steps; the rest
                        sit at a constant and give the readout nothing to work with
    """
    rng = rng if rng is not None else np.random.default_rng()
    u1 = rng.uniform(-1.0, 1.0, size=(n_steps, 1)).astype(np.float32)
    u2 = u1.copy()
    u2[:perturb] = rng.uniform(-1.0, 1.0, size=(perturb, 1))
    s1, s2 = reservoir.run(u1, record_idx=record_idx), reservoir.run(u2, record_idx=record_idx)
    diff = np.abs(s1[-1] - s2[-1])
    return {"echo_gap": float(diff.max()),
            "unstable_readouts": float((diff > UNSTABLE_DIFF).mean()),
            "active_readouts": float((s1[perturb:].std(axis=0) >= ACTIVE_STD).mean())}


def echo_state_gap(reservoir: Reservoir, record_idx=None, n_steps: int = 800, perturb: int = 300,
                   rng: np.random.Generator | None = None) -> float:
    """Largest end-state difference between two runs whose inputs differ only in the distant past.
    ~0 means the reservoir forgets its starting point, as it should (see readout_stability)."""
    return readout_stability(reservoir, record_idx, n_steps, perturb, rng)["echo_gap"]
