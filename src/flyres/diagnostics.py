"""Why a wiring behaves the way it does as a reservoir.

The recurrent matrix is rescaled so its largest eigenvalue hits a target. If that eigenvalue belongs
to a small, dense cluster of neurons (a "hot spot"), the rescaling is set by that cluster alone and
can turn the rest of the network down to near silence. These helpers find where the dominant mode
lives and follow the chain of hot spots when you remove them one by one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def top_mode(W: sp.spmatrix, seed: int = 0, mass: float = 0.9):
    """Dominant eigenvalue of W and where its eigenvector lives.

    Returns (|lambda|, spread, core):
      spread  participation ratio 1 / sum(p_i^2) with p_i = |v_i|^2 / ||v||^2: roughly how many neurons
              carry the mode (a few hundred = localized hot spot; tens of thousands = spread out)
      core    indices of the smallest set of neurons holding `mass` of the mode
    """
    n = W.shape[0]
    if W.nnz == 0:
        return 0.0, float(n), np.arange(0)
    if n <= 400:
        vals, vecs = np.linalg.eig(W.toarray().astype(np.float64))
        i = int(np.argmax(np.abs(vals)))
        lam, v = vals[i], vecs[:, i]
    else:
        A = sp.csr_matrix(W, dtype=np.float64)
        v0 = np.random.default_rng(seed).standard_normal(n)
        try:
            vals, vecs = spla.eigs(A, k=1, which="LM", tol=1e-5, maxiter=max(5000, 10 * n), v0=v0)
        except spla.ArpackNoConvergence as e:
            if not len(e.eigenvalues):
                return float("nan"), float("nan"), np.arange(0)
            vals, vecs = e.eigenvalues, e.eigenvectors
        lam, v = vals[0], vecs[:, 0]
    p = np.abs(v) ** 2
    p /= p.sum()
    order = np.argsort(-p)
    k = int(np.searchsorted(np.cumsum(p[order]), mass)) + 1
    return float(abs(lam)), float(1.0 / (p ** 2).sum()), np.sort(order[:k])


def neuron_label(neurons: pd.DataFrame, idx) -> dict:
    """Most common annotation among the given neurons (cell class if known, else superclass)."""
    sel = neurons.iloc[np.asarray(idx)]
    cls = sel["class"] if "class" in sel.columns else pd.Series([None] * len(sel), index=sel.index)
    return cls.fillna(sel["superclass"]).fillna("unknown").value_counts().head(3).to_dict()


def hot_spot_cascade(W: sp.spmatrix, neurons: pd.DataFrame, steps: int = 8, spread_stop: float = 2000.0,
                     seed: int = 0) -> pd.DataFrame:
    """Find the hot spot that sets the dominant eigenvalue, remove it, repeat.

    Each row: how many neurons were removed before this step, the largest |eigenvalue| of what's left,
    how spread out its mode is, the size of the core holding 90% of it, and what kind of neurons it is
    made of. Stops early once the dominant mode is spread over more than `spread_stop` neurons, i.e.
    there is no hot spot left.
    """
    W = sp.csr_matrix(W, dtype=np.float64)
    keep = np.ones(W.shape[0], dtype=bool)
    rows = []
    for step in range(steps):
        idx = np.flatnonzero(keep)
        lam, spread, core = top_mode(W[idx][:, idx], seed=seed)
        core = idx[core]
        rows.append({"step": step + 1, "removed_before": int((~keep).sum()), "top_abs_eigenvalue": lam,
                     "spread": spread, "core_size": len(core), "made_of": neuron_label(neurons, core)})
        if not np.isfinite(spread) or spread > spread_stop or len(core) == 0:
            break
        keep[core] = False
    return pd.DataFrame(rows)
