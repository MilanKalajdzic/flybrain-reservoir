"""Pick the piece of the connectome that becomes the reservoir.

The whole CNS works too (method="all"), but for development a few thousand neurons is plenty.
The default "grow" method starts from sensory neurons (where the market signal enters) and
greedily adds the neurons most strongly connected to the current set, so you get a connected
circuit instead of a random scatter of cells that barely talk to each other.

Everything downstream only needs the size, so going from 3k to 166k neurons is a config change.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .connectome import INHIBITORY, Connectome, signs_from_nt

METHODS = ("grow", "top_degree", "all")


@dataclass
class Subgraph:
    """A reservoir-sized piece of the connectome. Input neurons are always the first rows."""

    W: sp.csr_matrix       # synapse counts, W[post, pre]
    sign: np.ndarray       # +1 / -1 per neuron (sign of its outgoing synapses)
    neurons: pd.DataFrame  # metadata; column `global_idx` = row in the full connectome
    input_idx: np.ndarray  # neurons that receive the market inputs

    @property
    def n(self) -> int:
        return self.W.shape[0]

    @property
    def readout_pool(self) -> np.ndarray:
        """Neurons the readout may listen to: everything except the input neurons."""
        return np.setdiff1d(np.arange(self.n), self.input_idx)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        W = self.W.tocsr()
        np.savez_compressed(path.with_name(path.name + ".npz"), data=W.data, indices=W.indices,
                            indptr=W.indptr, shape=np.array(W.shape), sign=self.sign, input_idx=self.input_idx)
        self.neurons.to_parquet(path.with_name(path.name + "_neurons.parquet"), index=False)

    @classmethod
    def load(cls, path: str | Path) -> "Subgraph":
        path = Path(path)
        z = np.load(path.with_name(path.name + ".npz"))
        W = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        neurons = pd.read_parquet(path.with_name(path.name + "_neurons.parquet"))
        return cls(W, z["sign"], neurons, z["input_idx"])

    @staticmethod
    def exists(path: str | Path) -> bool:
        path = Path(path)
        return path.with_name(path.name + ".npz").exists()


def match(neurons: pd.DataFrame, filt: dict | None) -> np.ndarray:
    """Case-insensitive substring filter on metadata columns.

    {"superclass": "sensory"} -> any superclass containing "sensory".
    {"class": ["olfactory", "gustatory"]} -> either. Several keys are ANDed.
    """
    mask = np.ones(len(neurons), dtype=bool)
    for col, values in (filt or {}).items():
        if col not in neurons.columns:
            raise KeyError(f"no column {col!r} in neuron table; columns: {list(neurons.columns)}")
        values = [values] if isinstance(values, str) else list(values)
        s = neurons[col].astype("string").str.lower().fillna("")
        hit = np.zeros(len(neurons), dtype=bool)
        for v in values:
            hit |= s.str.contains(str(v).lower(), regex=False).to_numpy(dtype=bool)
        mask &= hit
    return mask


def select_subgraph(conn: Connectome, n_neurons: int = 3000, n_inputs: int = 100, input_filter: dict | None = None,
                    method: str = "grow", direction: str = "both", inhibitory=INHIBITORY,
                    verbose: bool = True) -> Subgraph:
    """Choose neurons + input neurons. Input neurons are the candidates with the most output synapses."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    W = conn.W
    out_syn = np.asarray(W.sum(axis=0)).ravel()
    in_syn = np.asarray(W.sum(axis=1)).ravel()
    is_candidate = match(conn.neurons, input_filter) & (out_syn > 0)
    if not is_candidate.any():
        values = sorted(conn.neurons["superclass"].dropna().astype(str).unique())
        raise ValueError(f"no neurons match input_filter={input_filter}. superclass values: {values}")

    n_neurons = conn.n if method == "all" else min(n_neurons, conn.n)

    if method == "top_degree":
        core = np.argsort(-(out_syn + in_syn), kind="stable")[:n_neurons]
        cands = core[is_candidate[core]]
        if len(cands) == 0:
            raise ValueError("no input candidates among the top-degree neurons; use method='grow'")
        seeds = cands[np.argsort(-out_syn[cands], kind="stable")[:n_inputs]]
        sel = np.concatenate([seeds, np.sort(np.setdiff1d(core, seeds))])
    else:
        cands = np.flatnonzero(is_candidate)
        seeds = cands[np.argsort(-out_syn[cands], kind="stable")[:min(n_inputs, n_neurons)]]
        if method == "grow":
            sel = _grow(W, seeds, n_neurons, direction)
        else:  # all
            sel = np.concatenate([seeds, np.setdiff1d(np.arange(conn.n), seeds)])

    if len(seeds) < n_inputs:
        warnings.warn(f"only {len(seeds)} input neurons available (asked for {n_inputs})")

    neurons = conn.neurons.iloc[sel].reset_index(drop=True)
    neurons.insert(0, "global_idx", sel)
    sub = Subgraph(W[sel][:, sel].tocsr(), signs_from_nt(neurons["nt"], inhibitory), neurons,
                   np.arange(len(seeds)))
    if verbose:
        s = graph_stats(sub.W, sub.sign)
        print(f"subgraph: {s['n']:,} neurons, {s['edges']:,} edges, {len(seeds)} inputs, "
              f"reciprocity {s['reciprocity']:.2f}, largest SCC {s['largest_scc_frac']:.0%}")
    return sub


def _grow(W: sp.csr_matrix, seeds: np.ndarray, n_neurons: int, direction: str = "both",
          n_steps: int = 50) -> np.ndarray:
    """Greedy expansion: repeatedly add the neurons with the strongest connections to the current set.

    direction="downstream" only counts synapses from the set (follows the signal forward);
    "both" also counts synapses onto the set, which pulls in feedback loops (more recurrence).
    """
    if direction not in ("both", "downstream"):
        raise ValueError("direction must be 'both' or 'downstream'")
    n = W.shape[0]
    W_csc = W.tocsc()  # column j = where neuron j sends its synapses
    selected = np.zeros(n, dtype=bool)
    selected[seeds] = True
    order = [np.asarray(seeds)]
    score = np.zeros(n)

    def absorb(idx):
        score[:] += np.asarray(W_csc[:, idx].sum(axis=1)).ravel()  # synapses received from idx
        if direction == "both":
            score[:] += np.asarray(W[idx, :].sum(axis=0)).ravel()  # synapses sent to idx

    absorb(seeds)
    count = len(seeds)
    step = max(1, (n_neurons - count) // n_steps)
    while count < n_neurons:
        cand = np.where(selected, -np.inf, score)
        k = min(step, n_neurons - count)
        top = np.argpartition(-cand, k - 1)[:k]
        top = top[cand[top] > 0]
        if len(top) == 0:
            warnings.warn(f"ran out of connected neurons at {count} (asked for {n_neurons})")
            break
        selected[top] = True
        order.append(np.sort(top))
        count += len(top)
        absorb(top)
    return np.concatenate(order)


def graph_stats(W: sp.spmatrix, sign: np.ndarray | None = None) -> dict:
    """Quick structural summary: size, density, reciprocity, largest strongly connected component."""
    A = sp.csr_matrix(W, copy=True)
    A.data = np.ones_like(A.data)
    A.eliminate_zeros()
    n, m = A.shape[0], A.nnz
    _, labels = connected_components(A, directed=True, connection="strong")
    stats = {
        "n": n,
        "edges": m,
        "density": m / (n * (n - 1)) if n > 1 else 0.0,
        "reciprocity": A.multiply(A.T).nnz / m if m else 0.0,
        "largest_scc_frac": float(np.bincount(labels).max() / n),
    }
    if sign is not None:
        stats["frac_inhibitory"] = float((np.asarray(sign) < 0).mean())
    return stats
