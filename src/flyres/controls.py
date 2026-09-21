"""Null models: same neurons, same inputs and readouts, different wiring.

Each control keeps some statistics of the real connectome and scrambles the rest, so if the
real wiring beats a control, the advantage has to come from whatever that control scrambled.

    connectome         the real thing
    degree_preserving  every neuron keeps its in-degree, out-degree and outgoing weights, but who
                       connects to whom is randomized (kills reciprocity, motifs, modules)
    weight_shuffle     same edges, synapse counts permuted across edges
    sign_shuffle       same edges and weights, excitatory/inhibitory identity permuted across
                       neurons (Dale's law still holds, E/I ratio unchanged)
    erdos_renyi        same number of edges placed uniformly at random, same weight distribution

All wirings get rescaled to the same spectral radius later (reservoir.scale_weights).
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp

WIRINGS = ("connectome", "degree_preserving", "weight_shuffle", "sign_shuffle", "erdos_renyi")


def make_wiring(kind: str, W: sp.csr_matrix, sign: np.ndarray, rng: np.random.Generator,
                swaps_per_edge: float = 10.0) -> tuple[sp.csr_matrix, np.ndarray]:
    """Return (synapse-count matrix, neuron signs) for one wiring."""
    if kind == "connectome":
        return W, sign
    if kind == "degree_preserving":
        return degree_preserving_rewire(W, rng, swaps_per_edge), sign
    if kind == "weight_shuffle":
        return shuffle_weights(W, rng), sign
    if kind == "sign_shuffle":
        return W, rng.permutation(sign)
    if kind == "erdos_renyi":
        return erdos_renyi_like(W, rng), sign
    raise ValueError(f"unknown wiring {kind!r}; choose from {WIRINGS}")


def _edges(W: sp.spmatrix):
    coo = sp.coo_matrix(W)
    return coo.col.astype(np.int64), coo.row.astype(np.int64), coo.data  # pre, post, weight


def _from_edges(pre, post, w, n: int) -> sp.csr_matrix:
    return sp.csr_matrix((w, (post, pre)), shape=(n, n), dtype=np.float32)


def _in_sorted(query: np.ndarray, sorted_keys: np.ndarray) -> np.ndarray:
    """Membership test against a sorted array. Queries are sorted first: binary searches in
    increasing order stay cache-friendly (~20x faster than random order at millions of edges)."""
    if len(sorted_keys) == 0:
        return np.zeros(len(query), dtype=bool)
    order = np.argsort(query)
    q = query[order]
    pos = np.minimum(np.searchsorted(sorted_keys, q), len(sorted_keys) - 1)
    hit = np.empty(len(query), dtype=bool)
    hit[order] = sorted_keys[pos] == q
    return hit


def degree_preserving_rewire(W: sp.csr_matrix, rng: np.random.Generator, swaps_per_edge: float = 10.0,
                             max_rounds: int = 1000) -> sp.csr_matrix:
    """Directed double-edge swaps (Maslov & Sneppen 2002), vectorized.

    Pick two edges a->b and c->d and rewire them to a->d and c->b. Every neuron keeps its in- and
    out-degree; each edge keeps its weight and its source, so out-strength and Dale's law survive.
    Swaps that would create a self-loop or a duplicate edge are rejected.

    Each round pairs up all edges at random (disjoint pairs), checks every proposed swap against
    the current edge set and against the other proposals, and applies the valid ones. The sorted
    edge-key array is updated by merging instead of re-sorting (timsort merges two sorted runs in
    linear time), which matters at whole-brain scale (~7M edges).
    """
    pre, post, w = _edges(W)
    n, m = W.shape[0], len(w)
    if m < 2:
        return _from_edges(pre, post, w, n)
    target = int(swaps_per_edge * m)
    pairs = m // 2
    keys = np.sort(pre * n + post)
    done = 0
    for _ in range(max_rounds):
        if done >= target:
            break
        idx = rng.permutation(m)[: 2 * pairs]
        e1, e2 = idx[:pairs], idx[pairs:]
        a, b, c, d = pre[e1], post[e1], pre[e2], post[e2]
        k1, k2 = a * n + d, c * n + b
        ok = (a != d) & (c != b) & (a != c) & (b != d)
        ok &= ~_in_sorted(k1, keys) & ~_in_sorted(k2, keys)
        # two accepted swaps in the same round must not create the same new edge
        new = np.sort(np.concatenate([k1[ok], k2[ok]]))
        dup = np.unique(new[1:][new[1:] == new[:-1]])
        if len(dup):
            okw = np.flatnonzero(ok)
            ok[okw[_in_sorted(k1[okw], dup) | _in_sorted(k2[okw], dup)]] = False
            new = np.sort(np.concatenate([k1[ok], k2[ok]]))
        # update the sorted key set: drop the rewired edges' old keys, merge in the new ones
        drop = np.zeros(m, dtype=bool)
        drop[np.searchsorted(keys, np.sort(np.concatenate([a[ok] * n + b[ok], c[ok] * n + d[ok]])))] = True
        keys = np.sort(np.concatenate([keys[~drop], new]), kind="stable")
        post[e1[ok]] = d[ok]
        post[e2[ok]] = b[ok]
        done += int(ok.sum())
    if done < target:
        warnings.warn(f"degree_preserving: only {done / m:.1f} swaps per edge (target {swaps_per_edge})")
    return _from_edges(pre, post, w, n)


def erdos_renyi_like(W: sp.csr_matrix, rng: np.random.Generator) -> sp.csr_matrix:
    """Same neuron count, edge count and weight distribution; edges placed uniformly (no self-loops)."""
    _, _, w = _edges(W)
    n, m = W.shape[0], len(w)
    if m > n * (n - 1):
        raise ValueError("more edges than possible pairs")
    keys = np.empty(0, dtype=np.int64)
    while len(keys) < m:
        draw = rng.integers(0, n * n, size=int(1.1 * (m - len(keys))) + 16, dtype=np.int64)
        keys = np.unique(np.concatenate([keys, draw[draw // n != draw % n]]))
    if len(keys) > m:
        keys = rng.choice(keys, size=m, replace=False)
    return _from_edges(keys // n, keys % n, rng.permutation(w), n)


def shuffle_weights(W: sp.csr_matrix, rng: np.random.Generator) -> sp.csr_matrix:
    """Same edges, synapse counts permuted across them."""
    pre, post, w = _edges(W)
    return _from_edges(pre, post, rng.permutation(w), W.shape[0])
