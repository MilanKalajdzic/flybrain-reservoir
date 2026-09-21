"""Fake data for tests and offline demos: a connectome-shaped graph and a GARCH-ish price series.

Nothing here is used for real results; it lets the whole pipeline run without the 1.1 GB download
or an internet connection, and gives tests a case where the right answer is known.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import scipy.sparse as sp

from .connectome import FILES, Connectome, with_degree_columns


def synthetic_connectome(n: int = 2000, mean_degree: float = 25.0, frac_inhibitory: float = 0.25,
                         frac_sensory: float = 0.08, frac_descending: float = 0.03, reciprocity: float = 0.15,
                         seed: int = 0) -> Connectome:
    """Heavy-tailed degrees, extra reciprocal edges, sensory neurons that mostly send, lognormal synapse counts."""
    rng = np.random.default_rng(seed)
    kind = np.array(["cb_intrinsic"] * n, dtype=object)
    order = rng.permutation(n)
    n_s, n_d = int(frac_sensory * n), int(frac_descending * n)
    kind[order[:n_s]] = "sensory"
    kind[order[n_s:n_s + n_d]] = "descending_neuron"

    out_prop = rng.lognormal(0.0, 1.0, n)
    in_prop = rng.lognormal(0.0, 1.0, n)
    in_prop[kind == "sensory"] *= 0.02  # sensory neurons get almost no input from the brain
    m = int(n * mean_degree)
    pre = rng.choice(n, m, p=out_prop / out_prop.sum())
    post = rng.choice(n, m, p=in_prop / in_prop.sum())
    back = rng.random(m) < reciprocity  # real connectomes are far more reciprocal than random graphs
    pre, post = np.concatenate([pre, post[back]]), np.concatenate([post, pre[back]])
    keys = np.unique(pre[pre != post].astype(np.int64) * n + post[pre != post])
    pre, post = keys // n, keys % n
    w = np.maximum(1.0, np.round(rng.lognormal(1.2, 1.0, len(keys)))).astype(np.float32)

    inhib = (rng.random(n) < frac_inhibitory) & (kind != "sensory")
    nt = np.where(inhib, np.where(rng.random(n) < 0.5, "gaba", "glutamate"), "acetylcholine")
    neurons = pd.DataFrame({
        "bodyId": np.sort(rng.choice(10**7, n, replace=False)).astype(np.int64),
        "superclass": kind,
        "class": np.where(kind == "sensory", "olfactory", None),
        "type": [f"syn{i}" for i in range(n)],
        "nt": nt,
    })
    W = sp.csr_matrix((w, (post, pre)), shape=(n, n), dtype=np.float32)
    return Connectome(W, with_degree_columns(neurons, W))


def write_mock_raw_files(conn: Connectome, raw_dir: str | Path, n_fragments: int = 300, batch_rows: int = 1000,
                         seed: int = 0) -> None:
    """Write feather files with the real male CNS schema (incl. fragments, 'unclear' labels, duplicates,
    autapses, many record batches) so the loader can be tested end to end."""
    rng = np.random.default_rng(seed)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    body = conn.neurons["bodyId"].to_numpy()
    frag = np.setdiff1d(rng.choice(10**8, n_fragments * 2, replace=False) + 10**7, body)[:n_fragments]

    ann = pd.DataFrame({
        "bodyId": np.concatenate([body, frag]).astype(np.uint64),
        "status": ["Traced"] * len(body) + ["Orphan"] * len(frag),
        "superclass": list(conn.neurons["superclass"]) + [None] * len(frag),
        "class": list(conn.neurons["class"]) + [None] * len(frag),
        "type": list(conn.neurons["type"]) + [None] * len(frag),
    })
    feather.write_feather(ann, raw_dir / FILES["annotations"])

    labels = conn.neurons["nt"].to_numpy().astype(object)
    consensus = labels.copy()
    consensus[rng.random(len(labels)) < 0.3] = "unclear"  # forces the fallback columns to be used
    nt = pd.DataFrame({
        "body": body.astype(np.uint64),
        "predicted_nt": labels,
        "predicted_nt_confidence": rng.random(len(labels)),
        "celltype_predicted_nt": np.where(rng.random(len(labels)) < 0.5, labels, None),
        "consensus_nt": consensus,
    })
    nt = pd.concat([nt, nt.iloc[:5]], ignore_index=True)  # duplicated rows exist in real exports too
    feather.write_feather(nt, raw_dir / FILES["neurotransmitters"])

    coo = conn.W.tocoo()
    pre, post, w = body[coo.col], body[coo.row], coo.data.astype(np.int64)
    k = 200  # extra junk: fragment edges and autapses, which the loader must drop
    pre = np.concatenate([pre, rng.choice(frag, k), body[:k]])
    post = np.concatenate([post, rng.choice(body, k), body[:k]])
    w = np.concatenate([w, rng.integers(1, 20, k), rng.integers(1, 20, k)])
    edges = pa.table({"body_pre": pre.astype(np.uint64), "body_post": post.astype(np.uint64),
                      "weight": w.astype(np.int64)})
    feather.write_feather(edges, raw_dir / FILES["weights"], chunksize=batch_rows)


def synthetic_prices(n_days: int = 5000, predictability: float = 0.0, seed: int = 0,
                     start: str = "2005-01-03") -> pd.Series:
    """GARCH(1,1) returns with fat tails. With predictability > 0, tomorrow's mean depends
    nonlinearly on the last two shocks, so a model with memory can find it."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_t(5, n_days) / np.sqrt(5 / 3)  # unit variance
    omega, alpha, beta, mu = 2e-6, 0.08, 0.90, 3e-4
    var = omega / (1 - alpha - beta)
    r = np.zeros(n_days)
    for t in range(n_days):
        sigma = np.sqrt(var)
        signal = predictability * sigma * np.tanh(eps[t - 1] + eps[t - 2]) if t >= 2 else 0.0
        r[t] = mu + signal + sigma * eps[t]
        var = omega + alpha * (r[t] - mu) ** 2 + beta * var
    dates = pd.bdate_range(start, periods=n_days)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=dates, name="SYNTH")
