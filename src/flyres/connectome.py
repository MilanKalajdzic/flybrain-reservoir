"""Download, parse and cache the male CNS v1.0 connectome.

Data: FlyEM (HHMI Janelia), University of Cambridge, MRC LMB and Google Research,
https://male-cns.janelia.org, licensed CC-BY 4.0.

Raw files (the "flat connectome" export):
    body-annotations-...feather        one row per body: bodyId, superclass, class, type, ...
    body-neurotransmitters-...feather  one row per body: body, consensus_nt, predicted_nt, ...
    connectome-weights-...feather      one row per connected body pair: body_pre, body_post, weight

A "neuron" is a body with a superclass annotation (166,700 of them). Everything else in the
weights table is an unproofread fragment and gets dropped.

Convention used everywhere in this package: W[post, pre] = number of synapses from neuron
`pre` onto neuron `post`, so `W @ x` is the input each neuron receives.
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import scipy.sparse as sp

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",  # ~13 MB
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",  # ~42 MB
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",  # ~1.1 GB
}
ANNOTATION_COLUMNS = ["bodyId", "superclass", "class", "subclass", "type", "instance", "somaSide", "status"]
NT_COLUMNS = ["body", "consensus_nt", "celltype_predicted_nt", "predicted_nt"]

# Sign rule from Shiu et al. 2024 (FlyWire whole-brain model): GABA and glutamate are
# inhibitory, everything else excitatory. Histamine is inhibitory too (chloride channels in flies).
INHIBITORY = frozenset({"gaba", "glutamate", "histamine"})

# Label spellings vary between releases (and some configs have typos like "gluatmate"),
# so match on prefixes. Anything unmatched ("unclear", "unknown", NaN) becomes None.
_NT_PREFIXES = [
    ("acetyl", "acetylcholine"), ("ach", "acetylcholine"), ("gaba", "gaba"), ("glu", "glutamate"),
    ("dop", "dopamine"), ("da", "dopamine"), ("ser", "serotonin"), ("5ht", "serotonin"),
    ("5-ht", "serotonin"), ("oct", "octopamine"), ("oa", "octopamine"), ("his", "histamine"),
    ("tyr", "tyramine"),
]


def canonical_nt(value) -> str | None:
    """Map a raw neurotransmitter label to a canonical lowercase name, or None if unknown."""
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    for prefix, name in _NT_PREFIXES:
        if v.startswith(prefix):
            return name
    return None


def signs_from_nt(nt_labels, inhibitory=INHIBITORY) -> np.ndarray:
    """+1 (excitatory) / -1 (inhibitory) per neuron. Dale's law: one sign per presynaptic neuron."""
    inhib = {s.lower() for s in inhibitory}
    return np.where(pd.Series(list(nt_labels)).isin(inhib).to_numpy(), -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- download

def download(raw_dir: str | Path = "data/raw", which=None, force: bool = False) -> dict[str, Path]:
    """Download the raw feather files (resumes interrupted downloads). Returns {key: path}."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key in which or FILES:
        dest = raw_dir / FILES[key]
        paths[key] = dest
        if dest.exists() and not force:
            print(f"already have {dest.name}")
            continue
        print(f"downloading {dest.name}")
        _download_file(BASE_URL + FILES[key], dest)
    return paths


def _download_file(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    try:
        resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and start:  # the .part file is already complete
            part.replace(dest)
            return
        raise
    with resp:
        if start and resp.status != 206:  # server ignored the range request: start over
            start = 0
        total = start + int(resp.headers.get("Content-Length") or 0)
        done, last_print = start, 0.0
        with open(part, "ab" if start else "wb") as f:
            while block := resp.read(chunk):
                f.write(block)
                done += len(block)
                if time.time() - last_print > 1.0:
                    pct = f" ({100 * done / total:.1f}%)" if total else ""
                    print(f"\r  {done / 1e6:,.0f} MB{pct}", end="", flush=True)
                    last_print = time.time()
    print(f"\r  {done / 1e6:,.0f} MB done".ljust(40))
    part.replace(dest)


# --------------------------------------------------------------------------- parsing

def _read_columns(path: Path, wanted: list[str], required: list[str]) -> pd.DataFrame:
    """Read only the columns that exist; fail loudly (with the real schema) if required ones are missing."""
    names = ipc.open_file(pa.memory_map(str(path), "r")).schema.names
    missing = [c for c in required if c not in names]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}; it has {names}")
    return pd.read_feather(path, columns=[c for c in wanted if c in names])


def load_neurons(raw_dir: str | Path) -> pd.DataFrame:
    """Neuron table: annotated bodies sorted by bodyId, plus a canonical `nt` label."""
    raw_dir = Path(raw_dir)
    ann = _read_columns(raw_dir / FILES["annotations"], ANNOTATION_COLUMNS, ["bodyId", "superclass"])
    neurons = ann[ann["superclass"].notna()].copy()
    neurons["bodyId"] = neurons["bodyId"].astype(np.int64)
    neurons = neurons.drop_duplicates("bodyId").sort_values("bodyId").reset_index(drop=True)

    nt = _read_columns(raw_dir / FILES["neurotransmitters"], NT_COLUMNS, ["body"])
    nt["body"] = nt["body"].astype(np.int64)
    nt = nt.drop_duplicates("body").set_index("body").reindex(neurons["bodyId"].to_numpy())

    # Curated cell-type consensus first, then the cell-type prediction, then the neuron's own.
    label = np.full(len(neurons), None, dtype=object)
    for col in ("consensus_nt", "celltype_predicted_nt", "predicted_nt"):
        if col in nt.columns:
            canon = np.array([canonical_nt(v) for v in nt[col].to_numpy()], dtype=object)
            unset = np.array([v is None for v in label])
            label[unset] = canon[unset]
    neurons["nt"] = ["unknown" if v is None else v for v in label]
    return neurons


def load_edges(weights_path: str | Path, body_ids, min_weight: int = 1, drop_autapses: bool = True,
               verbose: bool = True):
    """Stream the weights table batch by batch, keeping neuron->neuron edges only.

    Returns (pre, post, weight) with pre/post as row indices into the sorted `body_ids`.
    Streaming keeps memory low: the file has ~100M+ rows, most of them fragments.
    """
    body_ids = np.asarray(body_ids, dtype=np.int64)
    n = len(body_ids)
    reader = ipc.open_file(pa.memory_map(str(weights_path), "r"))
    names = reader.schema.names
    for col in ("body_pre", "body_post", "weight"):
        if col not in names:
            raise ValueError(f"weights file is missing {col!r}; it has {names}")
    c_pre, c_post, c_w = names.index("body_pre"), names.index("body_post"), names.index("weight")

    pre_l, post_l, w_l = [], [], []
    rows, t0, nb = 0, time.time(), reader.num_record_batches
    for b in range(nb):
        batch = reader.get_batch(b)
        pre = batch.column(c_pre).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        post = batch.column(c_post).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        w = batch.column(c_w).to_numpy(zero_copy_only=False)
        rows += len(w)
        ip = np.minimum(np.searchsorted(body_ids, pre), n - 1)
        iq = np.minimum(np.searchsorted(body_ids, post), n - 1)
        keep = (w >= min_weight) & (body_ids[ip] == pre) & (body_ids[iq] == post)
        if drop_autapses:
            keep &= ip != iq
        pre_l.append(ip[keep].astype(np.int32))
        post_l.append(iq[keep].astype(np.int32))
        w_l.append(w[keep].astype(np.float32))
        if verbose and (b % 250 == 0 or b == nb - 1):
            print(f"\r  weights: batch {b + 1}/{nb}, {rows:,} rows read, {time.time() - t0:.0f}s",
                  end="", flush=True)
    if verbose:
        print()
    return np.concatenate(pre_l), np.concatenate(post_l), np.concatenate(w_l)


# --------------------------------------------------------------------------- cache

@dataclass
class Connectome:
    """Whole-CNS graph. W[post, pre] = synapse count (unsigned, float32). Row i of `neurons` = node i."""

    W: sp.csr_matrix
    neurons: pd.DataFrame

    @property
    def n(self) -> int:
        return self.W.shape[0]

    @property
    def n_edges(self) -> int:
        return self.W.nnz

    def signs(self, inhibitory=INHIBITORY) -> np.ndarray:
        return signs_from_nt(self.neurons["nt"], inhibitory)

    def describe(self) -> dict:
        sign = self.signs()
        return {
            "neurons": self.n,
            "edges": self.n_edges,
            "synapses": int(self.W.sum()),
            "frac_inhibitory_neurons": float((sign < 0).mean()),
        }

    def save(self, cache_dir: str | Path, stem: str) -> None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        sp.save_npz(cache_dir / f"{stem}.npz", self.W)
        self.neurons.to_parquet(cache_dir / f"{stem}_neurons.parquet", index=False)

    @classmethod
    def load(cls, cache_dir: str | Path, stem: str) -> "Connectome":
        cache_dir = Path(cache_dir)
        W = sp.load_npz(cache_dir / f"{stem}.npz").tocsr()
        neurons = pd.read_parquet(cache_dir / f"{stem}_neurons.parquet")
        return cls(W, neurons)


def with_degree_columns(neurons: pd.DataFrame, W: sp.csr_matrix) -> pd.DataFrame:
    neurons = neurons.copy()
    neurons["in_degree"] = np.diff(W.indptr)
    neurons["out_degree"] = np.bincount(W.indices, minlength=W.shape[0])
    neurons["in_synapses"] = np.asarray(W.sum(axis=1)).ravel()
    neurons["out_synapses"] = np.asarray(W.sum(axis=0)).ravel()
    return neurons


def cache_stem(min_weight: int, drop_autapses: bool = True) -> str:
    return f"malecns_v1.0_w{min_weight}" + ("" if drop_autapses else "_autapses")


def build_cache(raw_dir: str | Path = "data/raw", cache_dir: str | Path = "data/cache", min_weight: int = 5,
                drop_autapses: bool = True, verbose: bool = True) -> Connectome:
    """Raw feather files -> sparse matrix + neuron table on disk. Takes a few minutes for the full file."""
    raw_dir = Path(raw_dir)
    missing = [name for name in FILES.values() if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing raw files in {raw_dir}: {missing}. Run scripts/download_data.py first.")
    neurons = load_neurons(raw_dir)
    if verbose:
        print(f"neurons: {len(neurons):,}")
    pre, post, w = load_edges(raw_dir / FILES["weights"], neurons["bodyId"].to_numpy(), min_weight,
                              drop_autapses, verbose)
    n = len(neurons)
    W = sp.csr_matrix((w, (post, pre)), shape=(n, n), dtype=np.float32)
    W.sum_duplicates()
    conn = Connectome(W, with_degree_columns(neurons, W))
    conn.save(cache_dir, cache_stem(min_weight, drop_autapses))
    if verbose:
        d = conn.describe()
        print(f"edges (>= {min_weight} synapses): {d['edges']:,}   synapses: {d['synapses']:,}   "
              f"inhibitory neurons: {d['frac_inhibitory_neurons']:.1%}")
        print("transmitters:\n" + conn.neurons["nt"].value_counts().to_string())
    return conn


def load_connectome(raw_dir: str | Path = "data/raw", cache_dir: str | Path = "data/cache", min_weight: int = 5,
                    drop_autapses: bool = True, verbose: bool = True) -> Connectome:
    """Load the cached graph, building it from the raw files on first use."""
    stem = cache_stem(min_weight, drop_autapses)
    if (Path(cache_dir) / f"{stem}.npz").exists():
        return Connectome.load(cache_dir, stem)
    return build_cache(raw_dir, cache_dir, min_weight, drop_autapses, verbose)
