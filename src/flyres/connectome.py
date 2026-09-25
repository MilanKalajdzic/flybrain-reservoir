"""Download, parse and cache a fly connectome: the male CNS v1.0 (the main one) or FlyWire (a replication).

Male CNS: FlyEM (HHMI Janelia), University of Cambridge, MRC LMB and Google Research,
https://male-cns.janelia.org, licensed CC-BY 4.0. Raw files (the "flat connectome" export):
    body-annotations-...feather        one row per body: bodyId, superclass, class, type, ...
    body-neurotransmitters-...feather  one row per body: body, consensus_nt, predicted_nt, ...
    connectome-weights-...feather      one row per connected body pair: body_pre, body_post, weight
A "neuron" is a body with a superclass annotation (166,700 of them). Everything else in the
weights table is an unproofread fragment and gets dropped.

FlyWire: the adult female brain (FAFB), public release 783, 139,255 neurons, no nerve cord. FlyWire
Consortium, CC-BY 4.0 (Dorkenwald et al. 2024, Schlegel et al. 2024). Two files, pinned to fixed commits:
    neuron annotations (flyconnectome/flywire_annotations, Supplemental file 1): root_id, super_class,
        cell_class, cell_type, top_nt (predicted transmitter), known_nt (from the literature), soma_x/y/z
    connectivity (philshiu/Drosophila_brain_model, Connectivity_783.parquet, the table behind Shiu et al.
        2024's brain model): one row per connected neuron pair with its synapse count
Its annotations use their own words ("optic", "central", "ascending"); activity.neuron_group maps both
vocabularies onto the same groups, so everything downstream runs unchanged.

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
import pyarrow.parquet as pq
import scipy.sparse as sp

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",  # ~13 MB
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",  # ~42 MB
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",  # ~1.1 GB
}
ANNOTATION_COLUMNS = ["bodyId", "superclass", "class", "subclass", "type", "instance", "somaSide", "status"]
NT_COLUMNS = ["body", "consensus_nt", "celltype_predicted_nt", "predicted_nt"]

_GH = "https://raw.githubusercontent.com/"
FLYWIRE_FILES = {  # key: (local file name, url pinned to a commit so the data can't change underneath)
    "annotations": ("flywire_783_neuron_annotations.tsv", _GH + "flyconnectome/flywire_annotations/"
                    "8587524c1748ce5ef2080822a2fc890fc03bf597/supplemental_files/"
                    "Supplemental_file1_neuron_annotations.tsv"),  # ~32 MB
    "connectivity": ("flywire_783_connectivity.parquet", _GH + "philshiu/Drosophila_brain_model/"
                     "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/Connectivity_783.parquet"),  # ~100 MB
}
# FlyWire annotation columns -> the male CNS names used everywhere else
FLYWIRE_COLUMNS = {"root_id": "bodyId", "super_class": "superclass", "cell_class": "class",
                   "cell_sub_class": "subclass", "cell_type": "type", "side": "somaSide", "flow": "flow"}
SOURCES = ("malecns", "flywire")

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


def known_nt(value) -> str | None:
    """First classical transmitter named in FlyWire's `known_nt` field ("histamine; acetylcholine, histamine"
    -> histamine), ignoring "-negative" results and neuropeptides. None if it names none."""
    if not isinstance(value, str):
        return None
    for token in value.replace(";", ",").split(","):
        token = token.strip().lower()
        if token and not token.endswith("negative"):
            nt = canonical_nt(token)
            if nt is not None and token.startswith(nt[:3]):
                return nt
    return None


def signs_from_nt(nt_labels, inhibitory=INHIBITORY) -> np.ndarray:
    """+1 (excitatory) / -1 (inhibitory) per neuron. Dale's law: one sign per presynaptic neuron."""
    inhib = {s.lower() for s in inhibitory}
    return np.where(pd.Series(list(nt_labels)).isin(inhib).to_numpy(), -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- download

def raw_files(source: str = "malecns") -> dict[str, tuple[str, str]]:
    """{key: (local file name, url)} for a connectome source."""
    if source == "malecns":
        return {k: (name, BASE_URL + name) for k, name in FILES.items()}
    if source == "flywire":
        return dict(FLYWIRE_FILES)
    raise ValueError(f"unknown connectome source {source!r}; choose from {SOURCES}")


def download(raw_dir: str | Path = "data/raw", which=None, force: bool = False,
             source: str = "malecns") -> dict[str, Path]:
    """Download the raw files (resumes interrupted downloads). Returns {key: path}."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = raw_files(source)
    paths = {}
    for key in which or files:
        name, url = files[key]
        dest = raw_dir / name
        paths[key] = dest
        if dest.exists() and not force:
            print(f"already have {dest.name}")
            continue
        print(f"downloading {dest.name}")
        _download_file(url, dest)
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


def load_flywire_neurons(raw_dir: str | Path) -> pd.DataFrame:
    """FlyWire neuron table with the male CNS column names, sorted by bodyId (= FlyWire root_id).
    Transmitter: the literature's (`known_nt`) where it names one, else the prediction (`top_nt`). That
    fixes the classifier's known misses, e.g. Kenyon cells predicted dopaminergic (they're cholinergic)
    and photoreceptors, which release histamine, a transmitter the classifier doesn't include."""
    path = Path(raw_dir) / FLYWIRE_FILES["annotations"][0]
    names = pd.read_csv(path, sep="\t", nrows=0).columns
    missing = [c for c in ("root_id", "super_class", "top_nt") if c not in names]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}; it has {list(names)}")
    cols = [c for c in (*FLYWIRE_COLUMNS, "top_nt", "known_nt") if c in names]
    ann = pd.read_csv(path, sep="\t", usecols=cols, dtype={"root_id": np.int64}, low_memory=False)
    ann = ann[ann["super_class"].notna()].drop_duplicates("root_id")
    known = ann["known_nt"].map(known_nt) if "known_nt" in ann.columns else pd.Series(None, index=ann.index)
    predicted = ann["top_nt"].map(canonical_nt)
    nt = known.where(known.notna(), predicted)
    neurons = ann.rename(columns=FLYWIRE_COLUMNS).drop(columns=["top_nt", "known_nt"], errors="ignore")
    neurons["nt"] = nt.fillna("unknown").to_numpy()
    return neurons.sort_values("bodyId").reset_index(drop=True)


def load_flywire_edges(path: str | Path, body_ids, min_weight: int = 1, drop_autapses: bool = True):
    """(pre, post, weight) from FlyWire's pair table, as row indices into the sorted `body_ids`."""
    body_ids = np.asarray(body_ids, dtype=np.int64)
    n = len(body_ids)
    names = pq.read_schema(path).names
    cols = ("Presynaptic_ID", "Postsynaptic_ID", "Connectivity")
    missing = [c for c in cols if c not in names]
    if missing:
        raise ValueError(f"{Path(path).name} is missing columns {missing}; it has {names}")
    t = pd.read_parquet(path, columns=list(cols))
    pre = t["Presynaptic_ID"].to_numpy(np.int64)
    post = t["Postsynaptic_ID"].to_numpy(np.int64)
    w = t["Connectivity"].to_numpy()
    ip = np.minimum(np.searchsorted(body_ids, pre), n - 1)
    iq = np.minimum(np.searchsorted(body_ids, post), n - 1)
    keep = (w >= min_weight) & (body_ids[ip] == pre) & (body_ids[iq] == post)
    if drop_autapses:
        keep &= ip != iq
    return ip[keep].astype(np.int32), iq[keep].astype(np.int32), w[keep].astype(np.float32)


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


def cache_stem(min_weight: int, drop_autapses: bool = True, source: str = "malecns") -> str:
    base = {"malecns": "malecns_v1.0", "flywire": "flywire_783"}[source]
    return f"{base}_w{min_weight}" + ("" if drop_autapses else "_autapses")


def build_cache(raw_dir: str | Path = "data/raw", cache_dir: str | Path = "data/cache", min_weight: int = 5,
                drop_autapses: bool = True, verbose: bool = True, source: str = "malecns") -> Connectome:
    """Raw files -> sparse matrix + neuron table on disk. Takes a few minutes for the full male CNS file."""
    raw_dir = Path(raw_dir)
    missing = [name for name, _ in raw_files(source).values() if not (raw_dir / name).exists()]
    if missing:
        flag = "" if source == "malecns" else f" --source {source}"
        raise FileNotFoundError(f"missing raw files in {raw_dir}: {missing}. "
                                f"Run scripts/download_data.py{flag} first.")
    if source == "flywire":
        neurons = load_flywire_neurons(raw_dir)
        if verbose:
            print(f"neurons: {len(neurons):,}")
        pre, post, w = load_flywire_edges(raw_dir / FLYWIRE_FILES["connectivity"][0], neurons["bodyId"].to_numpy(),
                                          min_weight, drop_autapses)
    else:
        neurons = load_neurons(raw_dir)
        if verbose:
            print(f"neurons: {len(neurons):,}")
        pre, post, w = load_edges(raw_dir / FILES["weights"], neurons["bodyId"].to_numpy(), min_weight,
                                  drop_autapses, verbose)
    n = len(neurons)
    W = sp.csr_matrix((w, (post, pre)), shape=(n, n), dtype=np.float32)
    W.sum_duplicates()
    conn = Connectome(W, with_degree_columns(neurons, W))
    conn.save(cache_dir, cache_stem(min_weight, drop_autapses, source))
    if verbose:
        d = conn.describe()
        print(f"edges (>= {min_weight} synapses): {d['edges']:,}   synapses: {d['synapses']:,}   "
              f"inhibitory neurons: {d['frac_inhibitory_neurons']:.1%}")
        print("transmitters:\n" + conn.neurons["nt"].value_counts().to_string())
    return conn


def load_connectome(raw_dir: str | Path = "data/raw", cache_dir: str | Path = "data/cache", min_weight: int = 5,
                    drop_autapses: bool = True, verbose: bool = True, source: str = "malecns") -> Connectome:
    """Load the cached graph, building it from the raw files on first use."""
    stem = cache_stem(min_weight, drop_autapses, source)
    if (Path(cache_dir) / f"{stem}.npz").exists():
        return Connectome.load(cache_dir, stem)
    return build_cache(raw_dir, cache_dir, min_weight, drop_autapses, verbose, source)
