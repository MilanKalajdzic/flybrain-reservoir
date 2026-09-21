"""What the reservoir neurons do over time: the data behind the "brain lighting up" figures.

The reservoir is re-run with the experiment's own seeding (connectome wiring, same input weights),
every neuron is recorded, and each neuron's activity is expressed relative to its own normal level
(z-score over the whole run). Neurons are grouped into readable pathway stages so you can see which
parts of the circuit get stirred up when markets crash. Descriptive only: the reservoir reacts to
its inputs (returns and volatility), it doesn't see the future.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .connectome import FILES

INPUT_GROUP = "Input neurons (market enters here)"
GROUP_ORDER = [
    INPUT_GROUP,
    "Smell sensors", "Taste sensors", "Temperature & humidity sensors", "Touch & body-position sensors",
    "Other sensors", "Antennal lobe (smell processing)", "Mushroom body (learning)",
    "Central complex (navigation)", "Other central brain", "Descending (brain to body)",
    "Ascending (body to brain)", "Motor neurons", "Nerve cord", "Visual", "Other",
]

# Crashes worth labelling (peak-to-trough windows of the big drawdowns in the sample).
EPISODES = {
    "2008": ("2008-09-01", "2009-03-09"),
    "COVID": ("2020-02-19", "2020-03-23"),
    "2022": ("2022-01-03", "2022-10-12"),
}


def neuron_group(superclass, cls) -> str:
    """Readable pathway stage from the connectome's superclass/class annotations."""
    s = superclass.lower() if isinstance(superclass, str) else ""
    c = cls.lower() if isinstance(cls, str) else ""
    if "sensory" in s:
        if c == "olfactory":
            return "Smell sensors"
        if c == "gustatory":
            return "Taste sensors"
        if c in ("thermosensory", "hygrosensory"):
            return "Temperature & humidity sensors"
        if c.startswith("mechanosensory"):
            return "Touch & body-position sensors"
        return "Other sensors"
    if c in ("alpn", "alln", "alin", "alon"):
        return "Antennal lobe (smell processing)"
    if c in ("mbon", "dan", "kenyon_cell", "mbin", "apl", "dpm"):
        return "Mushroom body (learning)"
    if c == "cx":
        return "Central complex (navigation)"
    if s == "descending_neuron":
        return "Descending (brain to body)"
    if s == "ascending_neuron":
        return "Ascending (body to brain)"
    if "motor" in s:
        return "Motor neurons"
    if s.startswith(("visual", "ol_")):
        return "Visual"
    if s.startswith("vnc"):
        return "Nerve cord"
    if s.startswith("cb"):
        return "Other central brain"
    return "Other"


def neuron_groups(neurons: pd.DataFrame, input_idx, min_size: int = 8) -> pd.Series:
    """Group label per neuron. Input neurons get their own group; tiny groups fold into a catch-all."""
    cls = neurons["class"] if "class" in neurons.columns else pd.Series([None] * len(neurons))
    groups = pd.Series([neuron_group(s, c) for s, c in zip(neurons["superclass"], cls)], index=neurons.index)
    groups.iloc[np.asarray(input_idx)] = INPUT_GROUP
    counts = groups.value_counts()
    small = [g for g in counts.index if counts[g] < min_size and g != INPUT_GROUP]
    fold = {g: ("Other central brain" if g in ("Central complex (navigation)", "Mushroom body (learning)")
                else "Other") for g in small}
    return groups.replace(fold)


def zscore(states: np.ndarray) -> np.ndarray:
    """Each neuron relative to its own normal level. Neurons that never move stay at 0."""
    mean = states.mean(axis=0)
    sd = states.std(axis=0)
    return ((states - mean) / np.where(sd > 1e-6, sd, np.inf)).astype(np.float32)


def weekly(values: np.ndarray, dates: pd.DatetimeIndex, freq: str = "W-FRI") -> pd.DataFrame:
    """Period means of a (T, n) array (weekly by default; "MS" = monthly)."""
    return pd.DataFrame(values, index=pd.DatetimeIndex(dates)).resample(freq).mean().dropna(how="all")


def group_deviation(z: np.ndarray, groups: pd.Series, dates: pd.DatetimeIndex, freq: str = "W-FRI") -> pd.DataFrame:
    """Mean |z| per group and period: how far from normal each pathway stage is (about 0.8 = a normal week)."""
    wk = weekly(np.abs(z), dates, freq)
    out = {g: wk.loc[:, np.flatnonzero((groups == g).to_numpy())].mean(axis=1)
           for g in GROUP_ORDER if (groups == g).any()}
    return pd.DataFrame(out)


def soma_positions(raw_dir: str | Path, body_ids) -> np.ndarray:
    """(n, 3) soma coordinates (8 nm voxels) for the given neurons, NaN where unknown.

    Sensory neurons have their cell bodies outside the imaged CNS (antennae, legs), so they have none.
    Falls back to `tosomaLocation`, a point on the neurite pointing toward the soma.
    """
    path = Path(raw_dir) / FILES["annotations"]
    ann = pd.read_feather(path, columns=["bodyId", "somaLocation", "tosomaLocation"])
    ann["bodyId"] = ann["bodyId"].astype(np.int64)

    def ok(v):
        return v is not None and not isinstance(v, float) and len(v) == 3

    lookup = {b: (s if ok(s) else t) for b, s, t in zip(ann["bodyId"], ann["somaLocation"], ann["tosomaLocation"])
              if ok(s) or ok(t)}
    return np.array([lookup.get(int(b), (np.nan,) * 3) for b in body_ids], dtype=np.float64)


def all_soma_positions(raw_dir: str | Path) -> np.ndarray:
    """Soma coordinates of every neuron in the CNS (for drawing the brain's outline)."""
    path = Path(raw_dir) / FILES["annotations"]
    ann = pd.read_feather(path, columns=["superclass", "somaLocation"])
    locs = [v for s, v in zip(ann["superclass"], ann["somaLocation"])
            if isinstance(s, str) and v is not None and not isinstance(v, float) and len(v) == 3]
    return np.array(locs, dtype=np.float64)
