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


# A neuron counts as moving if its activity varies by at least this much (0.1% of its maximum, tanh activity runs from -1 to 1), the
# same bar as the readout noise and "active readouts" in the benchmarks. Below it, z-scoring would blow
# fluctuations of a millionth up to full size and light up neurons that carry nothing usable.
MOVE_THRESHOLD = 1e-3


def zscore(states: np.ndarray, min_std: float = MOVE_THRESHOLD) -> np.ndarray:
    """Each neuron relative to its own normal level. Neurons that barely move (std below `min_std`) stay at 0."""
    mean = states.mean(axis=0)
    sd = states.std(axis=0)
    return ((states - mean) / np.where(sd >= min_std, sd, np.inf)).astype(np.float32)


def weekly(values: np.ndarray, dates: pd.DatetimeIndex, freq: str = "W-FRI") -> pd.DataFrame:
    """Period means of a (T, n) array (weekly by default; "MS" = monthly)."""
    return pd.DataFrame(values, index=pd.DatetimeIndex(dates)).resample(freq).mean().dropna(how="all")


def group_deviation(z: np.ndarray, groups: pd.Series, dates: pd.DatetimeIndex, freq: str = "W-FRI") -> pd.DataFrame:
    """Mean |z| per group and period over the group's moving neurons: how far from normal each pathway
    stage is (about 0.8 = a normal week). A group with no moving neuron is all NaN.

    z comes from `zscore`, so neurons that barely move are exactly 0 throughout."""
    wk = weekly(np.abs(z), dates, freq)
    live = np.abs(z).max(axis=0) > 0
    g = np.asarray(groups)
    out = {name: (wk.loc[:, np.flatnonzero((g == name) & live)].mean(axis=1) if ((g == name) & live).any()
                  else pd.Series(np.nan, index=wk.index))
           for name in GROUP_ORDER if (g == name).any()}
    return pd.DataFrame(out)


def moving_share(groups: pd.Series, moving: np.ndarray) -> pd.Series:
    """Fraction of each group's neurons that move, in GROUP_ORDER."""
    share = pd.Series(np.asarray(moving, dtype=float)).groupby(np.asarray(groups)).mean()
    return share.reindex([g for g in GROUP_ORDER if g in share.index])


def stream_activity(reservoir, X: np.ndarray, washout: int, dates, groups: pd.Series, windows: dict,
                    chunk: int = 20_000, min_std: float = MOVE_THRESHOLD, freq: str = "MS"):
    """Group deviations and weekly glow without holding every neuron's full history in memory.

    The whole CNS over 20 years is ~166k neurons x ~5,500 days, too big for one array, so the reservoir
    is re-run once per block of `chunk` neurons (same inputs, so the same dynamics) and only the
    summaries are kept. Returns (group deviation per `freq` period, averaged over each group's moving
    neurons as in `group_deviation`, {window name: weekly mean |z| over that window, weeks x neurons},
    bool mask of neurons that move by at least `min_std`).
    """
    dates = pd.DatetimeIndex(dates)
    n = reservoir.n
    groups = pd.Series(np.asarray(groups), dtype=object)
    sums, counts, parts = {}, {}, {name: [] for name in windows}
    moving = np.zeros(n, dtype=bool)
    week_index = {}
    for start in range(0, n, chunk):
        idx = np.arange(start, min(n, start + chunk))
        states = reservoir.run(X, record_idx=idx, washout=washout)
        moving[idx] = states.std(axis=0) >= min_std
        az = np.abs(zscore(states, min_std))
        del states
        per = weekly(az, dates, freq)
        g = groups.iloc[idx].to_numpy()
        for name in pd.unique(g):
            cols = np.flatnonzero((g == name) & moving[idx])  # the others are 0 throughout
            sums[name] = sums.get(name, 0) + per.iloc[:, cols].sum(axis=1)
            counts[name] = counts.get(name, 0) + len(cols)
        wk = weekly(az, dates)
        for name, (a, b) in windows.items():
            part = wk.loc[a:b]
            week_index[name] = part.index
            parts[name].append(part.to_numpy(np.float32))
    dev = pd.DataFrame({g: sums[g] / counts[g] if counts[g] else sums[g] * np.nan for g in GROUP_ORDER if g in sums})
    glow = {name: pd.DataFrame(np.hstack(p), index=week_index[name]) for name, p in parts.items()}
    return dev, glow, moving


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
