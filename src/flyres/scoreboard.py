"""One table of every memory result: the fly against the better of the two random rewirings, per setup.

Collects what the other scripts already wrote into each run folder (gain sweep, per-region and per-neuron
gains, robustness checks), so the README's summary figure is rebuilt from the same files as its tables.
Memory is always the version with readout noise, 3 seeds per setup.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .sweep import best_valid

SCRAMBLED = ("degree_preserving", "erdos_renyi")  # the two controls that rewire who connects to whom
SETUPS = {  # key: label
    "standard": "standard gain (0.9)",
    "best_gain": "each wiring at its best gain",
    "regions": "a gain per brain region",
    "neurons": "a gain per neuron",
    "random_inputs": "random input neurons",
    "leak_0.5": "leaky neurons (leak 0.5)",
    "leak_0.2": "leaky neurons (leak 0.2)",
}


def _per_wiring(values: pd.Series) -> dict:
    return {w: float(v) for w, v in values.items() if np.isfinite(v)}


def folder_results(folder: str | Path) -> dict[str, dict]:
    """{setup: {wiring: memory with readout noise}} for whatever was run in one result folder."""
    folder = Path(folder)
    out = {}
    grid = folder / "gain_sweep" / "grid.csv"
    if grid.exists():
        g = pd.read_csv(grid)
        used = folder / "gain_sweep" / "config_used.yaml"
        standard = load_config(used).reservoir.spectral_radius if used.exists() else 0.9
        at = g[np.isclose(g["gain"], standard)]
        out["standard"] = _per_wiring(at.set_index("wiring")["memory_noisy"])
        out["best_gain"] = _per_wiring(best_valid(g).set_index("wiring")["memory_noisy"])
    for setup, path, col in (("regions", "region_gains", "memory_noisy_regions"),
                             ("neurons", "homeostasis", "memory_noisy_homeostatic")):
        f = folder / path / "results.csv"
        if f.exists():
            out[setup] = _per_wiring(pd.read_csv(f).groupby("wiring")[col].mean())
    rob = folder / "robustness" / "best.csv"
    if rob.exists():
        b = pd.read_csv(rob)
        for v in ("random_inputs", "leak_0.5", "leak_0.2"):
            if v in set(b["variant"]):
                out[v] = _per_wiring(b[b["variant"] == v].set_index("wiring")["memory_noisy"])
    return out


def scoreboard(folders: dict[tuple[str, str], str | Path]) -> pd.DataFrame:
    """One row per (connectome, scale, setup): the fly, the better scrambled wiring and which one it was.
    folders: {(connectome label, scale label): run folder}."""
    rows = []
    for (connectome, scale), folder in folders.items():
        for setup, by_wiring in folder_results(folder).items():
            scrambled = {w: by_wiring[w] for w in SCRAMBLED if w in by_wiring}
            if "connectome" not in by_wiring or not scrambled:
                continue
            best = max(scrambled, key=scrambled.get)
            rows.append({"connectome": connectome, "scale": scale, "setup": setup, "label": SETUPS[setup],
                         "fly": by_wiring["connectome"], "random": scrambled[best], "random_wiring": best,
                         "ratio": scrambled[best] / by_wiring["connectome"]})
    return pd.DataFrame(rows, columns=["connectome", "scale", "setup", "label", "fly", "random", "random_wiring",
                                       "ratio"])
