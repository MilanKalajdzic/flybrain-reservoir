import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from flyres.config import load_config
from flyres.homeostasis import homeostasis_markdown, homeostatic_gains, run_homeostasis
from flyres.subgraph import Subgraph

REPO = Path(__file__).resolve().parents[1]


def _toy(n=120, seed=0):
    """A random network whose rows differ wildly in strength: some neurons get 100x more input than others."""
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=0.08, random_state=rng, data_rvs=lambda k: rng.normal(0, 1, k), format="csr")
    A = (sp.diags(10.0 ** rng.uniform(-2, 0, n)) @ A).tocsr()
    import pandas as pd
    return Subgraph(A.astype(np.float32), np.ones(n, dtype=np.float32), pd.DataFrame({"superclass": ["x"] * n}),
                    np.arange(10))


def test_homeostatic_gains_equalize_input_size():
    sub = _toy()
    cfg = load_config(None, ["reservoir.leak_rate=1.0"])
    g, hist = homeostatic_gains(sub, cfg, sub.W, seed=0, target=0.5, rounds=25)
    assert hist["near_target"].iloc[-1] > 0.8 > hist["near_target"].iloc[0]
    has = np.diff(sub.W.indptr) > 0
    assert (g[~has] == 1).all() and g[has].std() > 0
    # weak rows get turned up, strong rows down
    strength = np.asarray(abs(sub.W).sum(axis=1)).ravel()
    assert np.corrcoef(np.log(strength[has]), np.log(g[has]))[0, 1] < -0.5


def _cfg(tmp_path):
    return load_config(None, [
        "name=h", f"output_dir={tmp_path.as_posix()}", "n_jobs=1", "seeds=[0, 1]",
        "wirings=[connectome, erdos_renyi]", "connectome.source=synthetic", "connectome.synthetic_n=500",
        "subgraph.n_neurons=200", "subgraph.n_inputs=15", "subgraph.n_readout=40",
        "memory.n_steps=700", "memory.max_delay=15", "memory.washout=100"])


def test_run_homeostasis_end_to_end(tmp_path):
    cfg = _cfg(tmp_path)
    results, regions, trials = run_homeostasis(cfg, gains=[0.5, 1.0, 2.0], targets=(0.1, 1.0), min_size=10,
                                               verbose=False)
    assert len(results) == 4 and len(trials) == 4 * 2
    for tag in ("single", "homeostatic"):
        assert {f"memory_{tag}", f"memory_noisy_{tag}", f"valid_{tag}", f"active_readouts_{tag}"} <= set(results)
    assert results["target"].isin([0.1, 1.0]).all() and results["single_gain"].isin([0.5, 1.0, 2.0]).all()
    assert (regions["factor"] > 0).all() and regions["region"].nunique() >= 2
    assert {"median_drive", "near_target", "valid"} <= set(trials)
    md = homeostasis_markdown(cfg, results, regions, trials)
    assert "Homeostatic gains" in md and "Gains by region" in md and "| connectome |" in md


def test_homeostasis_script_and_report(tmp_path):
    overrides = ["name=h", f"output_dir={tmp_path.as_posix()}", "n_jobs=1", "wirings=[connectome, erdos_renyi]",
                 "connectome.source=synthetic", "connectome.synthetic_n=500", "subgraph.n_neurons=200",
                 "subgraph.n_inputs=15", "subgraph.n_readout=40", "memory.n_steps=600", "memory.max_delay=10",
                 "memory.washout=100"]
    cmd = [sys.executable, str(REPO / "scripts" / "homeostasis.py"), "--config", str(REPO / "configs" / "small.yaml"),
           "--gains", "0.9", "2", "--targets", "0.1", "1", "--seeds", "0", "--quiet"]
    cmd += [x for o in overrides for x in ("--set", o)]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    out = tmp_path / "h" / "homeostasis"
    for f in ("summary.md", "results.csv", "regions.csv", "trials.csv", "homeostasis.png", "config_used.yaml"):
        assert (out / f).exists(), f
    (out / "homeostasis.png").unlink()
    replot = subprocess.run(cmd + ["--replot"], capture_output=True, text=True, timeout=120)
    assert replot.returncode == 0, replot.stderr[-2000:]
    assert (out / "homeostasis.png").exists()
    rep = subprocess.run([sys.executable, str(REPO / "scripts" / "report.py"), str(tmp_path / "h")],
                         capture_output=True, text=True, timeout=120)
    assert rep.returncode == 0, rep.stderr[-2000:]
    assert "Homeostatic gains" in rep.stdout
