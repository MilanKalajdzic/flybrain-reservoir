import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flyres.config import load_config
from flyres.narma import (best_narma, linear_baseline, memory_link, narma10, narma_markdown, nrmse, run_narma,
                          summarize_narma)

REPO = Path(__file__).resolve().parents[1]


def test_narma10_follows_the_recursion():
    u, y = narma10(500, 3)
    assert u.min() >= 0 and u.max() <= 0.5 and np.abs(y).max() < 1
    t = 200  # y[t] is the value at t+1, computed from u and y up to t
    yy = np.r_[0.0, y]  # yy[s] = NARMA value at time s
    expect = 0.3 * yy[t] + 0.05 * yy[t] * yy[t - 9:t + 1].sum() + 1.5 * u[t - 9] * u[t] + 0.1
    assert y[t] == pytest.approx(expect)
    assert narma10(500, 3)[0] is u and not u.flags.writeable  # cached, read-only


def test_nrmse_and_linear_baseline():
    y = np.random.default_rng(0).normal(size=100)
    assert nrmse(y, y) == 0 and nrmse(np.full(100, y.mean()), y) == pytest.approx(1.0)
    base = linear_baseline(seed=0, n_steps=3000)
    assert 0.4 < base < 0.8  # lags alone can't compute the u(t-9) u(t) product


def _cfg(tmp_path):
    return load_config(None, [
        "name=n", f"output_dir={tmp_path.as_posix()}", "n_jobs=1", "seeds=[0, 1]",
        "wirings=[connectome, erdos_renyi]", "connectome.source=synthetic", "connectome.synthetic_n=500",
        "subgraph.n_neurons=300", "subgraph.n_inputs=15", "subgraph.n_readout=120"])


def test_run_narma_end_to_end(tmp_path):
    cfg = _cfg(tmp_path)
    runs, baseline = run_narma(cfg, gains=[0.5, 0.9, 2.0], verbose=False)
    assert len(runs) == 2 * 2 * 3 and runs["nrmse"].between(0, 2).all()
    # readout noise can only hide information from the readout (up to fitting noise)
    assert (runs["nrmse_noisy"] >= runs["nrmse"] - 0.02).all()
    grid = summarize_narma(runs)
    best = best_narma(grid, 0.9)
    assert list(best["wiring"]) == ["connectome", "erdos_renyi"] and best["standard"].notna().all()
    assert {"nrmse_noisy", "nrmse"} <= set(best.columns)
    # some gain beats the lags-only model clearly, noise-free: the reservoir computes the product term
    # (margin ~0.07 on numpy/scipy versions tested; a 200-neuron reservoir only ties the baseline)
    assert grid["nrmse"].min() < baseline["nrmse"].mean() - 0.03
    sweep = runs[["wiring", "seed", "gain"]].assign(memory_capacity_noisy=np.arange(len(runs), dtype=float))
    link = memory_link(runs, sweep)
    assert link is not None and "memory" in link
    md = narma_markdown(cfg, grid, best, baseline, link)
    assert "NARMA-10" in md and "Linear baseline" in md and "Spearman" in md and "readout noise" in md
    cfg.memory.readout_noise = 0.0  # noise-free only still works
    runs0, _ = run_narma(cfg, gains=[0.9], verbose=False)
    assert "nrmse_noisy" not in runs0 and "nrmse_noisy" not in best_narma(summarize_narma(runs0), 0.9)


def test_narma_script_and_report(tmp_path):
    overrides = ["name=n", f"output_dir={tmp_path.as_posix()}", "n_jobs=1", "wirings=[connectome, erdos_renyi]",
                 "connectome.source=synthetic", "connectome.synthetic_n=500", "subgraph.n_neurons=200",
                 "subgraph.n_inputs=15", "subgraph.n_readout=40"]
    cmd = [sys.executable, str(REPO / "scripts" / "narma.py"), "--config", str(REPO / "configs" / "small.yaml"),
           "--gains", "0.9", "2", "--seeds", "0", "--quiet"] + [x for o in overrides for x in ("--set", o)]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    out = tmp_path / "n" / "narma"
    for f in ("summary.md", "runs.csv", "grid.csv", "linear_baseline.csv", "narma.png", "config_used.yaml"):
        assert (out / f).exists(), f
    (out / "narma.png").unlink()
    replot = subprocess.run(cmd + ["--replot"], capture_output=True, text=True, timeout=300)
    assert replot.returncode == 0, replot.stderr[-2000:]
    assert (out / "narma.png").exists() and "linear baseline" in replot.stdout
    rep = subprocess.run([sys.executable, str(REPO / "scripts" / "report.py"), str(tmp_path / "n")],
                         capture_output=True, text=True, timeout=120)
    assert rep.returncode == 0, rep.stderr[-2000:]
    assert "NARMA-10" in rep.stdout and pd.read_csv(out / "grid.csv")["valid"].dtype == bool
