import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyres.benchmarks import memory_capacities, memory_capacity, readout_stability
from flyres.config import load_config
from flyres.diagnostics import hot_spot_cascade, top_mode
from flyres.experiment import build_matrix, prepare_subgraph
from flyres.reservoir import Reservoir, normalize_inputs, scale_weights, signed_weights
from flyres.sweep import best_valid, run_gain_sweep, summarize_sweep, sweep_markdown

REPO = Path(__file__).resolve().parents[1]


def _random_graph(n=600, p=0.01, seed=0):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=p, random_state=rng, data_rvs=lambda k: np.ones(k), format="lil")
    A.setdiag(0)
    return A


def _with_clique(A, members, weight):
    A = A.tolil(copy=True)
    for i in members:
        for j in members:
            if i != j:
                A[i, j] = weight
    return A.tocsr()


def test_top_mode_finds_a_planted_hot_spot():
    A = _with_clique(_random_graph(), range(30), 5.0)
    lam, spread, core = top_mode(A)
    assert spread < 60
    assert np.isin(core, np.arange(30)).mean() > 0.9
    _, spread_random, _ = top_mode(_random_graph().tocsr())
    assert spread_random > 150


def test_hot_spot_cascade_removes_hot_spots_in_order():
    A = _with_clique(_with_clique(_random_graph(), range(30), 5.0), range(100, 125), 3.0)
    neurons = pd.DataFrame({"superclass": ["x"] * A.shape[0], "class": ["A"] * 30 + ["y"] * 70 + ["B"] * 25
                            + ["z"] * (A.shape[0] - 125)})
    table = hot_spot_cascade(A, neurons, steps=5)
    assert list(table["made_of"].iloc[0])[0] == "A"
    assert list(table["made_of"].iloc[1])[0] == "B"
    assert table["top_abs_eigenvalue"].iloc[0] > table["top_abs_eigenvalue"].iloc[1]
    assert table["spread"].iloc[-1] > table["spread"].iloc[0]  # ends once nothing is localized any more


def test_normalize_inputs():
    W = sp.csr_matrix(np.array([[0, 2.0, -2.0], [0, 0, 0], [3.0, 0, 1.0]]))
    N = normalize_inputs(W, "l1").toarray()
    np.testing.assert_allclose(np.abs(N).sum(axis=1), [1, 0, 1])
    np.testing.assert_array_equal(np.sign(N), np.sign(W.toarray()))
    assert normalize_inputs(W, "none") is not W and (normalize_inputs(W, "none") != W).nnz == 0
    with pytest.raises(ValueError):
        normalize_inputs(W, "l2")


def test_readout_stability(sub):
    W, _ = scale_weights(signed_weights(sub.W, sub.sign), 0.9)
    s = readout_stability(Reservoir(W, sub.input_idx, 1, rng=np.random.default_rng(0)),
                          rng=np.random.default_rng(1))
    assert s["unstable_readouts"] == 0 and s["active_readouts"] > 0.2
    # self-exciting flip-flops with no bias: the first inputs decide which state each one locks into, and the
    # two runs get different first inputs, so a good share must end up in different states
    latching = Reservoir(sp.identity(40, format="csr") * 4.0, np.arange(40), 1, input_scaling=0.1,
                         bias_scaling=0.0, rng=np.random.default_rng(0))
    assert readout_stability(latching, rng=np.random.default_rng(1))["unstable_readouts"] > 0.25
    silent = Reservoir(sp.csr_matrix((40, 40), dtype=np.float32), np.arange(5), 1, bias_scaling=0.0,
                       rng=np.random.default_rng(0))
    assert readout_stability(silent, np.arange(5, 40), rng=np.random.default_rng(1))["active_readouts"] == 0


def _synthetic_cfg(tmp_path, *extra):
    return load_config(None, [
        "name=t", f"output_dir={tmp_path.as_posix()}", "n_jobs=1", "seeds=[0, 1]",
        "connectome.source=synthetic", "connectome.synthetic_n=500",
        "subgraph.n_neurons=200", "subgraph.n_inputs=15", "subgraph.n_readout=40",
        "memory.n_steps=700", "memory.max_delay=15", "memory.washout=100", *extra])


def test_input_normalization_option(tmp_path):
    cfg = _synthetic_cfg(tmp_path, "reservoir.input_normalization=l1")
    sub = prepare_subgraph(cfg, verbose=False)
    W, *_ = build_matrix(sub, cfg, "connectome", 0)
    assert W.nnz == sub.W.nnz


def test_gain_sweep_end_to_end(tmp_path):
    cfg = _synthetic_cfg(tmp_path, "wirings=[connectome, erdos_renyi]")
    df = run_gain_sweep(cfg, gains=[0.5, 1.0, 8.0], verbose=False)
    assert len(df) == 2 * 2 * 3
    assert {"memory_capacity", "memory_capacity_noisy", "active_readouts", "unstable_readouts",
            "echo_gap"} <= set(df.columns)
    assert (df["memory_capacity_noisy"] <= df["memory_capacity"] + 0.5).all()
    summary = summarize_sweep(df)
    assert len(summary) == 2 * 3 and summary["valid"].dtype == bool
    best = best_valid(summary)
    assert list(best["wiring"]) == ["connectome", "erdos_renyi"]
    assert {"memory_noisy", "memory"} <= set(best.columns)
    md = sweep_markdown(cfg, summary, best)
    assert "best valid gain" in md and "memory with readout noise" in md
    # without readout noise the sweep still works, on noise-free memory only
    cfg.memory.readout_noise = 0.0
    df0 = run_gain_sweep(cfg, gains=[1.0], verbose=False)
    assert "memory_capacity_noisy" not in df0.columns
    assert "memory_noisy" not in best_valid(summarize_sweep(df0)).columns


@pytest.mark.parametrize("script, args, produced", [
    ("gain_sweep.py", ["--gains", "0.9", "2", "--seeds", "0"], "gain_sweep/summary.md"),
    ("hot_spots.py", ["--steps", "3"], "hot_spots_connectome.csv"),
])
def test_scripts_run(tmp_path, script, args, produced):
    overrides = ["name=t", f"output_dir={tmp_path.as_posix()}", "n_jobs=1", "wirings=[connectome, erdos_renyi]",
                 "connectome.source=synthetic", "connectome.synthetic_n=500", "subgraph.n_neurons=200",
                 "subgraph.n_inputs=15", "subgraph.n_readout=40", "memory.n_steps=600", "memory.max_delay=10",
                 "memory.washout=100"]
    cmd = [sys.executable, str(REPO / "scripts" / script), "--config", str(REPO / "configs" / "small.yaml")]
    cmd += [x for o in overrides for x in ("--set", o)] + args
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    assert (tmp_path / "t" / produced).exists()


def test_readout_noise_removes_memory_hidden_in_tiny_fluctuations(sub):
    """With a tiny input signal the states wiggle by ~1e-6: a noise-free readout still decodes it, a readout
    that sees 0.1% noise can't. That is exactly the fragile memory the noisy number is meant to exclude."""
    W, _ = scale_weights(signed_weights(sub.W, sub.sign), 0.9)
    kw = dict(n_steps=1500, max_delay=20, washout=100)
    faint = Reservoir(W, sub.input_idx, 1, input_scaling=1e-6, rng=np.random.default_rng(0))
    caps = memory_capacities(faint, None, (0.0, 1e-3), rng=np.random.default_rng(1), **kw)
    assert caps[0.0][0] > 3 * caps[1e-3][0]
    assert caps[0.0][0] == pytest.approx(memory_capacity(faint, None, rng=np.random.default_rng(1), **kw)[0])
    faint_noisy = caps[1e-3][0]
    normal = Reservoir(W, sub.input_idx, 1, rng=np.random.default_rng(0))
    caps = memory_capacities(normal, None, (0.0, 1e-3), rng=np.random.default_rng(1), **kw)
    assert caps[1e-3][0] > 10 * faint_noisy  # a properly driven reservoir keeps real, noise-proof memory


def test_report_script(tmp_path):
    from flyres.experiment import run_experiment

    cfg = _synthetic_cfg(tmp_path, "market.source=synthetic", "market.synthetic_days=1500", "eval.train_min=400",
                         "eval.refit_every=300", "eval.n_boot=100", "wirings=[connectome, erdos_renyi]")
    run_experiment(cfg, verbose=False)
    run = subprocess.run([sys.executable, str(REPO / "scripts" / "report.py"), str(tmp_path / "t")],
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]
    assert "memory (readout noise)" in run.stdout and "| connectome |" in run.stdout
