import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from flyres import plotting
from flyres.config import load_config
from flyres.robustness import (VARIANTS, fly_gap, random_inputs, robustness_markdown, run_robustness,
                               summarize_robustness, variant_setup)

REPO = Path(__file__).resolve().parents[1]


def _cfg(tmp_path):
    return load_config(None, [
        "name=r", f"output_dir={tmp_path.as_posix()}", "n_jobs=1", "seeds=[0, 1]",
        "wirings=[connectome, erdos_renyi]", "connectome.source=synthetic", "connectome.synthetic_n=500",
        "subgraph.n_neurons=300", "subgraph.n_inputs=15", "subgraph.n_readout=120", "memory.n_steps=800"])


def test_random_inputs_keep_the_circuit(sub):
    moved = random_inputs(sub, seed=0)
    assert moved.W is sub.W and len(moved.input_idx) == len(sub.input_idx)
    assert not np.array_equal(moved.input_idx, sub.input_idx)
    sends = np.asarray(sub.W.sum(axis=0)).ravel() > 0
    assert sends[moved.input_idx].all()  # every input neuron can pass the signal on
    np.testing.assert_array_equal(moved.input_idx, random_inputs(sub, seed=0).input_idx)
    assert not np.array_equal(moved.input_idx, random_inputs(sub, seed=1).input_idx)


def test_variant_setup_leaves_the_base_config_alone(sub, tmp_path):
    cfg = _cfg(tmp_path)
    _, leaky = variant_setup(sub, cfg, "leak_0.2", seed=0)
    assert leaky.reservoir.leak_rate == 0.2 and cfg.reservoir.leak_rate == 1.0
    same, _ = variant_setup(sub, cfg, "standard", seed=0)
    assert same is sub


def test_robustness_end_to_end(tmp_path):
    cfg = _cfg(tmp_path)
    df = run_robustness(cfg, ["standard", "random_inputs", "leak_0.5"], gains=[0.9, 3.0], verbose=False)
    assert len(df) == 3 * 2 * 2 * 2 and set(df["variant"]) == {"standard", "random_inputs", "leak_0.5"}
    grid, best = summarize_robustness(df)
    assert list(best["variant"].unique()) == ["standard", "random_inputs", "leak_0.5"]  # VARIANTS order
    assert len(grid) == 3 * 2 * 2
    gap = fly_gap(best)
    assert len(gap) == 3 and {"ratio", "rank"} <= set(gap.columns)
    md = robustness_markdown(cfg, best)
    assert "random input neurons" in md and "fly against the best control" in md
    labels = {v: label for v, (label, _) in VARIANTS.items()}
    plt.close(plotting.plot_robustness(best, labels, noise=0.001, path=tmp_path / "rob.png"))
    assert (tmp_path / "rob.png").stat().st_size > 0
    wide = best.copy()  # values spanning > 20x switch to a log scale; a wiring with no valid gain gets an x
    wide.loc[wide["variant"] == "leak_0.5", "memory_noisy"] /= 100
    wide.loc[0, "memory_noisy"] = np.nan
    fig = plotting.plot_robustness(wide, labels, noise=0.001)
    assert fig.axes[0].get_yscale() == "log"
    plt.close(fig)


def test_robustness_script(tmp_path):
    sets = ["name=r", f"output_dir={tmp_path.as_posix()}", "n_jobs=1", "wirings=[connectome, erdos_renyi]",
            "connectome.source=synthetic", "connectome.synthetic_n=500", "subgraph.n_neurons=200",
            "subgraph.n_inputs=10", "subgraph.n_readout=80", "memory.n_steps=600"]
    cmd = [sys.executable, str(REPO / "scripts" / "robustness.py"), "--config", str(REPO / "configs" / "small.yaml"),
           "--seeds", "0", "--gains", "0.9", "--variants", "standard", "leak_0.2", "--quiet"]
    cmd += [x for s in sets for x in ("--set", s)]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    out = tmp_path / "r" / "robustness"
    assert {"summary.md", "runs.csv", "grid.csv", "best.csv", "robustness.png"} <= {f.name for f in out.iterdir()}
    stamp = (out / "runs.csv").stat().st_mtime_ns
    (out / "robustness.png").unlink()
    replot = subprocess.run(cmd + ["--replot"], capture_output=True, text=True, timeout=300)
    assert replot.returncode == 0, replot.stderr[-2000:]
    assert (out / "robustness.png").exists() and (out / "runs.csv").stat().st_mtime_ns == stamp  # nothing rerun
