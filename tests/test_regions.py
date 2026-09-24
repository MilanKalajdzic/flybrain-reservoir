import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from flyres.config import load_config  # noqa: E402
from flyres.plotting import plot_region_gains  # noqa: E402
from flyres.regions import (MAX_FACTOR, geo_mean_factors, internal_radius, neuron_regions,  # noqa: E402
                            region_markdown, run_region_search, scale_rows, single_gain_peaks)

REPO = Path(__file__).resolve().parents[1]


def test_neuron_regions_merge_sensors_and_fold_small_ones():
    neurons = pd.DataFrame({
        "superclass": ["sensory"] * 20 + ["sensory"] * 20 + ["cb_intrinsic"] * 50 + ["cb_intrinsic"] * 35
                      + ["descending_neuron"] * 5 + ["weird"] * 3,
        "class": ["olfactory"] * 20 + ["gustatory"] * 20 + [None] * 50 + ["ALLN"] * 35 + [None] * 5 + [None] * 3,
    })
    codes, names = neuron_regions(neurons, min_size=30)
    assert names == ["other central brain", "sensory", "antennal lobe", "other"]  # biggest first, "other" last
    assert np.bincount(codes).tolist() == [50, 40, 35, 8]
    assert (np.asarray(names)[codes[:40]] == "sensory").all()


def test_scale_rows_and_internal_radius():
    W = sp.csr_matrix(np.array([[0, 2.0, 0, 0], [3.0, 0, 0, 1.0], [0, 0, 0, 4.0], [0, 0, 1.0, 0]]))
    S = scale_rows(W, [1.0, 0.5, 2.0, 1.0]).toarray()
    np.testing.assert_allclose(S, np.diag([1.0, 0.5, 2.0, 1.0]) @ W.toarray())
    rad = internal_radius(W, np.array([0, 0, 1, 1]), 2)
    np.testing.assert_allclose(rad, [np.sqrt(6.0), 2.0], rtol=1e-6)  # 2-cycles: sqrt(2*3) and sqrt(4*1)


def test_single_gain_peaks():
    """Two humps (edge of chaos near 1, saturated regime at high gain), an invalid gain in between."""
    mem = {0.5: 3.6, 1.0: 6.3, 1.5: 5.3, 2.0: 4.7, 4.0: 5.0, 8.0: 5.9, 12.0: 6.2, 20.0: 5.7}
    rows = [{"gain": g, "m": m, "valid": g != 4.0} for g, m in mem.items()]
    assert [r["gain"] for r in single_gain_peaks(rows, "m")] == [1.0, 12.0]
    assert [r["gain"] for r in single_gain_peaks(rows, "m", k=1)] == [1.0]
    rows = [{"gain": g, "m": g, "valid": True} for g in (0.5, 1.0, 2.0)]  # monotone: one peak, at the end
    assert [r["gain"] for r in single_gain_peaks(rows, "m")] == [2.0]
    assert single_gain_peaks([{"gain": 1.0, "m": 1.0, "valid": False}], "m") == []


def _cfg(tmp_path):
    return load_config(None, [
        "name=t", f"output_dir={tmp_path.as_posix()}", "n_jobs=1", "seeds=[0, 1]",
        "wirings=[connectome, erdos_renyi]", "connectome.source=synthetic", "connectome.synthetic_n=500",
        "subgraph.n_neurons=200", "subgraph.n_inputs=15", "subgraph.n_readout=40",
        "memory.n_steps=700", "memory.max_delay=15", "memory.washout=100"])


def test_region_search_end_to_end(tmp_path):
    cfg = _cfg(tmp_path)
    results, regions, trace = run_region_search(cfg, gains=[0.5, 1.0, 2.0], min_size=10, verbose=False)
    assert len(results) == 4 and set(results["wiring"]) == {"connectome", "erdos_renyi"}
    # the search only accepts improvements on the input it searches on, starting from the best single gain
    assert (results["picked_regions"] >= results["picked_single"]).all()
    for tag in ("single", "regions"):
        assert {f"memory_{tag}", f"memory_noisy_{tag}", f"valid_{tag}", f"active_readouts_{tag}"} <= set(results)
    assert results["single_gain"].isin([0.5, 1.0, 2.0]).all() and results["start_gain"].isin([0.5, 1.0, 2.0]).all()
    assert all(f"{g:g}" in s.split(";") for g, s in zip(results["start_gain"], results["starts_tried"]))
    assert (regions["factor"] > 0).all() and (regions["factor"] <= MAX_FACTOR).all()
    n_regions = regions["region"].nunique()
    assert len(regions) == 4 * n_regions and n_regions >= 2
    assert (regions.groupby(["wiring", "seed"])["readouts"].sum() == 40).all()
    assert (trace.groupby(["wiring", "seed"])["stage"].apply(lambda s: (s == "single gain").sum()) == 3).all()
    assert (results["evals"] == trace.groupby(["wiring", "seed"]).size().to_numpy()).all()
    factors = geo_mean_factors(regions)
    assert list(factors.columns) == ["connectome", "erdos_renyi"]
    md = region_markdown(cfg, results, regions)
    assert "Per-region gains" in md and "fresh input" in md and "| connectome |" in md and "search from" in md
    plt.close(plot_region_gains(results, regions, noise=cfg.memory.readout_noise, path=tmp_path / "r.png"))
    assert (tmp_path / "r.png").exists()


def test_region_search_without_readout_noise(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.memory.readout_noise = 0.0
    cfg.wirings, cfg.seeds = ["connectome"], [0]
    results, regions, _ = run_region_search(cfg, gains=[1.0], min_size=10, verbose=False)
    assert "memory_noisy_single" not in results.columns and results["metric"].iloc[0] == "memory"
    assert "Per-region gains" in region_markdown(cfg, results, regions)
    plt.close(plot_region_gains(results, regions))


def test_region_gains_script_and_report(tmp_path):
    overrides = ["name=t", f"output_dir={tmp_path.as_posix()}", "n_jobs=1", "wirings=[connectome, erdos_renyi]",
                 "connectome.source=synthetic", "connectome.synthetic_n=500", "subgraph.n_neurons=200",
                 "subgraph.n_inputs=15", "subgraph.n_readout=40", "memory.n_steps=600", "memory.max_delay=10",
                 "memory.washout=100"]
    cmd = [sys.executable, str(REPO / "scripts" / "region_gains.py"), "--config", str(REPO / "configs" / "small.yaml"),
           "--gains", "0.9", "2", "--seeds", "0", "--min-region", "10"]
    cmd += [x for o in overrides for x in ("--set", o)]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    out = tmp_path / "t" / "region_gains"
    for f in ("summary.md", "results.csv", "regions.csv", "evaluations.csv", "region_gains.png", "config_used.yaml"):
        assert (out / f).exists(), f
    (out / "region_gains.png").unlink()
    replot = subprocess.run(cmd + ["--replot"], capture_output=True, text=True, timeout=120)
    assert replot.returncode == 0, replot.stderr[-2000:]
    assert (out / "region_gains.png").exists()
    rep = subprocess.run([sys.executable, str(REPO / "scripts" / "report.py"), str(tmp_path / "t")],
                         capture_output=True, text=True, timeout=120)
    assert rep.returncode == 0, rep.stderr[-2000:]
    assert "Per-region gains" in rep.stdout and "| connectome |" in rep.stdout
