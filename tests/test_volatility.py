import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from flyres import plotting  # noqa: E402
from flyres.config import load_config  # noqa: E402
from flyres.market import make_dataset  # noqa: E402
from flyres.readout import walk_forward  # noqa: E402
from flyres.synthetic import synthetic_prices  # noqa: E402
from flyres.volatility import (BASELINES, diebold_mariano, historical_mean, make_vol_data, qlike,  # noqa: E402
                               run_vol_experiment, vol_markdown)

REPO = Path(__file__).resolve().parents[1]


def test_targets_and_features_are_causal():
    close = synthetic_prices(900, seed=1)
    ds = make_dataset(close, horizon=1)
    v = make_vol_data(close, ds, (5,))
    r = np.log(close).diff()
    i = 400
    t = close.index.get_loc(v.dates[i])
    assert v.y[5][i] == pytest.approx(np.log((r.iloc[t + 1:t + 6] ** 2).mean()))
    assert v.har[i, 1] == pytest.approx(np.log((r.iloc[t - 4:t + 1] ** 2).mean()))
    # change prices after day t: features at t stay, targets up to t - 5 stay, the target at t changes
    bumped = close.copy()
    bumped.iloc[t + 1:] *= np.exp(np.random.default_rng(0).normal(0, 0.05, len(close) - t - 1)).cumprod()
    v2 = make_vol_data(bumped, make_dataset(bumped, horizon=1), (5,))
    np.testing.assert_allclose(v2.har[:i + 1], v.har[:i + 1])
    np.testing.assert_allclose(v2.ewma[:i + 1], v.ewma[:i + 1])
    np.testing.assert_allclose(v2.y[5][:i - 4], v.y[5][:i - 4])
    assert v2.y[5][i] != pytest.approx(v.y[5][i])


def test_qlike_diebold_mariano_and_historical_mean():
    rv = np.array([1.0, 2.0, 4.0])
    np.testing.assert_allclose(qlike(rv, rv), 0.0, atol=1e-12)
    assert (qlike(rv, rv * 2) > 0).all() and (qlike(rv, rv / 2) > qlike(rv, rv * 2)).all()  # under-forecasting hurts more
    rng = np.random.default_rng(0)
    y = rng.normal(size=2000)
    good, bad = (y - 0.9 * y) ** 2, (y - 0.0 * y) ** 2
    t, p = diebold_mariano(good, bad, lag=5)
    assert t < -5 and p < 1e-6
    assert np.isnan(diebold_mariano(good, good, lag=5)[0])
    np.testing.assert_allclose(historical_mean(np.arange(6.0), 2)[2:], [0.0, 0.5, 1.0, 1.5])


def test_walk_forward_records_validation_error():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3))
    y = X @ [1.0, -0.5, 0.0] + rng.normal(0, 0.3, 600)
    _, splits = walk_forward(X, y, 1, train_min=300, refit_every=100)
    assert splits["val_mse"].between(0.03, 0.3).all()


def _cfg(tmp_path):
    return load_config(None, [
        "name=v", f"output_dir={tmp_path.as_posix()}", "n_jobs=1", "seeds=[0, 1]",
        "wirings=[connectome, erdos_renyi]", "connectome.source=synthetic", "connectome.synthetic_n=500",
        "subgraph.n_neurons=200", "subgraph.n_inputs=15", "subgraph.n_readout=40", "market.source=synthetic",
        "market.synthetic_days=1600", "eval.train_min=500", "eval.refit_every=300", "memory.n_steps=600",
        "memory.max_delay=10", "memory.washout=100"])


def test_vol_experiment_end_to_end(tmp_path):
    cfg = _cfg(tmp_path)
    graph = pd.DataFrame({"wiring": ["connectome", "connectome", "erdos_renyi", "erdos_renyi"], "seed": [0, 1, 0, 1],
                          "memory_capacity_noisy": [2.0, 2.5, 6.0, 6.5]})
    res = run_vol_experiment(cfg, graph, verbose=False)
    m = res.metrics
    assert set(m["model"]) == {"connectome", "erdos_renyi", *BASELINES}
    assert set(m["horizon"]) == {5, 22} and m.groupby("horizon")["n_test"].nunique().eq(1).all()
    assert m["r2_log"].notna().all() and (m["qlike"] > 0).all()
    # GARCH-type synthetic prices: volatility is predictable, so HAR beats the historical mean
    assert m.loc[(m["model"] == "har") & (m["horizon"] == 5), "r2_log"].iloc[0] > 0.02
    c = res.comparisons
    assert {"connectome vs har", "connectome vs har_inputs", "connectome vs erdos_renyi"} <= set(c["comparison"])
    assert c["dm_t"].notna().all()
    assert len(res.link) == 2 * 2 * 2 and (res.link["memory_measure"] == "with readout noise").all()
    md = vol_markdown(cfg, res)
    assert "Does memory help?" in md and "HAR + inputs" in md and "spectral radius 0.9" in md
    for fig in (plotting.plot_vol_models(m, "SYNTH"), plotting.plot_vol_memory(res.link, "SYNTH"),
                plotting.plot_vol_forecast(res.predictions, "SYNTH", 5, {"all": ("2000-01-01", "2030-01-01")})):
        plt.close(fig)
    # other gains per wiring: same pipeline, the markdown says so
    res2 = run_vol_experiment(cfg, None, verbose=False, gains={"connectome": 2.0, "erdos_renyi": 1.0})
    assert res2.link is None and "best valid gain" in vol_markdown(cfg, res2)
    conn = lambda r: r.metrics.query("model == 'connectome'")["mse_log"].to_numpy()  # noqa: E731
    assert not np.allclose(conn(res), conn(res2))


def test_vol_script_and_report(tmp_path):
    overrides = ["name=v", f"output_dir={tmp_path.as_posix()}", "n_jobs=1", "wirings=[connectome, erdos_renyi]",
                 "connectome.source=synthetic", "connectome.synthetic_n=500", "subgraph.n_neurons=200",
                 "subgraph.n_inputs=15", "subgraph.n_readout=40", "market.source=synthetic",
                 "market.synthetic_days=1600", "eval.train_min=500", "eval.refit_every=300", "memory.n_steps=600",
                 "memory.max_delay=10", "memory.washout=100", "seeds=[0, 1]"]
    sets = [x for o in overrides for x in ("--set", o)]
    base = ["--config", str(REPO / "configs" / "small.yaml")] + sets

    def run(script, *extra):
        out = subprocess.run([sys.executable, str(REPO / "scripts" / script), *base, *extra],
                             capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr[-2000:]
        return out

    run("vol_forecast.py", "--quiet")
    out = tmp_path / "v" / "volatility"
    for f in ("summary.md", "metrics.csv", "comparisons.csv", "predictions.parquet", "config_used.yaml",
              "figures/vol_models_SYNTH.png", "figures/vol_forecast_SYNTH.png"):
        assert (out / f).exists(), f
    run("gain_sweep.py", "--gains", "0.9", "2", "--seeds", "0", "1", "--quiet")
    run("vol_forecast.py", "--quiet", "--best-gains")
    best = tmp_path / "v" / "volatility_best_gain"
    assert (best / "summary.md").exists() and (best / "memory_vs_skill.csv").exists()
    assert (best / "figures" / "vol_memory_SYNTH.png").exists()
    rep = subprocess.run([sys.executable, str(REPO / "scripts" / "report.py"), str(tmp_path / "v")],
                         capture_output=True, text=True, timeout=120)
    assert rep.returncode == 0, rep.stderr[-2000:]
    assert "Volatility forecasts" in rep.stdout and "best valid gain" in rep.stdout
