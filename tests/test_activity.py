import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from flyres import activity, plotting  # noqa: E402
from flyres.benchmarks import echo_state_gap  # noqa: E402
from flyres.reservoir import Reservoir, scale_weights, signed_weights  # noqa: E402
from flyres.synthetic import synthetic_connectome, synthetic_prices, write_mock_raw_files  # noqa: E402


@pytest.mark.parametrize("superclass, cls, expected", [
    ("cb_sensory", "olfactory", "Smell sensors"),
    ("vnc_sensory", "mechanosensory_tactile", "Touch & body-position sensors"),
    ("cb_intrinsic", "ALPN", "Antennal lobe (smell processing)"),
    ("cb_intrinsic", "MBON", "Mushroom body (learning)"),
    ("descending_neuron", None, "Descending (brain to body)"),
    ("cb_intrinsic", None, "Other central brain"),
    (None, None, "Other"),
])
def test_neuron_group(superclass, cls, expected):
    assert activity.neuron_group(superclass, cls) == expected


def test_groups_label_inputs_and_fold_tiny_groups(sub):
    groups = activity.neuron_groups(sub.neurons, sub.input_idx, min_size=8)
    assert (groups.iloc[sub.input_idx] == activity.INPUT_GROUP).all()
    counts = groups.value_counts()
    assert all(counts[g] >= 8 for g in counts.index if g not in (activity.INPUT_GROUP, "Other"))
    assert set(groups) <= set(activity.GROUP_ORDER)


def test_zscore_leaves_constant_neurons_at_zero():
    rng = np.random.default_rng(0)
    states = np.column_stack([rng.standard_normal(500) * 3 + 1, np.full(500, 0.2)]).astype(np.float32)
    z = activity.zscore(states)
    assert abs(z[:, 0].mean()) < 1e-5 and abs(z[:, 0].std() - 1) < 1e-4
    assert np.all(z[:, 1] == 0)


def test_zscore_ignores_neurons_that_barely_move():
    """Fluctuations below 0.1% of the range carry nothing a noisy readout could use; they stay dark."""
    rng = np.random.default_rng(0)
    states = np.column_stack([rng.standard_normal(500) * 1e-5, rng.standard_normal(500) * 0.01]).astype(np.float32)
    z = activity.zscore(states)
    assert np.all(z[:, 0] == 0) and abs(z[:, 1].std() - 1) < 1e-3
    assert np.abs(activity.zscore(states, min_std=1e-6)[:, 0]).mean() > 0.5  # the old threshold would light it up


def test_stream_activity_matches_the_in_memory_version(sub):
    """Running block by block (the whole brain doesn't fit in memory) gives exactly the in-memory numbers."""
    W, _ = scale_weights(signed_weights(sub.W, sub.sign), 0.9)
    res = Reservoir(W, sub.input_idx, 3, rng=np.random.default_rng(0))
    close = synthetic_prices(700, seed=5)
    X = np.random.default_rng(1).standard_normal((len(close), 3)).astype(np.float32)
    dates = close.index[50:]
    groups = activity.neuron_groups(sub.neurons, sub.input_idx)
    windows = {"a": (str(dates[100].date()), str(dates[200].date())), "b": (str(dates[300].date()), str(dates[400].date()))}
    dev, glow, moving = activity.stream_activity(res, X, 50, dates, groups, windows, chunk=37)
    states = res.run(X, washout=50)
    z = activity.zscore(states)
    pd.testing.assert_frame_equal(dev, activity.group_deviation(z, groups, dates, freq="MS"), check_exact=False,
                                  rtol=1e-5)
    wk = activity.weekly(np.abs(z), dates)
    for name, (a, b) in windows.items():
        np.testing.assert_allclose(glow[name].to_numpy(), wk.loc[a:b].to_numpy(), rtol=1e-5, atol=1e-6)
        assert glow[name].shape[1] == sub.n
    np.testing.assert_array_equal(moving, states.std(axis=0) >= activity.MOVE_THRESHOLD)


def test_group_deviation_shape_and_scale():
    dates = pd.bdate_range("2020-01-01", periods=300)
    z = np.random.default_rng(1).standard_normal((300, 6)).astype(np.float32)
    groups = pd.Series(["Smell sensors"] * 3 + ["Other central brain"] * 3)
    dev = activity.group_deviation(z, groups, dates)
    assert list(dev.columns) == ["Smell sensors", "Other central brain"]
    assert 0.6 < dev.to_numpy().mean() < 1.0  # E|z| = 0.8 for standard normal noise
    monthly = activity.group_deviation(z, groups, dates, freq="MS")
    assert len(monthly) < len(dev)


def test_group_deviation_averages_moving_neurons_only():
    """A neuron that never moves (z = 0 throughout) doesn't dilute its group; a group with none is blank."""
    dates = pd.bdate_range("2020-01-01", periods=300)
    z = np.random.default_rng(2).standard_normal((300, 5)).astype(np.float32)
    z[:, [1, 3, 4]] = 0
    groups = pd.Series(["Smell sensors"] * 2 + ["Other central brain"] + ["Visual"] * 2)
    dev = activity.group_deviation(z, groups, dates, freq="MS")
    solo = activity.group_deviation(z[:, [0]], pd.Series(["Smell sensors"]), dates, freq="MS")
    pd.testing.assert_series_equal(dev["Smell sensors"], solo["Smell sensors"])
    assert dev["Visual"].isna().all() and dev["Other central brain"].notna().all()
    share = activity.moving_share(groups, np.abs(z).max(axis=0) > 0)
    assert list(share.index) == ["Smell sensors", "Other central brain", "Visual"]
    np.testing.assert_allclose(share.to_numpy(), [0.5, 1.0, 0.0])


def test_soma_positions_from_mock_files(tmp_path):
    conn = synthetic_connectome(n=300, mean_degree=10, seed=4)
    write_mock_raw_files(conn, tmp_path, n_fragments=20)
    xyz = activity.soma_positions(tmp_path, conn.neurons["bodyId"])
    sensory = (conn.neurons["superclass"] == "sensory").to_numpy()
    assert np.isfinite(xyz[~sensory]).all()
    assert np.isnan(xyz[sensory, 0]).mean() > 0.5  # most sensory neurons have no soma in the CNS
    assert np.isfinite(xyz[sensory, 0]).any()  # a few fall back to tosomaLocation
    assert len(activity.all_soma_positions(tmp_path)) == (~sensory).sum()


def test_echo_state_gap_on_a_normal_reservoir(sub):
    W, _ = scale_weights(signed_weights(sub.W, sub.sign), 0.9)
    res = Reservoir(W, sub.input_idx, 1, rng=np.random.default_rng(0))
    assert echo_state_gap(res, rng=np.random.default_rng(1)) < 1e-4


def test_echo_state_gap_detects_latching():
    """Strongly self-exciting neurons are flip-flops: early inputs decide their state forever."""
    import scipy.sparse as sp

    def gap(self_weight):
        n = 64
        res = Reservoir(sp.identity(n, format="csr") * self_weight, np.arange(n), 1, input_scaling=0.1,
                        bias_scaling=0.1, rng=np.random.default_rng(0))
        return echo_state_gap(res, rng=np.random.default_rng(1))

    assert gap(0.5) < 1e-4  # contracting: forgets
    assert gap(4.0) > 1.0  # bistable: remembers the distant past


def test_activity_figures_render(tmp_path, sub):
    close = synthetic_prices(400, seed=2)
    dates = close.index[100:]
    rng = np.random.default_rng(3)
    z = rng.standard_normal((len(dates), sub.n)).astype(np.float32)
    groups = activity.neuron_groups(sub.neurons, sub.input_idx)
    dev = activity.group_deviation(z, groups, dates, freq="MS")
    plt.close(plotting.plot_activity_heatmap(dev, close, {"dip": (str(dates[50].date()), str(dates[80].date()))},
                                             path=tmp_path / "heat.png"))
    assert (tmp_path / "heat.png").stat().st_size > 0

    glow = activity.weekly(np.abs(z), dates)
    xy = rng.uniform(0, 100, size=(sub.n, 2))
    xy[: len(sub.input_idx)] = np.nan
    window = (str(glow.index[2].date()), str(glow.index[5].date()))
    gif = plotting.animate_brain(glow, xy, sub.input_idx, close, rng.uniform(0, 100, (2000, 2)), window, "test",
                                 tmp_path / "brain.gif", ticker="SYNTH", fps=4)
    from PIL import Image
    with Image.open(gif) as im:
        assert im.n_frames == 4
    moving = rng.random(sub.n) < 0.3  # only these are drawn; a two-line subtitle fits
    gif = plotting.animate_brain(glow, xy, sub.input_idx, close, rng.uniform(0, 100, (2000, 2)), window, "test",
                                 tmp_path / "brain2.gif", ticker="SYNTH", fps=4, moving=moving,
                                 subtitle="line one\nline two")
    with Image.open(gif) as im:
        assert im.n_frames == 4

    share = activity.moving_share(groups, moving)  # labelled rows; a group where nothing moves is blank
    dev.iloc[:, 1] = np.nan
    plt.close(plotting.plot_activity_heatmap(dev, close, share=share, title="t", path=tmp_path / "heat2.png"))
    assert (tmp_path / "heat2.png").stat().st_size > 0

    gif = plotting.animate_brain_pair([glow, glow * 0.5], xy, sub.input_idx, close, rng.uniform(0, 100, (2000, 2)),
                                      window, "pair", tmp_path / "pair.gif", [("a", "x"), ("b", "y")],
                                      [moving, ~moving], subtitle="s", fps=4)
    with Image.open(gif) as im:
        assert im.n_frames == 4 and im.size == (960, 620)


@pytest.mark.parametrize("wiring, suffix, compare", [("connectome", "", None),
                                                     ("degree_preserving", "_degree_preserving", "connectome")])
def test_brain_activity_script(tmp_path, wiring, suffix, compare):
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    sets = ["name=b", f"output_dir={tmp_path.as_posix()}", "connectome.source=synthetic", "connectome.synthetic_n=500",
            "subgraph.n_neurons=150", "subgraph.n_inputs=10", "market.source=synthetic", "market.synthetic_days=900"]
    cmd = [sys.executable, str(repo / "scripts" / "brain_activity.py"), "--config", str(repo / "configs" / "small.yaml"),
           "--wiring", wiring, "--chunk", "40"] + [x for s in sets for x in ("--set", s)]
    cmd += ["--compare", compare] if compare else []
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    out = tmp_path / "b" / "brain"
    assert (out / f"activity_heatmap{suffix}.png").exists() and (out / f"group_deviation_monthly{suffix}.csv").exists()
    assert "move by at least 0.1%" in run.stdout and "4 runs of 40" in run.stdout
    if compare:  # the second wiring is simulated and saved too
        assert run.stdout.count("simulating") == 2 and (out / "activity.npz").exists()
    (out / f"activity_heatmap{suffix}.png").unlink()
    replot = subprocess.run(cmd + ["--replot"], capture_output=True, text=True, timeout=300)
    assert replot.returncode == 0, replot.stderr[-2000:]
    assert (out / f"activity_heatmap{suffix}.png").exists() and "simulating" not in replot.stdout
