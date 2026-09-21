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


def test_group_deviation_shape_and_scale():
    dates = pd.bdate_range("2020-01-01", periods=300)
    z = np.random.default_rng(1).standard_normal((300, 6)).astype(np.float32)
    groups = pd.Series(["Smell sensors"] * 3 + ["Other central brain"] * 3)
    dev = activity.group_deviation(z, groups, dates)
    assert list(dev.columns) == ["Smell sensors", "Other central brain"]
    assert 0.6 < dev.to_numpy().mean() < 1.0  # E|z| = 0.8 for standard normal noise
    monthly = activity.group_deviation(z, groups, dates, freq="MS")
    assert len(monthly) < len(dev)


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
