import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from flyres import plotting  # noqa: E402


@pytest.fixture
def neurons():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"in_degree": rng.integers(1, 100, 300), "out_degree": rng.integers(1, 100, 300)})


def test_figures_render_and_save(tmp_path, neurons):
    plotting.plot_degree_ccdf(neurons, path=tmp_path / "deg.png")
    plotting.plot_spectra({"connectome": np.exp(1j * np.linspace(0, 6, 50)) * 0.5}, radius=0.9,
                          path=tmp_path / "spec.png")
    assert (tmp_path / "deg.png").stat().st_size > 0 and (tmp_path / "spec.png").stat().st_size > 0
    plt.close("all")


def test_sweep_figures_keep_their_title():
    """Regression: the panel loop reused the name `title`, so the figure was titled after its last panel."""
    grid = pd.DataFrame([{"wiring": w, "gain": g, "memory": 5.0, "memory_noisy": 2.0, "nrmse": 0.5,
                          "nrmse_noisy": 0.7, "valid": g < 2} for w in ("connectome", "erdos_renyi") for g in (0.9, 2)])
    fig = plotting.plot_gain_sweep(grid, noise=0.001, title="Memory, whole male CNS")
    assert fig._suptitle.get_text() == "Memory, whole male CNS"
    assert [ax.get_title(loc="left") for ax in fig.axes] == ["With readout noise (0.1% of max activity)",
                                                             "Noise-free (standard benchmark)"]
    fig2 = plotting.plot_narma(grid, baseline=0.6, noise=0.001, title="NARMA, whole male CNS")
    assert fig2._suptitle.get_text() == "NARMA, whole male CNS"
    assert "linear model,\nlast 10 inputs" in [t.get_text() for t in fig2.axes[-1].get_legend().get_texts()]
    plt.close("all")


def test_plotting_leaves_global_matplotlib_state_alone(neurons):
    """Regression: an rc_context around figure creation used to reset IPython's interactive mode,
    so only the first figure in a notebook was shown."""
    before = dict(matplotlib.rcParams)
    plotting.plot_degree_ccdf(neurons)
    plt.close("all")
    after = dict(matplotlib.rcParams)
    assert {k for k in before if before[k] != after[k]} == set()


def test_every_figure_shows_in_a_notebook(neurons, tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    # A throwaway kernel that runs *this* Python, so the test can't pick up some other "python3"
    # kernel registered on the machine (one without flyres installed).
    import json
    import sys

    spec = tmp_path / "kernels" / "flyres-test"
    spec.mkdir(parents=True)
    (spec / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "flyres test", "language": "python"}))
    monkeypatch.setenv("JUPYTER_PATH", str(tmp_path))
    monkeypatch.delenv("MPLBACKEND", raising=False)  # a forced backend (e.g. Agg on CI) would override inline
    setup = ("import numpy as np, pandas as pd\nfrom flyres import plotting\n"
             "neu = pd.DataFrame({'in_degree': np.arange(1, 50), 'out_degree': np.arange(1, 50)})")
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(setup)] + [
        nbformat.v4.new_code_cell("plotting.plot_degree_ccdf(neu);") for _ in range(3)]
    nbclient.NotebookClient(nb, timeout=120, kernel_name="flyres-test").execute()
    shown = [any("image/png" in o.get("data", {}) for o in c.outputs) for c in nb.cells[1:]]
    assert shown == [True, True, True]


def test_run_label_names_the_brain():
    from pathlib import Path

    from flyres.config import load_config, run_label

    configs = Path(__file__).resolve().parents[1] / "configs"
    assert [run_label(load_config(configs / f"{n}.yaml")) for n in ("full", "small", "flywire_full", "flywire_small")] == [
        "whole male CNS", "3,000-neuron circuit, male CNS", "whole FlyWire brain (female)",
        "3,000-neuron circuit, FlyWire (female)"]
