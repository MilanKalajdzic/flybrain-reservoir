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
    setup = ("import numpy as np, pandas as pd\nfrom flyres import plotting\n"
             "neu = pd.DataFrame({'in_degree': np.arange(1, 50), 'out_degree': np.arange(1, 50)})")
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(setup)] + [
        nbformat.v4.new_code_cell("plotting.plot_degree_ccdf(neu);") for _ in range(3)]
    nbclient.NotebookClient(nb, timeout=120, kernel_name="flyres-test").execute()
    shown = [any("image/png" in o.get("data", {}) for o in c.outputs) for c in nb.cells[1:]]
    assert shown == [True, True, True]
