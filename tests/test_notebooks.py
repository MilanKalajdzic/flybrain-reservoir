import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_run_notebooks_script(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [nbformat.v4.new_code_cell("import flyres, sys\nprint(sys.executable)"),
                nbformat.v4.new_code_cell("import matplotlib.pyplot as plt\nplt.plot([1, 2, 3]);")]
    path = tmp_path / "tiny.ipynb"
    nbformat.write(nb, path)
    env = {**os.environ, "MPLBACKEND": "Agg"}  # the script has to undo this, or the figure never shows
    run = subprocess.run([sys.executable, str(REPO / "scripts" / "run_notebooks.py"), str(path)],
                         capture_output=True, text=True, timeout=300, env=env)
    assert run.returncode == 0, run.stderr[-2000:]
    done = json.loads(path.read_text())
    printed = "".join(done["cells"][0]["outputs"][0]["text"]).strip()
    assert Path(printed).resolve() == Path(sys.executable).resolve()  # ran with this Python, not another kernel
    assert any("image/png" in o.get("data", {}) for o in done["cells"][1]["outputs"])
    assert done["metadata"]["kernelspec"]["name"] == "python3"
