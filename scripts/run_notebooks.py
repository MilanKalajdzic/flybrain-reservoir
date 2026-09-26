"""Rerun the notebooks in place, so the outputs saved in them match your data and the current code.

    python scripts/run_notebooks.py                                   # both notebooks
    python scripts/run_notebooks.py notebooks/02_results.ipynb        # just one

Runs them with this Python (the repo's venv), not whatever Jupyter kernel called "python3" is registered on
the machine. 02_results.ipynb reruns configs/small.yaml, a few minutes. Needs the dev extras
(pip install -e ".[dev]").
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = [REPO / "notebooks" / "01_connectome_tour.ipynb", REPO / "notebooks" / "02_results.ipynb"]
KERNEL = "flyres-run"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("notebooks", nargs="*", type=Path, default=NOTEBOOKS)
    p.add_argument("--timeout", type=int, default=1800, help="seconds per cell (default 1800)")
    args = p.parse_args()

    try:
        import nbclient
        import nbformat
    except ImportError:
        sys.exit('needs nbclient, nbformat and ipykernel: pip install -e ".[dev]"')

    with tempfile.TemporaryDirectory() as tmp:
        # a throwaway kernel that runs this Python, found before any other kernel on the machine
        spec = Path(tmp) / "kernels" / KERNEL
        spec.mkdir(parents=True)
        (spec / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "flyres", "language": "python"}))
        os.environ["JUPYTER_PATH"] = tmp + os.pathsep + os.environ.get("JUPYTER_PATH", "")
        os.environ.pop("MPLBACKEND", None)  # a forced backend (e.g. Agg) would keep figures out of the notebook

        for path in args.notebooks:
            path = path.resolve()
            nb = nbformat.read(path, as_version=4)
            start = time.time()
            print(f"running {path.relative_to(REPO) if path.is_relative_to(REPO) else path} ...", flush=True)
            nbclient.NotebookClient(nb, timeout=args.timeout, kernel_name=KERNEL,
                                    resources={"metadata": {"path": str(path.parent)}}).execute()
            # saved as a plain Python 3 notebook, not tied to the throwaway kernel
            nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
            nbformat.write(nb, path)
            print(f"  done in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
