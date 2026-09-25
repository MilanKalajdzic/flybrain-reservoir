import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flyres import plotting
from flyres.scoreboard import SETUPS, folder_results, scoreboard

REPO = Path(__file__).resolve().parents[1]
WIRINGS = ["connectome", "degree_preserving", "weight_shuffle", "sign_shuffle", "erdos_renyi"]
MEMORY = {"connectome": 2.0, "degree_preserving": 16.0, "weight_shuffle": 2.5, "sign_shuffle": 3.0,
          "erdos_renyi": 20.0}


def _fake_run(folder: Path, full: bool = True) -> None:
    """The files the scripts write, with made-up numbers: the fly at 2, Erdős–Rényi best at 20."""
    rows = []
    for w in WIRINGS:
        for gain, scale, valid in ((0.9, 1.0, True), (2.0, 1.5, True), (8.0, 3.0, False)):
            rows.append({"wiring": w, "gain": gain, "memory": 10.0, "memory_sd": 1.0,
                         "memory_noisy": MEMORY[w] * scale, "memory_noisy_sd": 0.5, "active": 0.5,
                         "unstable_worst": 0.0 if valid else 0.5, "seeds": 3, "valid": valid})
    (folder / "gain_sweep").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(folder / "gain_sweep" / "grid.csv", index=False)
    if not full:
        return
    for sub, col in (("region_gains", "memory_noisy_regions"), ("homeostasis", "memory_noisy_homeostatic")):
        (folder / sub).mkdir()
        pd.DataFrame([{"wiring": w, "seed": s, col: MEMORY[w] * (1 + s)} for w in WIRINGS for s in range(3)]).to_csv(
            folder / sub / "results.csv", index=False)
    (folder / "robustness").mkdir()
    pd.DataFrame([{"variant": v, "wiring": w, "memory_noisy": MEMORY[w] / k, "memory_noisy_sd": 0.1}
                  for v, k in (("standard", 1), ("leak_0.5", 4)) for w in WIRINGS]).to_csv(
        folder / "robustness" / "best.csv", index=False)


def test_folder_results_reads_every_setup(tmp_path):
    _fake_run(tmp_path / "full")
    res = folder_results(tmp_path / "full")
    assert set(res) == {"standard", "best_gain", "regions", "neurons", "leak_0.5"}
    assert res["standard"]["connectome"] == 2.0 and res["best_gain"]["erdos_renyi"] == 30.0  # gain 8 is invalid
    assert res["regions"]["connectome"] == 4.0  # mean over seeds
    assert res["leak_0.5"]["erdos_renyi"] == 5.0


def test_scoreboard_picks_the_better_scrambled_wiring(tmp_path):
    _fake_run(tmp_path / "full")
    _fake_run(tmp_path / "flywire_full", full=False)
    table = scoreboard({("male CNS", "whole brain"): tmp_path / "full",
                        ("FlyWire", "whole brain"): tmp_path / "flywire_full"})
    assert len(table) == 5 + 2 and (table["random_wiring"] == "erdos_renyi").all()
    np.testing.assert_allclose(table["ratio"], 10.0)  # never the weight or sign shuffle, which keep the fly's edges
    plt.close(plotting.plot_scoreboard(table, SETUPS, path=tmp_path / "board.png"))
    assert (tmp_path / "board.png").stat().st_size > 0


def test_scoreboard_script(tmp_path):
    _fake_run(tmp_path / "full")
    _fake_run(tmp_path / "small", full=False)
    run = subprocess.run([sys.executable, str(REPO / "scripts" / "scoreboard.py"), "--results", str(tmp_path)],
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    assert (tmp_path / "scoreboard.png").exists() and len(pd.read_csv(tmp_path / "scoreboard.csv")) == 7
    empty = subprocess.run([sys.executable, str(REPO / "scripts" / "scoreboard.py"), "--results",
                            str(tmp_path / "none")], capture_output=True, text=True, timeout=300)
    assert empty.returncode != 0 and "nothing to show" in empty.stderr
