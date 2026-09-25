"""The summary figure: the fly against the better of the two random rewirings, in every setup that was run.

    python scripts/scoreboard.py            # reads results/full, results/small and the FlyWire runs, if present

Reads what gain_sweep.py, region_gains.py, homeostasis.py and robustness.py wrote into each run folder
(memory with readout noise, 3 seeds per setup) and writes results/scoreboard.png and scoreboard.csv.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from flyres import plotting  # noqa: E402
from flyres.scoreboard import SETUPS, scoreboard  # noqa: E402

RUNS = {("male CNS", "whole brain"): "full", ("male CNS", "3,000-neuron circuit"): "small",
        ("FlyWire", "whole brain"): "flywire_full", ("FlyWire", "3,000-neuron circuit"): "flywire_small"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results", help="folder holding the run folders")
    args = p.parse_args()
    root = Path(args.results)
    folders = {key: root / name for key, name in RUNS.items() if (root / name).exists()}
    table = scoreboard(folders)
    if table.empty:
        raise SystemExit(f"nothing to show: run gain_sweep.py (at least) for the configs first; looked in {root}")
    table.to_csv(root / "scoreboard.csv", index=False)
    plt.close(plotting.plot_scoreboard(table, SETUPS, path=root / "scoreboard.png",
                                       scales=list(dict.fromkeys(k[1] for k in RUNS)),
                                       connectomes=list(dict.fromkeys(k[0] for k in RUNS))))
    print(table.drop(columns="setup").round(2).to_string(index=False))
    print(f"wrote {root / 'scoreboard.png'}")


if __name__ == "__main__":
    main()
