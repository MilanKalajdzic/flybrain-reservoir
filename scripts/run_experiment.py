"""Run the connectome-vs-controls experiment and write results/<name>/.

    python scripts/run_experiment.py --config configs/small.yaml
    python scripts/run_experiment.py --config configs/small.yaml --set subgraph.n_neurons=1000 --set "seeds=[0,1]"
    python scripts/run_experiment.py --config configs/demo_synthetic.yaml   # offline, fake data

Outputs: summary.md (start here), metrics.csv, comparisons.csv, memory_capacity.csv,
graph_stats.csv, predictions.parquet, figures/*.png, config_used.yaml.
"""
import argparse
import warnings

import matplotlib

matplotlib.use("Agg")  # figures go to files; no window needed

# joblib restarts a worker whose memory grew a lot between jobs (common on the full brain); harmless,
# no results are lost, but the warning looks alarming.
warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given to the executor")

from flyres.config import load_config  # noqa: E402
from flyres.experiment import run_experiment  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. reservoir.leak_rate=0.3 (repeatable)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    result = run_experiment(cfg, verbose=not args.quiet)
    print(f"open {result.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
