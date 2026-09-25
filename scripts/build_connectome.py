"""Turn the raw files into a cached sparse matrix + neuron table (one-off, a few minutes).

    python scripts/build_connectome.py                   # male CNS, edges with >= 5 synapses
    python scripts/build_connectome.py --min-weight 1    # keep everything
    python scripts/build_connectome.py --source flywire  # FlyWire 783 (female brain)

run_experiment.py does this automatically on first use; this script just lets you do it up front
and see the stats.
"""
import argparse

from flyres.connectome import SOURCES, build_cache


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--source", default="malecns", choices=SOURCES)
    p.add_argument("--min-weight", type=int, default=5, help="minimum synapses for an edge to count")
    p.add_argument("--keep-autapses", action="store_true", help="keep neuron-onto-itself synapses")
    args = p.parse_args()
    build_cache(args.raw_dir, args.cache_dir, args.min_weight, drop_autapses=not args.keep_autapses,
                source=args.source)


if __name__ == "__main__":
    main()
