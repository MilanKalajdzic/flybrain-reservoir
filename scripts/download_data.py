"""Download the male CNS v1.0 connectome files (~1.2 GB in total) into data/raw.

    python scripts/download_data.py               # everything
    python scripts/download_data.py --no-weights  # only the two small metadata files

Interrupted downloads resume where they stopped. Data: FlyEM / Janelia, CC-BY 4.0.
"""
import argparse

from flyres.connectome import FILES, download


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--no-weights", action="store_true", help="skip the 1.1 GB connectivity table")
    p.add_argument("--force", action="store_true", help="download again even if the file exists")
    args = p.parse_args()
    which = [k for k in FILES if not (args.no_weights and k == "weights")]
    download(args.raw_dir, which, args.force)


if __name__ == "__main__":
    main()
