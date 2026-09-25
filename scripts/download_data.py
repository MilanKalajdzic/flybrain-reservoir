"""Download the connectome files into data/raw.

    python scripts/download_data.py                    # male CNS v1.0, ~1.2 GB
    python scripts/download_data.py --no-weights       # only its two small metadata files
    python scripts/download_data.py --source flywire   # FlyWire 783 (female brain), ~130 MB

Interrupted downloads resume where they stopped. Data: FlyEM / Janelia and the FlyWire Consortium,
both CC-BY 4.0.
"""
import argparse

from flyres.connectome import SOURCES, download, raw_files


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--source", default="malecns", choices=SOURCES)
    p.add_argument("--no-weights", action="store_true", help="skip the connectivity table")
    p.add_argument("--force", action="store_true", help="download again even if the file exists")
    args = p.parse_args()
    which = [k for k in raw_files(args.source) if not (args.no_weights and k in ("weights", "connectivity"))]
    download(args.raw_dir, which, args.force, source=args.source)


if __name__ == "__main__":
    main()
