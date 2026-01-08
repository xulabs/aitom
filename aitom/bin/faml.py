"""
FAML command-line helper for subtomogram averaging.

Runs the EM-based FAML algorithm on an existing LSM database and a dj.pickle
key file, following the pattern used in doc/tutorials/009_faml.py.
"""

import argparse
from pathlib import Path
import pickle

import aitom.average.ml.faml.faml as faml


def load_img_data(db_path: Path, dj_path: Path):
    with dj_path.open("rb") as f:
        dj = pickle.load(f, encoding="iso-8859-1")
    return {"db_path": str(db_path), "dj": dj}


def cmd_run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_data = load_img_data(Path(args.db), Path(args.dj))

    faml.EM(
        img_data=img_data,
        K=args.K,
        iteration=args.iterations,
        path=str(out_dir),
        snapshot_interval=args.snapshot_interval,
        reg=args.reg,
        use_voronoi=not args.no_voronoi,
    )


def build_parser():
    p = argparse.ArgumentParser(description="FAML subtomogram averaging helper")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="Run EM-based FAML averaging")
    pr.add_argument("--db", required=True, help="Path to LSM database (e.g. aitom_demo_subtomograms.db)")
    pr.add_argument("--dj", required=True, help="Path to dj.pickle describing keys in the database")
    pr.add_argument("--K", type=int, required=True, help="Number of classes (clusters)")
    pr.add_argument("--iterations", type=int, default=20, help="Number of EM iterations (default: 20)")
    pr.add_argument("--snapshot-interval", type=int, default=5, help="Snapshot interval in iterations (default: 5)")
    pr.add_argument("--out-dir", default="faml_output", help="Output directory for checkpoints and averages")
    pr.add_argument("--reg", action="store_true", help="Enable regularization")
    pr.add_argument("--no-voronoi", action="store_true", help="Disable Voronoi weights")
    pr.set_defaults(func=cmd_run)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()


