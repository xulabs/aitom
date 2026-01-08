"""
DISCA command-line helper for clustering precomputed features.

This is a light-weight utility that:
  - loads feature vectors (N x D) from .npy
  - runs GaussianMixture for candidate Ks
  - selects the best K by lowest BIC
  - saves labels to .npy and a JSON summary

Example:
    disca cluster --features features.npy --candidate-k 5,10,20 \\
        --out-labels disca_labels.npy --out-summary disca_summary.json

Notes:
  - This operates on extracted features (e.g., from DISCA YOPO encoders).
  - It does not train the DISCA network; it focuses on the clustering step.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture


def parse_candidate_ks(value: str):
    parts = [p for p in value.split(",") if p.strip()]
    ks = [int(p.strip()) for p in parts]
    if not ks:
        raise argparse.ArgumentTypeError("candidate-k must contain at least one integer")
    return ks


def load_features(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected features with shape (N, D); got shape {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError("Need at least 2 samples for clustering.")
    return arr.astype(np.float32, copy=False)


def run_gmm_bic(features: np.ndarray, candidate_ks, reg_covar: float, max_iter: int, random_state: int):
    best = {
        "k": None,
        "bic": np.inf,
        "labels": None,
        "model": None,
    }
    for k in candidate_ks:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=reg_covar,
            max_iter=max_iter,
            random_state=random_state,
        )
        gmm.fit(features)
        bic = gmm.bic(features)
        if bic < best["bic"]:
            best.update({"k": k, "bic": bic, "labels": gmm.predict(features), "model": gmm})
    return best


def save_labels(labels: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, labels.astype(np.int64))


def save_summary(path: Path, summary: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def cmd_cluster(args):
    feats = load_features(Path(args.features))
    candidate_ks = args.candidate_k
    best = run_gmm_bic(
        feats,
        candidate_ks=candidate_ks,
        reg_covar=args.reg_covar,
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    if best["labels"] is None:
        raise RuntimeError("GMM did not produce labels.")

    save_labels(best["labels"], Path(args.out_labels))

    counts = {str(i): int((best["labels"] == i).sum()) for i in np.unique(best["labels"])}
    summary = {
        "chosen_k": int(best["k"]),
        "bic": float(best["bic"]),
        "counts": counts,
        "candidate_k": candidate_ks,
        "reg_covar": args.reg_covar,
        "max_iter": args.max_iter,
        "seed": args.seed,
    }
    save_summary(Path(args.out_summary), summary)

    print(f"[DISCA] Chosen K={best['k']} (BIC={best['bic']:.2f})")
    print(f"[DISCA] Cluster counts: {counts}")
    print(f"[DISCA] Labels saved to: {args.out_labels}")
    print(f"[DISCA] Summary saved to: {args.out_summary}")


def build_parser():
    p = argparse.ArgumentParser(description="DISCA clustering helper")
    sub = p.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("cluster", help="Cluster precomputed features with GMM + BIC")
    pc.add_argument("--features", required=True, help="Path to .npy features array of shape (N, D)")
    pc.add_argument(
        "--candidate-k",
        type=parse_candidate_ks,
        default=parse_candidate_ks("5,10,20"),
        help="Comma-separated candidate K values (default: 5,10,20)",
    )
    pc.add_argument("--reg-covar", type=float, default=1e-5, help="GMM reg_covar (default: 1e-5)")
    pc.add_argument("--max-iter", type=int, default=200, help="GMM max_iter (default: 200)")
    pc.add_argument("--seed", type=int, default=0, help="Random seed for GMM (default: 0)")
    pc.add_argument("--out-labels", default="disca_labels.npy", help="Output .npy for labels")
    pc.add_argument("--out-summary", default="disca_summary.json", help="Output summary JSON")
    pc.set_defaults(func=cmd_cluster)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

