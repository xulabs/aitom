import json
from pathlib import Path
import os
import sys

import numpy as np


def test_disca_cluster_on_synthetic_gaussians(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from aitom.bin.disca import main as disca_main  # type: ignore

    rng = np.random.default_rng(0)
    n_per = 20
    c1 = rng.normal(loc=[0.0, 0.0], scale=0.3, size=(n_per, 2))
    c2 = rng.normal(loc=[3.0, 0.0], scale=0.3, size=(n_per, 2))
    c3 = rng.normal(loc=[0.0, 3.0], scale=0.3, size=(n_per, 2))
    features = np.vstack([c1, c2, c3]).astype("float32")

    features_path = tmp_path / "features.npy"
    labels_path = tmp_path / "labels.npy"
    summary_path = tmp_path / "summary.json"

    np.save(features_path, features)

    disca_main(
        [
            "cluster",
            "--features",
            str(features_path),
            "--candidate-k",
            "2,3,4",
            "--out-labels",
            str(labels_path),
            "--out-summary",
            str(summary_path),
        ]
    )

    assert labels_path.is_file()
    assert summary_path.is_file()

    labels = np.load(labels_path)
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    assert labels.shape == (3 * n_per,)
    assert int(summary["chosen_k"]) == 3
    counts = {int(k): int(v) for k, v in summary["counts"].items()}
    assert sum(counts.values()) == labels.shape[0]
    assert len(counts) == 3


