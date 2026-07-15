from pathlib import Path
import sys
import uuid

import numpy as np
import pytest


def _make_small_faml_dataset(db_path: Path, dj_path: Path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from aitom.io.db.lsm_db import LSM  # type: ignore
    from aitom.image.vol.wedge.util import wedge_mask  # type: ignore

    lsm = LSM(str(db_path))

    v_shape = (16, 16, 16)
    m = wedge_mask(v_shape, ang1=30, sphere_mask=True, verbose=False)

    dj = []
    for _ in range(4):
        v = np.random.randn(*v_shape).astype(np.float32)
        v_fft = np.fft.fftshift(np.fft.fftn(v))
        v_key = str(uuid.uuid4())
        m_key = str(uuid.uuid4())
        lsm[v_key] = v_fft
        lsm[m_key] = m
        dj.append({"v": v_key, "m": m_key, "id": "synthetic"})

    lsm.close()

    import pickle

    with dj_path.open("wb") as f:
        pickle.dump(dj, f, protocol=-1)


def test_faml_cli_runs_on_small_dataset(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    try:
        from aitom.bin.faml import main as faml_main  # type: ignore
    except ModuleNotFoundError:
        pytest.skip("tomominer core not built; skipping FAML CLI test")

    db_path = tmp_path / "synthetic.db"
    dj_path = tmp_path / "dj.pickle"
    out_dir = tmp_path / "out"

    _make_small_faml_dataset(db_path, dj_path)

    faml_main(
        [
            "run",
            "--db",
            str(db_path),
            "--dj",
            str(dj_path),
            "--K",
            "1",
            "--iterations",
            "1",
            "--snapshot-interval",
            "1",
            "--out-dir",
            str(out_dir),
        ]
    )

    checkpoints = list((out_dir / "checkpoints").glob("*.pickle"))
    assert checkpoints, "Expected at least one checkpoint pickle file"


