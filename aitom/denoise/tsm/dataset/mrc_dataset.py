import os
import pandas as pd
import numpy as np
import torch
import mrcfile
import tifffile

from torch.utils.data import Dataset
import torch.nn.functional as F


def _normalize_2d(arr: np.ndarray) -> torch.Tensor:
    mean = float(arr.mean())
    std = float(arr.std() + 1e-8)
    arr = (arr - mean) / std
    arr = np.clip(arr, -5.0, 5.0) / 5.0
    return torch.from_numpy(arr)[None, ...]


def read_tif(image_file: str) -> torch.Tensor:
    arr = tifffile.imread(image_file).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    return _normalize_2d(arr)


def read_mrc(image_file: str) -> torch.Tensor:
    """
    Read a 2D MRC (or a slice from a 3D MRC/MRCS) and return a
    single-channel tensor in roughly [-1, 1], shape [1, H, W].

    NOTE: The normalization below is generic. If your pipeline expects
    a different convention (e.g., exact match to read_image), align it here.
    """
    with mrcfile.open(image_file, permissive=True) as mrc:
        arr = mrc.data.astype(np.float32)

    if arr.ndim == 3 and arr.shape[0] > 1:
        arr = arr[arr.shape[0] // 2]

    return _normalize_2d(arr)


class MRCDataset(Dataset):
    """
    Loads MRC files according to split CSVs stored under ``root_dir``.

    Expected naming follows the train.py call sites:
      - train/val/test: ``{mode}_{dataset}.csv`` or fallback ``{mode}.csv``
      - clean: ``clean_{dataset}_abinitio.csv`` when ``abinitio=True``,
        otherwise ``clean_{dataset}.csv``, with fallback to ``clean.csv``
      - tp/fp: ``tp.csv`` / ``fp.csv``

    Produces:
        clean mode -> {"clean_images": HWC, "file_name": str}
        other modes -> {"image": HWC, "noisy_image": HWC, "file_name": str}
    """

    def __init__(
        self,
        root_dir: str,
        mode: str = "train",
        dataset: str | None = None,
        abinitio: bool = False,
        mix: bool = False,
    ):
        self.root_dir = root_dir
        self.mode = mode
        self.dataset = str(dataset) if dataset not in (None, "") else None
        self.abinitio = abinitio
        self.mix = mix

        csv_path = self._resolve_csv_path()
        df = pd.read_csv(csv_path)
        self.data = df.iloc[:, 0].dropna().astype(str).tolist()

    def __len__(self):
        return len(self.data)

    def _resolve_csv_path(self) -> str:
        candidates = []

        if self.mode in {"tp", "fp"}:
            candidates.append(f"{self.mode}.csv")
        elif self.mode == "clean":
            if self.dataset and self.abinitio:
                candidates.append(f"clean_{self.dataset}_abinitio.csv")
            elif self.dataset and self.mix:
                candidates.append(f"clean_{self.dataset}_10A_5A_3A.csv")
            elif self.dataset:
                candidates.append(f"clean_{self.dataset}_10A.csv")
            candidates.append("clean.csv")
        else:
            if self.dataset:
                candidates.append(f"{self.mode}_{self.dataset}.csv")
            candidates.append(f"{self.mode}.csv")

        ordered_candidates = []
        seen = set()
        for name in candidates:
            if name not in seen:
                ordered_candidates.append(name)
                seen.add(name)

        for name in ordered_candidates:
            csv_path = os.path.join(self.root_dir, name)
            if os.path.exists(csv_path):
                return csv_path

        raise FileNotFoundError(
            f"Could not find a CSV split for mode={self.mode!r}, dataset={self.dataset!r}, "
            f"abinitio={self.abinitio}. Tried {ordered_candidates} under {self.root_dir!r}."
        )

    def _resolve_sample_path(self, rel_path: str) -> str:
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.normpath(os.path.join(self.root_dir, rel_path))

    def _to_CHW(self, img):
        if img.ndim == 4:
            img = img.squeeze(0)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        if img.ndim == 3 and img.shape[0] not in (1, 3):
            img = img.permute(2, 0, 1)
        return img.contiguous()

    def _to_HWC(self, img):
        if img.ndim == 3:
            return img.permute(1, 2, 0).contiguous()
        return img

    def __getitem__(self, idx):
        rel_path = self.data[idx]
        img_name = os.path.splitext(os.path.basename(rel_path))[0]

        abs_path = self._resolve_sample_path(rel_path)
        ext = os.path.splitext(abs_path)[1].lower()
        if ext in (".tif", ".tiff"):
            image = read_tif(abs_path)
        else:
            image = read_mrc(abs_path)
        image = self._to_CHW(image)

        if self.mode == "clean":
            image_resized = F.interpolate(
                image.unsqueeze(0),
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            )[0]

            return {
                "clean_images": self._to_HWC(image_resized),
                "file_name": img_name,
                "file_path": abs_path,
            }

        out_img = self._to_HWC(image)
        return {
            "image": out_img,
            "noisy_image": out_img,
            "file_name": img_name,
            "file_path": abs_path,
        }
