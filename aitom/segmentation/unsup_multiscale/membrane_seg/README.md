# Membrane Segmentation (Supervised Refinement)

Supervised deep learning refinement of membrane segmentation, trained on pseudo ground truth labels from the main unsupervised pipeline.

---

## Pseudo Ground Truth Labels

Run the main pipeline first — it generates membrane masks automatically via CellPose:

```bash
python unified_tomogram_pipeline.py --input_path input/raw_data --output_path output/results
```

Masks are saved to `output/<TOMOGRAM_NAME>/membrane_masks/` and used directly as training labels here.

---

## Data Preparation

Pack tomogram slices and their membrane masks into `.h5` files:

```python
import h5py

with h5py.File('data/sample1.h5', 'w') as f:
    f.create_dataset('image', data=image_array)   # tomogram slice (H, W)
    f.create_dataset('label', data=label_array)   # mask from membrane_masks/
```

Place files under `root_path/data/` and create two list files:

- `labeled.list` — sample names for training (one per line, no `.h5`)
- `val.list` — sample names for validation

---

## Train

```bash
python train.py --exp my_experiment --root_path /path/to/root_path
```

Key options: `--model` (`unet` / `vnet` / `unet3d` / `unetr`), `--max_iterations` (default `30000`), `--labeled_bs` (default `8`), `--gpu`.

---

## Test

```bash
python test_supervised.py --exp my_experiment --root_path /path/to/root_path
```

Predictions → `predictions/` | Metrics (Dice, Jaccard, HD95, ASD) → `metrics.txt`