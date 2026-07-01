# Score-Based Matching with Target Guidance for Cryo-EM Denoising

## Overview

This repository provides the reference implementation of Score-Based Matching with Target Guidance for Cryo-EM Denoising. The codebase includes model definition, dataset loading, clean-target projection utilities, data preprocessing, and training scripts required to reproduce the method at a code level.

This public release is organized for anonymous sharing. It contains the training and inference pipeline, but excludes private datasets, derived CSV lists, checkpoints, and experiment artifacts.

## Repository Structure

```text
.
├── configs/
├── data/
│   └── (empty, directory only; data files are not included)
├── dataset/
│   ├── mrc_dataset.py
│   └── rgb_dataset.py
├── models/
├── utils/
│   ├── CleanTargetProjector.py
│   ├── data_process.py
│   └── utils.py
├── .gitignore
├── dataset_split.py
├── requirements.txt
├── train.py
└── README.md
```

## Setup

1. Create a Python environment with the project dependencies.
2. Install the required packages listed in `requirements.txt`, together with the PyTorch build appropriate for your machine.
3. Install dependencies with `pip install -r requirements.txt`.
4. Place the repository on a machine with access to your training data.

The provided `requirements.txt` captures the main Python dependencies used by the released code. The environment should include packages such as:

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `matplotlib`
- `tqdm`
- `easydict`
- `mrcfile`
- `Pillow`
- `scikit-image`
- `scikit-learn`
- `seaborn`

You may adapt package versions to match your local CUDA, PyTorch, and system configuration.

## Data Preparation

This repository does not ship with raw data or CSV split files. The `data/` directory is intentionally included as an empty placeholder.

Before preparing CSV files, first adapt the config files under `configs/` to your dataset characteristics (for example, image size, normalization/statistics assumptions, and training/inference settings).

Before training, prepare your own dataset in the following form:

1. Put your MRC or related input files under your local data root.
2. Create CSV files that list relative file paths for each split.
3. Name the CSV files using the dataset suffix convention expected by `train.py` and `dataset/mrc_dataset.py`.

Expected CSV naming convention:

- `train_{dataset}.csv`
- `val_{dataset}.csv`
- `test_{dataset}.csv`
- `clean_{dataset}_10A.csv` (default clean targets, `--mix false`)
- `clean_{dataset}_10A_3A.csv` (mix mode clean targets, `--mix true`; paths from both 10A and 3A folders)
- `clean_{dataset}_abinitio.csv` if using ab initio clean targets (`--abinitio true`)
- `tp.csv` and `fp.csv` for comparison mode, if needed

The `--dataset` argument controls the `{dataset}` suffix in all of the filenames above.

## Training

Training is launched from `train.py` and requires an explicit dataset identifier.

Example:

```bash
python3 train.py --dataset 10291 --config_json configs/gaussian_map.json --exp_name train
```

If you want the clean target loader to use the ab initio CSV file, add:

```bash
python3 train.py --dataset 10291 --abinitio true --config_json configs/gaussian_map.json --exp_name train
```

Useful arguments:

- `--dataset`: dataset suffix used in CSV names and output naming
- `--abinitio`: when enabled, loads `clean_{dataset}_abinitio.csv` for the clean set
- `--mix`: enables dual-domain clean target mixing (see below)
- `--config_json`: path to the training configuration file
- `--data_dir`: root directory containing the CSV files and referenced data
- `--exp_name`: experiment/output folder name
- `--checkpoint`: checkpoint path used in test or compare modes

### Clean Target Modes

There are three supported modes, controlled by `--abinitio` and `--mix`.

**1. Standard clean targets (default, `--abinitio false --mix false`)**

Use this when you have a 10A resolution clean target folder.
The projector runs a single softmax over all clean samples in `clean_{dataset}_10A.csv`.

```bash
python3 train.py --dataset 10081
```

**2. Ab initio clean targets (`--abinitio true --mix false`)**

Use this when your clean targets come from ab initio reconstruction.
Loads `clean_{dataset}_abinitio.csv` instead of `clean_{dataset}_10A.csv`.
Same single-softmax projector logic as above.

```bash
python3 train.py --dataset 10081 --abinitio true
```

**3. Dual-domain mix mode (`--mix true`)**

Use this when you have clean images from both 10A and 3A resolution folders
(e.g. `clean_images_10081_10A/` and `clean_images_10081_3A/`).
The dataset loader reads `clean_{dataset}_10A_3A.csv`, which lists paths from both folders.
The projector separates them by filename (`10A`/`3A`), runs a separate softmax over each domain,
and combines the results with equal weight (0.5 / 0.5) so every noisy sample simultaneously
receives low-frequency (10A) and high-frequency (3A) supervision.

```bash
python3 train.py --dataset 10081 --mix true
```

## Inference (Test)

Inference is run through `train.py` with `--exp_name test` and a checkpoint.

For your current ab initio setup, use:

```bash
python3 train.py \
	--dataset 11183 \
	--exp_name test \
	--config_json configs/gaussian_map.json \
	--checkpoint /path/to/TSM_denoising/Unsupervised_Denoising_TSM/results/train/11183/TSM/ckpt/11183_TSM_10A_3A_3.pt \
	--abinitio false \
	--mix true \
	--data_dir data \
	--device 0
```

Notes:

- `--exp_name test` switches to inference mode.
- `--checkpoint` must point to a valid `.pt` checkpoint file.
- Set clean mode flags to match how your clean CSV is prepared:
	- `--abinitio true --mix false` -> `clean_{dataset}_abinitio.csv`
	- `--abinitio false --mix true` -> `clean_{dataset}_10A_3A.csv`
	- `--abinitio false --mix false` -> `clean_{dataset}_10A.csv`
- Test outputs are written under `results/test/{dataset}_test_outputs_TSM_10A+3A/`.

## Notes

- The repository is prepared for anonymous release.
- The `data/` folder should remain empty in the public version.
- Do not include private raw data, CSV lists, checkpoints, or result files when publishing.
