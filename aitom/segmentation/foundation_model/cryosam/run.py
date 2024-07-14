import argparse
import random
import warnings
from pathlib import Path

import mrcfile
import numpy as np
import torch

from cryosam.config import CFG
from cryosam.empiar import EMPIAR
from cryosam.model import CryoSAM

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="empiar",
        type=str,
        choices=["empiar"],
    )
    parser.add_argument(
        "--num_prompts",
        default=0.5,
        type=float,
        help="number of prompts in {1, 2, ...} or proportion of annotations in (0, 1)",
    )
    parser.add_argument(
        "--prompt_seed",
        default=42,
        type=int,
    )

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = parse_args()
    print(args)

    set_seed(args.prompt_seed)

    # add datasets here
    dataset_options = {"empiar": EMPIAR}
    configs = {"empiar": CFG}

    cfg = configs[args.dataset]
    dataset = dataset_options[args.dataset](root=cfg.data_dir)

    # test with the first tomogram
    data = dataset[0]

    num_prompts = int(args.num_prompts) if args.num_prompts >= 1 else int(args.num_prompts * len(data["point"]))
    indices = np.random.permutation(np.arange(len(data["point"])))
    input_prompts = data["point"][indices[:num_prompts]]

    print("Initializing models...")
    model = CryoSAM(cfg=cfg)

    print("Inferring...")
    masks = model.infer(data["voxel"], input_prompts, key=data["key"])

    output_dir = Path(CFG.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mrc_path = str(output_dir / (data["key"] + "_mask.mrc"))
    mrcfile.write(mrc_path, masks.astype(np.float32), overwrite=True)
    print(f"Result saved to {mrc_path}")


if __name__ == "__main__":
    main()
