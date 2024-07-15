# Training-free CryoET Tomogram Segmentation

![](https://private-user-images.githubusercontent.com/36667905/348603926-b2ec6fc3-e812-4d29-9d51-4e56930f6d74.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjEwMjA2NDEsIm5iZiI6MTcyMTAyMDM0MSwicGF0aCI6Ii8zNjY2NzkwNS8zNDg2MDM5MjYtYjJlYzZmYzMtZTgxMi00ZDI5LTlkNTEtNGU1NjkzMGY2ZDc0LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA3MTUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNzE1VDA1MTIyMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTcwNmQ1MzU3NTE3YjFkNWFkMWYwYjgxZmVhMDY2ODMwMTdmY2FmOGQwMzE5ZGViYzQ2ZjllNWYzMTk5NDE5ZGUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.zqICPnNvJ9oacQKrawjeCbaTwCitRVp7TFvsvCN2fmA)

> [**Training-free CryoET Tomogram Segmentation**](https://www.arxiv.org/abs/2407.06833),  
> Yizhou Zhao, Hengwei Bian, Michael Mu, Mostofa Rafid Uddin, Zhenyang Li, Xiang Li, Tianyang Wang, Min Xu,  
> MICCAI 2024

## Installation

We tested our code using:

- CentOS 7
- CUDA 11.8
- 1 x Nvidia A100 80G

To install the environment:

```shell
conda create -n cryosam python=3.10 -y
conda activate cryosam
conda install pytorch==2.2.0 torchvision==0.17.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install einops matplotlib mrcfile pandas tqdm
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Data Preparation

We use [Warp](http://www.warpem.com/warp/) to reconstruct and denoise the tomogram before running CryoSAM.

## Getting Started

First download SAM [checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) to `model/`, then the model can be used in just a few lines to get masks from given prompts:

```python
import numpy as np
from cryosam.model import CryoSAM

model = CryoSAM()
# voxel: tomogram of shape (D, H, W)
# input_prompts: point prompts of shape (N, 3)
output = model.infer(voxel, input_prompts)
```

## Citing CryoSAM

If you find this project helpful for your research, please consider citing the following BibTeX entry.

```
@article{zhao2024training,
  title={Training-free CryoET Tomogram Segmentation},
  author={Zhao, Yizhou and Bian, Hengwei and Mu, Michael and Uddin, Mostofa R and Li, Zhenyang and Li, Xiang and Wang, Tianyang and Xu, Min},
  journal={arXiv preprint arXiv:2407.06833},
  year={2024}
}
```