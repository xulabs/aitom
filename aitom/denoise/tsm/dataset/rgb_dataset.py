import os
import pandas as pd
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset


class RgbDataset(Dataset):
    """
    读取 {root_dir}/{mode}.csv
    - CSV 格式：第一行是列名(=domain)，其后每行一个相对路径 'domain/filename.png'
    - 图片路径: root_dir + 相对路径
    - 返回 (image, None)，不含 label
    """
    def __init__(self, root_dir: str, mode: str = "train"):
        self.root_dir = root_dir

        csv_path = os.path.join(root_dir, f"{mode}.csv")
        df = pd.read_csv(csv_path)
        self.data = df.iloc[:, 0].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        noise_img_name = self.data[idx]                          # noise_img/image_0001.png
        img_name = noise_img_name.split("/")[-1].split(".")[0]                 # image_0001.png
        noisy_image_path = os.path.join(self.root_dir, noise_img_name)    # root_dir + 相对路径

        noisy_image = Image.open(noisy_image_path).convert("RGB")
        noisy_image = torch.from_numpy(np.array(noisy_image))

        return dict(image=noisy_image,
        noisy_image=noisy_image,file_name=img_name
    )

