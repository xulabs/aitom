import os
import torch
import torch.utils.data as data
import numpy as np
import mrcfile as mrc


class CustomDataset(data.Dataset):
    def __init__(self, root_dir):
        self.image_dir = "./data/subtomogram_mrc/"
        self.mask_dir = "./data/densitymap_mrc/"
        self.image_list = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_name = os.path.join(self.image_dir, self.image_list[item])
        mask_name = os.path.join(self.mask_dir, self.image_list[item].replace('tomo','pack'))
        with mrc.open(img_name, permissive=True) as f:
            img = f.data  # (32, 32, 32)
        with mrc.open(mask_name, permissive=True) as f:
            mask = f.data  # (32, 32, 32)
        img = np.expand_dims(img,0)
        mask = np.expand_dims(mask, 0)
        img = torch.from_numpy(img)
        sample = {'image': img, 'mask': mask}
        return sample


if __name__ == '__main__':
    ds = DUTSDataset('../DUTS-TR')
    ds.arrange()
