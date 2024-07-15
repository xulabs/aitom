import glob
import os

import mrcfile
import numpy as np
from torch.utils.data import Dataset

from cryosam.config import CFG
from cryosam.starfile import Starfile


class EMPIAR(Dataset):
    def __init__(self, root=CFG.data_dir, index=10499):
        self.root = os.path.join(root, str(index))
        self.mrc_files = sorted(glob.glob(os.path.join(self.root, "denoised/*.mrc")))
        prompts = Starfile.load(os.path.join(self.root, "CmMpribo_18987_10A_for_original_and_deconv_tomograms.star"))
        self.prompts = {}
        for tomostar_file in np.unique(np.asarray(prompts.df["_rlnMicrographName"])):
            index = os.path.splitext(tomostar_file)[0]
            prompt = prompts.df[prompts.df["_rlnMicrographName"] == tomostar_file]
            self.prompts[index] = np.flip(np.asarray(prompt)[:, 1:].astype(np.float32), axis=1)

    def __len__(self):
        return len(self.mrc_files)

    def __getitem__(self, item):
        mrc_file = self.mrc_files[item]
        index = os.path.splitext(os.path.basename(mrc_file))[0].split("_")[0]
        voxel = mrcfile.read(mrc_file)
        point = self.prompts[index]
        d, h, w = voxel.shape
        point = np.round(point * [1, -1, -1] + [-16, h - 1, w - 1]).astype(np.int32)
        return {"key": os.path.basename(mrc_file)[:-4], "voxel": voxel, "point": point}
