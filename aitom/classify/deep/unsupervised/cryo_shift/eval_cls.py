from __future__ import print_function, division

import torch

import numpy as np
import scipy.misc
import os

import sys

import pickle
import config

from torch.utils.data import TensorDataset
import sys
from utils import test


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.benchmark = True

    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)
try:
    device = torch.device(sys.argv[1])
except:
    device = torch.device("cuda")

model_path = sys.argv[1]


NAME = model_path[model_path.rindex("/") + 1 :]
NAME = NAME[: NAME.index("_")]
if NAME == "dense3D":
    from models.densenet3d import DenseNet as MatNet

    net = MatNet()
elif NAME == "resnet3D":
    from models.resnet3d import generate_model

    net = generate_model(101)
elif NAME == "dsrf3D":
    from models.dsrf3d_v2 import Net as MatNet

    net = MatNet()
elif NAME == "cb3D":
    from models.cb3d import Net as MatNet

    net = MatNet()
elif NAME == "FSDA":
    from models.model_fsda import FSDA as MatNet

    net = MatNet()
elif NAME == "model":
    from densenet3d import DenseNet as MatNet

    net = MatNet()

net.load_state_dict(torch.load(model_path, map_location=device))
net.eval()


test_batch_X = np.load(config.testX_path)
test_batch_y = np.load(config.testy_path)

test_batch_mask = pickle.load(open(config.test_file_mask_path, "rb"), encoding="bytes")


test_batch_mask = {
    "ribosome": 1.0 * (np.stack(test_batch_mask[b"ribosome"]) < -1),
    "membrane": 1.0 * (np.stack(test_batch_mask[b"membrane"]) < 0),
    "TRiC": 1.0 * (np.stack(test_batch_mask[b"TRiC"]) < -3),
    "proteasome_s": 1.0 * (np.stack(test_batch_mask[b"proteasome_s"]) < -2),
}

test_batch_mask = np.concatenate(
    [
        test_batch_mask["ribosome"],
        test_batch_mask["proteasome_s"],
        test_batch_mask["TRiC"],
        test_batch_mask["membrane"],
    ]
)
r0 = int(list(test_batch_y).index(1))
r1 = int(list(test_batch_y).index(2))
r2 = int(list(test_batch_y).index(3))
r3 = int(len(list(test_batch_y)))
r = [0, r0, r1, r2, r3]
test_batch_X_1 = np.zeros((test_batch_X.shape), dtype=np.float64)
test_batch_y_1 = np.zeros((test_batch_y.shape), dtype=np.longlong)
test_batch_mask_1 = np.zeros((test_batch_mask.shape), dtype=np.float64)
test_batch_X_2 = np.zeros((test_batch_X.shape), dtype=np.float64)
test_batch_y_2 = np.zeros((test_batch_y.shape), dtype=np.longlong)
test_batch_mask_2 = np.zeros((test_batch_mask.shape), dtype=np.float64)
ctr_1 = 0
ctr_2 = 0
for i in range(4):
    l1 = (r[i + 1] - r[i]) // 10
    test_batch_X_2[ctr_1 : ctr_1 + l1] = test_batch_X[r[i] : r[i] + l1]
    test_batch_y_2[ctr_1 : ctr_1 + l1] = test_batch_y[r[i] : r[i] + l1]
    test_batch_mask_2[ctr_1 : ctr_1 + l1] = test_batch_mask[r[i] : r[i] + l1]
    l2 = r[i + 1] - r[i] - l1
    test_batch_X_1[ctr_2 : ctr_2 + l2] = test_batch_X[r[i] + l1 : r[i + 1]]
    test_batch_y_1[ctr_2 : ctr_2 + l2] = test_batch_y[r[i] + l1 : r[i + 1]]
    test_batch_mask_1[ctr_2 : ctr_2 + l2] = test_batch_mask[r[i] + l1 : r[i + 1]]
    ctr_1 = ctr_1 + l1
    ctr_2 = ctr_2 + l2
test_batch_X_2 = test_batch_X_2[:ctr_1]
test_batch_y_2 = test_batch_y_2[:ctr_1]
test_batch_mask_2 = test_batch_mask_2[:ctr_1]
test_batch_X_1 = test_batch_X_1[:ctr_2]
test_batch_y_1 = test_batch_y_1[:ctr_2]
test_batch_mask_1 = test_batch_mask_1[:ctr_2]
dataset_val = TensorDataset(
    torch.tensor(test_batch_X_2),
    torch.tensor(test_batch_y_2),
    torch.tensor(test_batch_mask_2),
)
dataset_test = TensorDataset(
    torch.tensor(test_batch_X_1),
    torch.tensor(test_batch_y_1),
    torch.tensor(test_batch_mask_1),
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=8, num_workers=0, shuffle=True, drop_last=False
)
dataloader_val = torch.utils.data.DataLoader(
    dataset_val, batch_size=8, num_workers=0, shuffle=True, drop_last=False
)
net = net.to(device)
if NAME == "cb3D" or NAME == "FSDA" or NAME == "dsrf3D":
    test(net, device, dataloader_test, use_tqdm=True, classification=True, out_size=2)
else:
    test(net, device, dataloader_test, use_tqdm=True, classification=True, out_size=1)
