from __future__ import print_function, division


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pandas as pd
from skimage import io, transform
import numpy as np
import scipy
import scipy.misc
import time
import os
import math
from sklearn.utils import shuffle
from model_disc import Disc

import sys


from torch.autograd import Function
from utils import test, get_batch, create_set


class GradReverse(Function):
    @staticmethod
    def forward(self, x):
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return grad_output.neg()


def reverse_grad(x):
    return GradReverse.apply(x)


import sys


# from random import shuffle
from sklearn.utils import shuffle
import multiprocessing
from sklearn.feature_extraction.image import extract_patches_2d

sys.path.append("../../")
sys.path.append("../")

import pickle
import tomominer.simulation.reconstruction__simple_convolution as recon
import tomominer.image.io as TIIO
import tomominer.image.vol.util as TIVU
import tomominer.config as config
import shutil
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset
import sys

name = sys.argv[1]
hyperparams = []
if name == "dense":
    from models.densenet3d import DenseNet as Net

    NAME = "dense3D"
    net = Net().cuda()
    net.train()
    hyperparams = [1e-6, 5e-6, 5e-3, 0.8, 3]
    net.load_state_dict(torch.load("./running_weights/{}_basic_cls.pt".format(NAME)))
elif name == "resnet":
    from models.resnet3d import generate_model

    NAME = "resnet3D"
    net = generate_model(101).cuda()
    net.train()
    net.load_state_dict(torch.load("./running_weights/{}_basic_cls.pt".format(NAME)))
    hyperparams = [1e-6, 5e-6, 5e-4, 0.8, 3]


elif name == "dsrf":
    from models.dsrf3d_v2 import Net as Net

    NAME = "dsrf3D"
    net = Net().cuda()
    net.train()
    net.load_state_dict(torch.load("./running_weights/{}_basic_cls.pt".format(NAME)))
    hyperparams = [1e-6, 1e-4, 5e-2, 0.8, 3]

elif name == "cb3d":
    from models.cb3d import Net as Net

    NAME = "cb3D"
    net = Net().cuda()
    net.train()
    net.load_state_dict(torch.load("./running_weights/{}_basic_cls.pt".format(NAME)))
    hyperparams = [1e-6, 5e-5, 5e-3, 0.8, 3]


import os


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.benchmark = True

    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(1)

os.system("clear")
all_training_results = []
all_training_accs = []
all_training_pixel_accs = []
all_training_IoU = []
densityMap_file = open(config.densityMap_path, "rb")
DMs = pickle.load(densityMap_file, encoding="bytes")
DM_masks = {
    b"ribosome": 1.0 * (DMs[b"ribosome"] < -1),
    b"membrane": 1.0 * (DMs[b"membrane"] < 0),
    b"TRiC": 1.0 * (DMs[b"TRiC"] < -3),
    b"proteasome_s": 1.0 * (DMs[b"proteasome_s"] < -2),
}
dec_mod = dec_mod().cuda()

rangeParams = {
    "SNR_min": np.log(0.03),
    "SNR_max": np.log(10),
    "MWA_min": 0,
    "MWA_max": 50,
    "Dz_min": -12,
    "Dz_max": 0,
    "Cs_min": 1.5,
    "Cs_max": 3.0,
}


test_batch_X = np.load(config.testX_path)
test_batch_y = np.load(config.testy_path)

test_batch_mask = pickle.load(open(config.test_file_mask_path, "rb"), encoding="bytes")


test_batch_mask = {
    "ribosome": 1.0 * (np.stack(test_batch_mask[b"ribosome"]) < -1),
    "membrane": 1.0 * (np.stack(test_batch_mask[b"membrane"]) < 0),
    "TRiC": 1.0 * (np.stack(test_batch_mask[b"TRiC"]) < -3),
    "proteasome_s": 1.0 * (np.stack(test_batch_mask[b"proteasome_s"]) < -2),
}

r0 = int(list(test_batch_y).index(1))
r1 = int(list(test_batch_y).index(2))
r2 = int(list(test_batch_y).index(3))
r3 = int(len(list(test_batch_y)))
r = [0, r0, r1, r2, r3]

test_batch_mask = np.concatenate(
    [
        test_batch_mask["ribosome"],
        test_batch_mask["proteasome_s"],
        test_batch_mask["TRiC"],
        test_batch_mask["membrane"],
    ]
)

criterion_disc = nn.BCELoss().cuda()
criterion_cls = nn.CrossEntropyLoss().cuda()


save_dir = os.path.join(config.save_dir_root, config.exp_name)
if os.path.isdir(save_dir):
    shutil.rmtree(save_dir)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)


minval = 0
maxval = 0
curEpochloss = []
curEpochacc = []
curEpoch_pixel_acc = []
curEpoch_IoU = []
min_l = 1e10


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
print("Validation {} Test {}".format(len(dataset_val), len(dataset_test)))

dataloader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=2, num_workers=0, shuffle=True, pin_memory=True
)
dataloader_val = torch.utils.data.DataLoader(
    dataset_val, batch_size=2, num_workers=0, shuffle=True, pin_memory=True
)
epochs = 50

try:
    n_batch = int(sys.argv[2])
    num_iterations = 120 * 4 // n_batch
except:
    n_batch = 4
    num_iterations = 120
phase = "decept"
iter_count = 0


str__ = " "

disc_use = True

acc_t_ = 0
import time

base_time = time.time()


net_disc = Disc().cuda()
optimizer_disc = optim.Adam(
    net_disc.parameters(), lr=hyperparams[0], betas=(0.9, 0.999)
)
criterion_seg = nn.BCELoss().cuda()
optimizer = optim.Adam(net.parameters(), lr=hyperparams[1], betas=(0.9, 0.999))
optimizer_dec = optim.Adam(dec_mod.parameters(), lr=hyperparams[2], betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=hyperparams[3], patience=hyperparams[4], verbose=True
)


if os.path.isfile("./running_weights/disc_0.pt"):
    print("Disc model found")
    net_disc.load_state_dict(torch.load("./running_weights/disc_{}.pt".format(0)))


for i in range(epochs):
    print("Epoch {}".format(i))
    print("Name {} Abl {}".format(NAME, abl))
    loss_acc = 0
    for k in range(num_iterations):
        iter_count += 1

        pool = multiprocessing.Pool(8)
        try:
            batch_X, batch_y, batch_mask = get_batch(
                DMs, DM_masks, n_batch, rangeParams, pool
            )
            pool.close()

        except KeyboardInterrupt:
            print("got ^C while pool mapping, terminating the pool")
            pool.terminate()
            print("pool is terminated")
        except Exception as e:
            print("got exception: %r, terminating the pool" % (e,))
            pool.terminate()
            print("pool is terminated")
        finally:
            # print('joining pool processes')
            pool.join()
            # print('join complete')

        batch_X, batch_y, batch_mask = batch_X.cuda(), batch_y.cuda(), batch_mask.cuda()

        batch_X_warp, batch_mask = dec_mod(batch_X, batch_mask)

        batch_X_disc, batch_y_disc, labels_disc = create_set(
            reverse_grad(batch_X_warp), batch_y, r, test_batch_X, test_batch_y
        )
        batch_X_disc, batch_y_disc, labels_disc = shuffle(
            batch_X_disc, batch_y_disc, labels_disc
        )
        loss_disc = 0
        acc_disc = 0
        ctr_disc = 0

        out = net_disc(batch_X_disc)
        loss_disc = criterion_disc(out.squeeze(), batch_y_disc.float())
        pred_disc = out.detach().squeeze().cpu().numpy()
        pred_disc[pred_disc < 0.5] = 0
        pred_disc[pred_disc >= 0.5] = 1

        acc_disc = np.sum(pred_disc == batch_y_disc.detach().cpu().numpy()) / len(
            batch_y_disc
        )
        input = torch.cat([batch_X_warp.detach(), batch_X], dim=0).cuda()
        batch_y = torch.cat([batch_y, batch_y], dim=0).cuda()

        if name == "cb3d" or name == "dsrf" or name == "fsada":
            _, out_cls = net.forward(input.float())
        else:
            out_cls = net.forward(input.float())

        _, prediction = torch.max(torch.softmax(out_cls, dim=1).data, 1)
        total = np.sum(
            prediction.detach().cpu().numpy()
            == batch_y.squeeze().detach().cpu().numpy()
        )
        acc = (total) / len(batch_y)
        loss_cls = criterion_cls(out_cls, batch_y)

        optimizer_dec.zero_grad()
        optimizer_disc.zero_grad()
        optimizer.zero_grad()
        loss = loss_disc * 1e1 + loss_cls * 1e1

        loss.backward()
        optimizer_dec.step()
        optimizer_disc.step()
        optimizer.step()

        print(
            "[Iter ,loss_class , loss_disc, time]: = [{}, {:.4f}, {:.4f}, {:.2f}]".format(
                k,
                (float(loss_cls * 1e1)),
                float(loss_disc * 1e1),
                (time.time() - base_time),
            ),
            end="\r",
        )
    print()

    net.eval()
    t_correct = 0
    loss_test = 0
    if name == "cb3d" or name == "dsrf" or name == "fsada":

        loss_test, acc_t = test(
            net, torch.device("cuda"), dataloader_test, classification=True, out_size=2
        )

        loss_test, _ = test(
            net,
            torch.device("cuda"),
            dataloader_val,
            classification=True,
            log=False,
            out_size=2,
        )
    else:
        loss_test, acc_t = test(
            net, torch.device("cuda"), dataloader_test, classification=True, out_size=1
        )

        loss_test, _ = test(
            net,
            torch.device("cuda"),
            dataloader_val,
            classification=True,
            log=False,
            out_size=1,
        )

    net.train()
    scheduler.step(loss_test)

    print("Loss Test {:.4f} Min Loss {:.4f}".format(loss_test, min_l))
    torch.save(dec_mod.state_dict(), "./running_weights/{}.pt".format("dec"))
    if loss_test < min_l:
        min_l = loss_test
        acc_t_ = acc_t
        print("Saving Model")

        torch.save(
            net.state_dict(), "./running_weights/{}_advanced_cls.pt".format(NAME)
        )

    print("Top Acc {}".format(acc_t_))


pool.close()
