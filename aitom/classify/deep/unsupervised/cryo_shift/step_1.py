from __future__ import print_function, division

import torch

import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config
import numpy as np

import time
import os
import shutil
import sys

import multiprocessing

from torch.utils.data import TensorDataset
import pickle
import aitom_core.simulation.reconstruction.reconstruction__simple_convolution as recon

name = sys.argv[1]

if name == "dense":
    from models.densenet3d import DenseNet as Net

    NAME = "dense3D"
    net = Net().cuda()
    net.train()

elif name == "resnet":
    from models.resnet3d import generate_model

    NAME = "resnet3D"
    net = generate_model(101).cuda()
    net.train()

elif name == "dsrf":
    from models.dsrf3d_v2 import Net as Net

    NAME = "dsrf3D"
    net = Net().cuda()
    net.train()

elif name == "fsda":
    from models.model_fsda import FSDA as Net

    NAME = "FSDA"
    net = Net().cuda()
    net.train()

elif name == "cb3d":
    from models.cb3d import Net as Net

    NAME = "cb3D"
    net = Net().cuda()
    net.train()


import os
from utils import test, get_batch

os.system("clear")

print("Running Exp: ", NAME)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.benchmark = True

    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(1)


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

# test_batch_X=test_batch_X-np.min(test_batch_X)
# test_batch_X/=np.max(test_batch_X)

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


criterion_cls = nn.CrossEntropyLoss().cuda()
criterion_seg = nn.BCELoss().cuda()
if name == "fsda":
    optimizer = optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))
else:
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999))

# num_iterations = 30000

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
b_acc = 0
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.9, patience=6, verbose=True
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
    dataset_test,
    batch_size=2,
    num_workers=0,
    shuffle=True,
    drop_last=False,
    pin_memory=True,
)
dataloader_val = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=2,
    num_workers=0,
    shuffle=True,
    drop_last=False,
    pin_memory=True,
)
epochs = 120
num_iterations = 120
n_batch = 32
h = time.time()
for i in range(epochs):
    print("Epoch {} Expt {}".format(i, NAME))
    loss_acc = 0
    for k in range(num_iterations):

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
        input = batch_X
        if name == "fsda" or name == "cb3d" or name == "dsrf":
            _, out_cls = net.forward(input.float())
        else:
            out_cls = net.forward(input.float())

        # classification

        _, prediction = torch.max(torch.softmax(out_cls, dim=1).data, 1)
        total = np.sum(
            prediction.detach().cpu().numpy()
            == batch_y.squeeze().detach().cpu().numpy()
        )
        acc = (total) / len(batch_y)
        loss_cls = criterion_cls(out_cls, batch_y)
        loss_acc += float(loss_cls)

        loss = loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        curEpochloss += [loss.item()]
        curEpochacc += [acc]
        print(
            "Status [Iter , Loss, Accuracy, Time]: = [{}, {:.3f}, {:.3f}, {:.2f}]".format(
                k, loss.item(), acc, time.time() - h
            ),
            end="\r",
        )
    print("===========")

    net.eval()
    t_correct = 0
    t_paccs = []

    t_IoUs = []
    y_pred = []
    y_true = []

    if name == "fsda" or name == "dsrf" or name == "cb3d":
        loss_test = 0
        _, c_acc = test(
            net,
            torch.device("cuda"),
            dataloader_test,
            classification=True,
            log=False,
            out_size=2,
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
        loss_test = 0
        _, c_acc = test(
            net, torch.device("cuda"), dataloader_test, log=False, classification=True
        )
        loss_test, _ = test(
            net,
            torch.device("cuda"),
            dataloader_val,
            classification=True,
            log=False,
        )

    net.train()
    scheduler.step(loss_test)
    print("Loss Test {:.4f} Min Loss {:.4f}".format(loss_test, min_l))
    if loss_test < min_l:
        min_l = loss_test
        b_acc = c_acc
        print("Saving Model")
        try:
            os.mkdir("running_weights")
            print("Created Directory")
        except:
            pass
        torch.save(net.state_dict(), "./running_weights/{}_basic_cls.pt".format(NAME))
    print("Best Accuracy {:.4f}".format(b_acc))

net.load_state_dict(torch.load("./running_weights/{}_basic_cls.pt".format(NAME)))
if name == "fsda" or name == "dsrf" or name == "cb3d":
    _, c_acc = test(
        net, torch.device("cuda"), dataloader_test, classification=True, out_size=2
    )
else:
    _, c_acc = test(
        net, torch.device("cuda"), dataloader_test, classification=True, out_size=1
    )
