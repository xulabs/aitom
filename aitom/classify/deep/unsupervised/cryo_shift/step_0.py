from __future__ import print_function, division


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.misc
import time
import os
from sklearn.utils import shuffle
from models.model_disc import Disc
import sys
from utils import create_set, get_batch
import sys


# from random import shuffle
from sklearn.utils import shuffle
import multiprocessing


import pickle
import aitom_core.simulation.reconstruction.reconstruction__simple_convolution as recon
import config
import sys


import os


os.system("clear")
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


net_disc = Disc().cuda()
net_disc.train()
optimizer_disc = optim.Adam(net_disc.parameters(), lr=1e-5, betas=(0.9, 0.999))
criterion_disc = nn.BCELoss().cuda()


minval = 0
maxval = 0
min_l = 1e10


epochs = 5

try:
    n_batch = int(sys.argv[1])
    num_iterations = 60 * 32 // n_batch
except:
    n_batch = 32
    num_iterations = 60


disc_use = True

acc_t_ = 0
import time

base_time = time.time()


for i in range(epochs):
    print("Epoch {}".format(i))
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

        batch_X_disc, batch_y_disc, labels_disc = create_set(
            batch_X, batch_y, r, test_batch_X, test_batch_y
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

        optimizer_disc.zero_grad()
        loss_disc.backward()
        print(
            "Training Discriminator [{}/{}] Loss [{:.4f}] Accuracy {:.4f}".format(
                k, num_iterations, float(loss_disc), acc_disc
            ),
            end="\r",
        )
        optimizer_disc.step()

    print()
    try:
        os.mkdir("running_weights")
        print("Created Directory")
    except:
        pass
    torch.save(net_disc.state_dict(), "./running_weights/disc_{}.pt".format(0))


pool.close()
