import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from config import *
from models import resnet
from preprocessor import DalDataset
from sampler import SubsetSequentialSampler
from utils import preprocess_data, train_validate_test_split
import time


parser = argparse.ArgumentParser("Options for training DAL on cryo-ET")
parser.add_argument(
    "--data_dir",
    type=str,
    default=os.path.join(
        os.path.abspath("../../../../.."), "cryoET/"
    ),
    help="default folder to place data",
)   # specify data path here
parser.add_argument(
    "--num_classes", type=int, default=50, help="how many classes training for"
)
parser.add_argument(
    "--target_snr_type",
    type=str,
    default="SNR005",
    help="specify target snr types for training",
)
args = parser.parse_args()


output_dict = preprocess_data(
    data_dir=args.data_dir,
    num_classes=args.num_classes,
    target_snr_type=args.target_snr_type,
    normalization=True
)
train, _, test = output_dict["train"], output_dict["valid"], output_dict["test"]
cryo_train = DalDataset(train)
cryo_unlabeled = DalDataset(train)
cryo_test = DalDataset(test)
# ----------- end data pipeline ------------


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


# Train Utils
iters = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_epoch(models, criterion, optimizers, dataloaders, cycle, epoch):
    models["backbone"].train()
    models["ema"].train()
    global iters

    for data in dataloaders["train"]:
        inputs = data[0].float().to(device)
        labels = data[1].to(device)
        iters += 1

        optimizers["backbone"].zero_grad()

        scores, cons_scores, _, _ = models["backbone"](inputs)
        target_loss = criterion(scores, labels)

        ema_scores, _, _, _ = models["ema"](inputs)

        res_loss = F.mse_loss(scores, cons_scores)
        consistency_loss = F.mse_loss(cons_scores, ema_scores)

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        if cycle == 0:
            loss = m_backbone_loss
        else:
            loss = m_backbone_loss + 0.03 * (res_loss + consistency_loss)
        loss.backward()
        optimizers["backbone"].step()
        update_ema_variables(models["backbone"], models["ema"], 0.999, iters)


#
def test(models, dataloaders, mode="val"):
    assert mode == "val" or mode == "test"
    models["backbone"].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            scores, _, _, _ = models["backbone"](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total


#
def train(
    models,
    criterion,
    optimizers,
    schedulers,
    dataloaders,
    num_epochs,
    cycle
):
    print(">> Train a Model...")
    best_acc = 0.0
    checkpoint_dir = os.path.join("./Cryo_Electron", "train", "weights")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(num_epochs):

        train_epoch(
            models, criterion, optimizers, dataloaders, cycle, epoch
        )
        schedulers["backbone"].step()

        # Save a checkpoint
        if epoch % 20 == 0 or epoch == 99:
            acc = test(models, dataloaders, "test")
            if best_acc < acc:
                best_acc = acc
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "state_dict_backbone": models["backbone"].state_dict()
                    },
                    "%s/active_resnet18_Cryo_Electron.pth" % (checkpoint_dir),
                )
            print(
                "Cycle:",
                cycle,
                "Epoch:",
                epoch,
                "---",
                "Val Acc: {:.3f} \t Best Acc: {:.3f}".format(acc, best_acc),
                flush=True,
            )
    print(">> Finished.")


def get_uncertainty(models, unlabeled_loader):
    models["backbone"].eval()
    models["ema"].eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.float().to(device)

            scores, _, _, feats_list = models["backbone"](inputs)
            ema_scores, _, _, ema_feats_list = models["ema"](inputs)

            variance = F.kl_div(F.log_softmax(scores, dim=1), F.softmax(ema_scores, dim=1), reduction='batchmean')
            variance = variance.unsqueeze(0)

            uncertainty = torch.cat((uncertainty, variance), dim=0)

    return uncertainty.cpu()


##
# Main
if __name__ == "__main__":

    for trial in range(TRIALS):
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)
        START = 2 * ADDENDUM
        labeled_set = indices[:START]
        unlabeled_set = indices[START:]

        train_loader = DataLoader(
            cryo_train,
            batch_size=BATCH,
            sampler=SubsetRandomSampler(labeled_set),
            pin_memory=True,
        )
        test_loader = DataLoader(cryo_test, batch_size=BATCH)
        extra_loader = DataLoader(
            cryo_train,
            batch_size=BATCH,
            sampler=SubsetSequentialSampler(unlabeled_set),
            pin_memory=True,
        )

        dataloaders = {
            "train": train_loader,
            "test": test_loader,
            "extra": extra_loader,
        }


        start = time.time()
        # Active learning cycles
        for cycle in range(CYCLES):
            criterion = nn.CrossEntropyLoss(reduction="none")

            # Model
            backbone_net = resnet.resnet18(num_classes=args.num_classes).to(device)

            ema_model = resnet.resnet18(num_classes=args.num_classes).to(device)
            for param in ema_model.parameters():
                param.detach_()

            models = {"backbone": backbone_net, "ema": ema_model}
            torch.backends.cudnn.benchmark = True

            optim_backbone = optim.SGD(
                models["backbone"].parameters(),
                lr=LR,
                momentum=MOMENTUM,
                weight_decay=WDECAY,
            )

            sched_backbone = lr_scheduler.MultiStepLR(
                optim_backbone, milestones=MILESTONES
            )

            optimizers = {"backbone": optim_backbone}
            schedulers = {"backbone": sched_backbone}

            # Training and test
            train(
                models,
                criterion,
                optimizers,
                schedulers,
                dataloaders,
                EPOCH,
                cycle
            )
            acc = test(models, dataloaders, mode="test")


            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]

            unlabeled_loader = DataLoader(
                cryo_unlabeled,
                batch_size=1,
                sampler=SubsetSequentialSampler(
                    subset
                ),
                pin_memory=True,
            )

            # Measure uncertainty of each data point in the subset
            uncertainty = get_uncertainty(models, unlabeled_loader)

            # Index in ascending order
            arg = np.argsort(uncertainty)

            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(
                torch.tensor(subset)[arg][-ADDENDUM:].numpy()
            )  # select largest
            unlabeled_set = (
                list(torch.tensor(subset)[arg][:-ADDENDUM].numpy())
                + unlabeled_set[SUBSET:]
            )

            # Create a new dataloader for the updated labeled dataset
            dataloaders["train"] = DataLoader(
                cryo_train,
                batch_size=BATCH,  # BATCH
                sampler=SubsetRandomSampler(labeled_set),
                pin_memory=True,
            )


        # Save a checkpoint
        torch.save(
            {
                "trial": trial + 1,
                "state_dict_backbone": models["backbone"].state_dict(),
            },
            "./Cryo_Electron/train/weights/active_resnet18_Cryo_Electron_trial{}.pth".format(
                trial
            ),
        )

        print(
            "---------------------------Current Trial is done-----------------------------",
            flush=True,
        )

        end = time.time()
        print(end-start)
