import os
import torch
from torch.autograd import Variable
from dataset import *
from models import *
import argparse
import logging


def train_one_epoch(data, batches, model, optimizer):
    total_c_loss = 0.0
    model.train()

    for i in range(batches):
        x1, x1_seg, y1 = data.sample_new_train_batch()
        x2, x2_seg, y2 = data.sample_new_train_batch()

        x1 = Variable(torch.from_numpy(x1)).float()
        x2 = Variable(torch.from_numpy(x2)).float()
        x1 = x1.permute(0, 4, 1, 2, 3)
        x2 = x2.permute(0, 4, 1, 2, 3)

        x1_seg = Variable(torch.from_numpy(x1_seg)).float()
        x2_seg = Variable(torch.from_numpy(x2_seg)).float()
        x1_seg = x1_seg.permute(0, 4, 1, 2, 3)
        x2_seg = x2_seg.permute(0, 4, 1, 2, 3)

        _, l_bce, l_dice = model(x1.cuda(), x1_seg, y1, x2.cuda(), x2_seg, y2, mode='train')
        l_total = l_bce + l_dice

        optimizer.zero_grad()
        l_total.backward()
        optimizer.step()

        if i > 0 and i % 100 == 0:
            print("Batch {}/{} => Train TotalLoss:{} DiceLoss:{} BCELoss:{}".format(i, batches,
                                                                                    l_total,
                                                                                    l_dice,
                                                                                    l_bce))

    return


def test_nets(data, model, batch_size):
    total_c_loss = 0.0
    total_c_accuracy = 0.0

    model.eval()
    x1, x1_seg, y1, x2, x2_seg, y2 = data.sample_new_test_batch()

    x1 = Variable(torch.from_numpy(x1)).float()
    x1 = x1.permute(0, 4, 1, 2, 3)

    x1_seg = Variable(torch.from_numpy(x1_seg)).float()
    x1_seg = x1_seg.permute(0, 4, 1, 2, 3)

    scores = torch.zeros((x2.shape[0], batch_size, 1))
    dice_losses = torch.zeros((x2.shape[0], 1))

    for i in range(x2.shape[0]):
        xx = Variable(torch.from_numpy(x2[i])).float()
        xx_seg = Variable(torch.from_numpy(x2_seg[i])).float()
        xx = xx.permute(0, 4, 1, 2, 3)
        xx_seg = xx_seg.permute(0, 4, 1, 2, 3)

        score, l_bce, l_dice = model(x1.cuda(), x1_seg, y1, xx.cuda(), xx_seg, y2[i], mode='test')
        scores[i] = score.data
        dice_losses[i] = l_dice.data

    acc = 0.0
    for i in range(batch_size):
        s = scores[:, i]
        ind = torch.argmax(s)
        if y1[i] == y2[ind, i]:
            acc += 1

    return acc/batch_size, dice_losses.mean().numpy()