import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
from nets import SCNN, DENSECNN


class SiameseNetwork(nn.Module):
    def __init__(self, keep_prob=0.2, num_channels=1, learning_rate=1e-3, tom_size=32, network_type='scnn'):
        super(SiameseNetwork, self).__init__()
        self.keep_prob = keep_prob
        self.num_channels = num_channels
        self.learning_rate = learning_rate
        self.tom_size = tom_size

        if network_type == 'scnn':
            self.net = SCNN(res=False, duse=False)
        elif network_type == 'dusescnn':
            self.net = SCNN(res=False, duse=True)
        else:
            raise NotImplementedError

        self.fc = nn.Linear(512, 1)

    def forward(self, x1, x1_seg, y1, x2, x2_seg, y2, mode='train'):
        x1_feat, x1_seg_pred = self.net(x1, mode)
        x2_feat, x2_seg_pred = self.net(x2, mode)

        # one shot learning
        score = torch.abs(x1_feat-x2_feat)
        score = self.fc(score)
        score = F.sigmoid(score)
        y = torch.zeros((y1.shape[0]))
        y = Variable(y)
        for b in range(y1.shape[0]):
            y[b] = 1 if y1[b] == y2[b] else 0
        l_bce = F.binary_cross_entropy(score.cpu(), y)
        self.score = score

        # segmentation learning
        l_dice_x1 = dice_loss(x1_seg.cpu(), x1_seg_pred.cpu())
        l_dice_x2 = dice_loss(x2_seg.cpu(), x2_seg_pred.cpu())
        l_dice = l_dice_x1 + l_dice_x2

        return score, l_bce, l_dice


def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
