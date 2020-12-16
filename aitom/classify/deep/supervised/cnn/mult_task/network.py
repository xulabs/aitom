# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
import numpy as np


class SSN3DED(nn.Module):
    def __init__(self, mode):
        super(SSN3DED, self).__init__()
        self.mode = mode

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # concat conv
        self.cat1 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.cat2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)

        # upsampling
        self.deconv3 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2),
            nn.Conv3d(128, 128, kernel_size=3, padding=1))

        self.deconv2 = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.Conv3d(64, 64, kernel_size=3, padding=1))

        self.deconv1 = nn.Sequential(
            nn.Conv3d(128, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2),
            nn.Conv3d(32, 2, kernel_size=3, padding=1))

        self.softmax = nn.Softmax(dim=1)    # (NCDHW)
        self.loss = nn.BCEWithLogitsLoss()  # nn.CrossEntropyLoss()

    def forward(self, *input):
        if len(input) == 2:
            x = input[0]
            tar = input[1]
        if len(input) == 1:
            x = input[0]
            tar = None
        # encoding
        layer1 = self.conv1(x)
        down1 = self.pool1(layer1)

        layer2 = self.conv2(down1)
        down2 = self.pool2(layer2)

        layer3 = self.conv3(down2)
        down3 = self.pool1(layer3)

        # decoding
        layer_3 = self.deconv3(down3)
        cat3 = torch.cat((self.cat2(down2), layer_3), 1)

        layer_2 = self.deconv2(cat3)
        cat2 = torch.cat((self.cat1(down1), layer_2), 1)

        layer_1 = self.deconv1(cat2)
        output = self.softmax(layer_1)

        loss = 0
        if self.mode == 'train':
            loss = self.loss(output[:, 0], tar[:, 0])
        return output, loss
