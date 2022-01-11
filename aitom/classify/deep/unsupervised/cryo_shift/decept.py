import torch
from torch import nn
from torch.nn import functional as F
import numbers
import math
from gaussian import gaussian3D
from utils import deform, Conv_block_T, Conv_block, decoder_general

import numpy as np

ns_lim = 0.1
alpha_m = 1
ds_lim = 0.01  
kernel = 7


device = "cuda"


class decoder_ds(nn.Module):
    def __init__(self, in_channels=128, bottle=125):
        super(decoder_ds, self).__init__()

        self.c1 = nn.Conv3d(
            in_channels=in_channels, out_channels=1, kernel_size=1, stride=1
        )
        self.f1 = nn.Sequential(nn.Linear(bottle, 1), nn.Sigmoid())

    def forward(self, x, orig, mask):

        global alpha_m, ds_lim, device
        alpha = torch.flatten(self.c1(x["encode_ds"]), start_dim=1)
        # print(alpha.shape)
        alpha = self.f1(alpha) * alpha_m
        # print(alpha)
        def_, mask = deform(orig.shape, kernel_size=kernel, device=device).to(device)(
            x, orig, alpha, mask, ds_lim=ds_lim
        )
        return def_, mask


class decoder_ns(nn.Module):
    def __init__(self):

        super(decoder_ns, self).__init__()
        # channel_list_ns = [128, 128, 128, 128, 128, 1]#, 128, 64, 64, 64, 64, 1]
        # channel_list_ns = [128, 128, 64, 64, 64, 1]#, 128, 64, 64, 64, 64, 1]
        channel_list_ns = [128, 128, 128, 128, 128, 128, 128, 64, 64, 64, 64, 1]
        self.dec = nn.Sequential(
            decoder_general(channel_list_ns, name="ns"),
            nn.Tanh(),
        )
        # self.up=nn.Upsample(scale_factor=2,mode="nearest")

    def forward(self, x):
        global ns_lim
        x = self.dec(x) * ns_lim

        # x=(x-0.5)*2.0*ns_lim
        # x=self.up(x*ns_lim)

        return x



class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        global padding_, kernel_

        self.d0 = Conv_block(1, 64)
        self.d1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            Conv_block(64, 64),
            Conv_block(64, 128),
        )

        self.d2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            Conv_block(128, 128),
            Conv_block(128, 128),
        )

        self.d3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2), Conv_block(128, 128 * 2)
        )

    def forward(self, x):
        x_0 = self.d0(x)
        x_1 = self.d1(x_0)
        x_2 = self.d2(x_1)
        x_3 = self.d3(x_2)

        len_ = x_3.shape[1]
        store = {
            "encode_ds": x_3[:, : len_ // 2, :, :, :],
            "encode_ns": x_3[:, len_ // 2 : , :, :, :],
            "0": x_0,
            "1": x_1,
            "2": x_2,
        }

        return store


class encode_warp(nn.Module):
    def __init__(self, shape=False):
        super(encode_warp, self).__init__()
        self.encoder = encoder()
        if not shape:
            self.decoder_1 = decoder_ds()
        else:
            self.decoder_1 = decoder_ds(bottle=64)

        self.decoder_2 = decoder_ns()
        self.re = nn.Hardtanh(0.0, 1.0)

    def forward(self, x, mask):
        
        x_en = self.encoder(x)  ## [2,128,5,5,5]
        x = self.re(self.decoder_2(x_en) + x)
        x, mask = self.decoder_1(x_en, x, mask)
        return x, mask

