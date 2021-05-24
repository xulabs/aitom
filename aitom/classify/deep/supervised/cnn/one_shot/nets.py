import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class SCNN(nn.Module):
    def __init__(self, tom_size=32, in_chan=1, res=False, duse=False):
        super(SCNN, self).__init__()

        # encoder
        self.e0_1 = nn.Conv3d(in_chan, 64, kernel_size=3, stride=1, padding=1)

        self.e1_1 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.ebn1_1 = nn.BatchNorm3d(64)
        self.e1_2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.ebn1_2 = nn.BatchNorm3d(64)
        self.eduse1 = ChannelSpatialSELayer3D(num_channels=64)
        self.mp1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.e2_1 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.ebn2_1 = nn.BatchNorm3d(64)
        self.e2_2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.ebn2_2 = nn.BatchNorm3d(64)
        self.eduse2 = ChannelSpatialSELayer3D(num_channels=64)
        self.mp2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.e3_1 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.ebn3_1 = nn.BatchNorm3d(64)
        self.e3_2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.ebn3_2 = nn.BatchNorm3d(64)

        # decoder
        self.d2 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.d2_1 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dbn2_1 = nn.BatchNorm3d(64)
        self.d2_2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dbn2_2 = nn.BatchNorm3d(64)
        self.dduse2 = ChannelSpatialSELayer3D(num_channels=64)

        self.d1 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.d1_1 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dbn1_1 = nn.BatchNorm3d(64)
        self.d1_2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dbn1_2 = nn.BatchNorm3d(64)
        self.dduse1 = ChannelSpatialSELayer3D(num_channels=64)

        self.d_last = nn.Conv3d(64, 1, kernel_size=1)

        # bottle neck fc layer for one-shot classification
        fs = int(math.floor(tom_size/(2*2)))
        linear_in_size = 64*fs*fs*fs
        self.l3 = nn.Sequential(
            nn.Linear(linear_in_size, 512),
            nn.Linear(512, 512)
        )
        self.outSize = 512

        self.res = res
        self.duse = duse

    def forward(self, in_tom, mode):
        # encoding
        e0 = self.e0_1(in_tom)
        e0 = F.relu(e0, inplace=True)

        e1 = self.e1_1(e0)
        e1 = F.relu(e1, inplace=True)
        e1 = self.ebn1_1(e1)
        e1 = self.e1_2(e1)
        e1 = F.relu(e1, inplace=True)
        e1 = self.ebn1_2(e1)
        if self.duse:
            e1 = self.eduse1(e1)
        if self.res:
            e1 = e1 + e0
        e1_down = self.mp1(e1)

        e2 = self.e2_1(e1_down)
        e2 = F.relu(e2, inplace=True)
        e2 = self.ebn2_1(e2)
        e2 = self.e2_2(e2)
        e2 = F.relu(e2, inplace=True)
        e2 = self.ebn2_2(e2)
        if self.duse:
            e2 = self.eduse1(e2)
        if self.res:
            e2 = e2 + e1_down
        e2_down = self.mp2(e2)

        e3 = self.e3_1(e2_down)
        e3 = F.relu(e3, inplace=True)
        e3 = self.ebn3_1(e3)
        e3 = self.e3_2(e3)
        e3 = F.relu(e3, inplace=True)
        e3 = self.ebn3_2(e3)
        if self.res:
            e3 = e3 + e2_down

        # decoding
        d2 = self.d2(e3)
        d2 = self.d2_1(d2)
        d2 = F.relu(d2, inplace=True)
        d2 = self.dbn2_1(d2)
        d2 = self.d2_2(d2)
        d2 = F.relu(d2, inplace=True)
        d2 = self.dbn2_2(d2)
        if self.duse:
            d2 = self.dduse2(d2)

        d1 = self.d1(d2)
        d1 = self.d1_1(d1)
        d1 = F.relu(d1, inplace=True)
        d1 = self.dbn1_1(d1)
        d1 = self.d1_2(d1)
        d1 = F.relu(d1, inplace=True)
        d1 = self.dbn1_2(d1)
        if self.duse:
            d1 = self.dduse1(d1)

        d_out = self.d_last(d1)
        d_out = F.sigmoid(d_out)

        # bottle neck output
        e4 = e3.view(e3.size()[0], -1)
        e4 = self.l3(e4)

        return e4, d_out


class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2, norm='None'):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.norm = norm
        self.bn = nn.BatchNorm3d(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        if self.norm == 'BN':
            output_tensor = self.bn(output_tensor)


        return output_tensor


class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels, norm = 'None'):
        """
        :param num_channels: No of input channels

        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.norm = norm
        self.bn = nn.BatchNorm3d(1)

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

            if self.norm == 'BN':
                out = self.bn(out)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):
    """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, num_channels, reduction_ratio=2, norm='None'):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio, norm=norm)
        self.sSE = SpatialSELayer3D(num_channels, norm=norm)
        self.norm = norm

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor