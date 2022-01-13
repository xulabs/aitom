import torch
from torch import nn as nn
import torch.nn.functional as F
class Disc(nn.Module):
    def __init__(self):
        super(Disc, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, 3, padding=1)
        self.conv5 = nn.Conv3d(64, 256, 3, padding=1)
        self.conv10 = nn.Conv3d(256, 512, 3, padding=1)
        self.fc13 = nn.Linear(512, 1)

    def forward(self, x):
        
        x = F.max_pool3d(F.leaky_relu((self.conv1(x))), kernel_size=4)
        x = F.max_pool3d(F.leaky_relu((self.conv5(x))), kernel_size=4)
        x = F.max_pool3d(F.leaky_relu(self.conv10((x))), kernel_size=2)

        x = x.view(-1, self.num_flat_features(x))
        x = torch.sigmoid(self.fc13(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
# Disc()(torch.rand(1,1,40,40,40)).shape
# sum(p.numel() for p in Disc().parameters())
