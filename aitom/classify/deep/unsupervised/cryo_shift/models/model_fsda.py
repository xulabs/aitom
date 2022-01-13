import torch.nn as nn
import torch
from torch.nn import functional as F


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(),
        nn.MaxPool3d(2),
    )


class FSDA(nn.Module):
    def __init__(self, in_channels=1):
        super(FSDA, self).__init__()

        self.classifier = nn.Linear(1024, 4)
        self.encoder_1 = nn.Sequential(
            conv_block(in_channels, 16),
            conv_block(16, 16),
        )
        self.encoder_2 = nn.Sequential(
            conv_block(16, 16), conv_block(16, 128), nn.Flatten()
        )

    def forward(self, x):
        hidden = self.encoder_2(self.encoder_1(x))
        return hidden, self.classifier(hidden)


class Disc_feat(nn.Module):
    def __init__(self):
        super(Disc_feat, self).__init__()
        self.fc12 = nn.Linear(1024, 512)
        self.fc13 = nn.Linear(512, 1)
        self.d = nn.Dropout(0.5)

    def forward(self, x_feat):
        x = self.fc13(self.d(F.relu(self.fc12(x_feat))))
        return torch.sigmoid(x)


if __name__ == "__main__":
    m = FSDA()
    print(m(torch.rand(2, 1, 40, 40, 40)).shape)
    print(sum(p.numel() for p in m.parameters()))