import torch
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv3d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )

        self.fc11 = nn.Linear(10240, 4096)
        self.fc12 = nn.Linear(4096, 4096)
        self.fc13 = nn.Linear(4096, 4)

        self.d = nn.Dropout(0.5)

    def forward(self, x):
        x = self.c1(x)

        x_feat = x.view(-1, self.num_flat_features(x))

        x = self.fc13(self.d(F.relu(self.fc12(self.d(F.relu(self.fc11(x_feat)))))))
        return x_feat, x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Disc_feat(nn.Module):
    def __init__(self):
        super(Disc_feat, self).__init__()
        self.fc11 = nn.Linear(10240, 4096)
        self.fc12 = nn.Linear(4096, 4096)
        self.fc13 = nn.Linear(4096, 1)
        self.d = nn.Dropout(0.5)

    def forward(self, x_feat):
        x = self.fc13(self.d(F.relu(self.fc12(self.d(F.relu(self.fc11(x_feat)))))))
        return torch.sigmoid(x)


if __name__ == "__main__":

    k = Net()
    print(k(torch.rand(1, 1, 40, 40, 40)).shape)
    print(sum(p.numel() for p in k.parameters()))