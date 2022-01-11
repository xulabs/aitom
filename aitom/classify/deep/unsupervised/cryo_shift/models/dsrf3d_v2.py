import torch
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv3d(32, 32, 3, padding=1)

        self.conv3 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv3d(64, 64, 3, padding=1)

        self.conv5 = nn.Conv3d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv3d(128, 128, 3, padding=1)

        self.fc11 = nn.Linear(16000, 1024)
        self.fc12 = nn.Linear(1024, 1024)
        self.fc13 = nn.Linear(1024, 4)

        self.d = nn.Dropout(0.7)

    def forward(self, x):
        x = F.max_pool3d(
            F.relu(self.conv2(F.relu(self.conv1(x)))), kernel_size=2, stride=2
        )
        x = F.max_pool3d(
            F.relu(self.conv4(F.relu(self.conv3(x)))), kernel_size=2, stride=2
        )
        x = F.max_pool3d(
            F.relu(self.conv6(F.relu(self.conv5(x)))), kernel_size=2, stride=2
        )

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
        self.fc11 = nn.Linear(16000, 4096)
        self.fc12 = nn.Linear(4096, 4096)
        self.fc13 = nn.Linear(4096, 1)
        self.d = nn.Dropout(0.7)

    def forward(self, x_feat):
        x = self.fc13(self.d(F.relu(self.fc12(self.d(F.relu(self.fc11(x_feat)))))))
        return torch.sigmoid(x)


if __name__ == "__main__":
    k = Net()
    print(k(torch.rand(2, 1, 40, 40, 40)).shape)
    print(sum(p.numel() for p in k.parameters()))
