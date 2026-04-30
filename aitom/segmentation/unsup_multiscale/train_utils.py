import torch, pdb
import torch.nn as nn
import numpy as np
import glob, os
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset

def off_diagonal(x):
    b, n, m = x.shape
    assert n == m
    return x.flatten(1,-1)[:, :-1].view(b, n - 1, n + 1)[:, :, 1:]

def build_matrix(affinity, feat, symmetric):
    #affinity: b * N * N
    #feat: b * c * N
    if symmetric == 1:
        A = affinity.exp()
        A = 0.5*(A.permute(0,2,1) + A)
        D = A.sum(dim = -1)
        D_sqrt = torch.sqrt(D)
        D_rsqrt = torch.rsqrt(D)
        # (D - W)
        A_diag_val =  A[:,torch.arange(A.shape[-1]), torch.arange(A.shape[-1])]
        D_minus_W = -1 * A
        D_minus_W[:,torch.arange(A.shape[-1]), torch.arange(A.shape[-1])] = D - A_diag_val
        # D^{-0.5}(D-W)D^{-0.5}
        matrix = (D_minus_W * D_rsqrt[:,:,None] * D_rsqrt[:,None,:])
        # b * c * n
        feat = feat * D_sqrt[:,None]
    else:
        matrix = affinity
    return matrix, feat

def ncut_loss(affinity, feat, symmetric = 0):
    #affinity: b * N * N
    # feat: b * c * N
    npixel = affinity.shape[-1]
    if feat.shape[2] * feat.shape[3] != npixel:
        affinity_img_size = int(np.sqrt(npixel))
        Z = torch.nn.AdaptiveAvgPool2d(output_size = affinity_img_size)(feat)
    else:
        Z = feat
    b,c,h,w = Z.shape
    Z = Z.reshape(b, c, -1)
    matrix, Z = build_matrix(affinity, Z, symmetric)
    Z = F.normalize(Z, dim = -1)
    # K * N * N
    #affinity: b * N * N
    # feat: b * c * N
    eigval = ((matrix @ Z[None,:].permute(0,1,3,2)) * Z.permute(0,2,1)[:,None]).sum(dim = 2)
    ortho = off_diagonal(Z @ Z.permute(0,2,1))
    return eigval, ortho.reshape(-1)

class ToyCNN(nn.Module):
    def __init__(self, feat_h, feat_w, num_of_eig, hidden_dim = 128):
        super().__init__()
        self.feat_var = torch.nn.parameter.Parameter(data = torch.randn(1, hidden_dim, feat_h, feat_w), requires_grad = True)
        self.fc1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size = 3, stride = 1, padding = 1)
        self.fc2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size = 3, stride = 1, padding = 1)
        self.fc3 = nn.Conv2d(hidden_dim, num_of_eig, kernel_size = 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

    def forward(self, input = None):
        feat = F.relu(self.bn1(self.fc1(self.feat_var)))
        feat = F.relu(self.bn2(self.fc2(feat)))
        feat = self.fc3(feat)
        return feat

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, img_transform, return_raw_rgb = False):
        if os.path.isdir(data_path):
            self.img_list = glob.glob(data_path + '/*.png') + \
                            glob.glob(data_path + '/*.jpeg') + \
                            glob.glob(data_path + '/*.jpg')
        else:
            self.img_list = [data_path]
        self.img_list.sort()
        self.img_transform = img_transform
        self.return_raw_rgb = return_raw_rgb

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB')
        if self.return_raw_rgb:
            return self.img_transform(img), np.array(img)
        else:
            return self.img_transform(img) 