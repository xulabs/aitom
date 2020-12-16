import torch
import torch.nn as nn
import torch.nn.functional as F
from .models.BasicModule import BasicModule
import numpy as np


def CORAL(X_s, X_t, device):
    dim = X_s.shape[1]
    n_s = X_s.shape[0]
    ones = torch.ones([1, n_s]).to(device)
    ones = ones.mm(X_s)

    Cs = (X_s.transpose(0, 1).mm(X_s) - ones.transpose(0, 1).mm(ones) / n_s) / (n_s - 1)

    n_t = X_t.shape[0]
    ones = torch.ones([1, n_t]).to(device)
    ones = ones.mm(X_t)

    Ct = (X_t.transpose(0, 1).mm(X_t) - ones.transpose(0, 1).mm(ones) / n_t) / (n_t - 1)

    loss = torch.norm(Cs - Ct) / 4 / dim / dim
    return loss
