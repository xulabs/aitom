import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule
import numpy as np


class DCD(BasicModule):
    def __init__(self, opt):
        super(DCD, self).__init__()
        h_features = 64
        input_features = 2 * opt['classifier_input_dim']
        self.fc1 = nn.Linear(input_features, h_features)
        self.fc2 = nn.Linear(h_features, h_features)
        self.fc3 = nn.Linear(h_features * 2, h_features)
        self.fc4 = nn.Linear(h_features, 4)

    def forward(self, inputs, conv_inputs):
        hidden = F.relu(self.fc1(inputs))
        hidden = F.relu(self.fc2(hidden))
        X_cat = torch.cat([hidden, conv_inputs], 1)
        hidden = F.relu(self.fc3(X_cat))
        return F.softmax(self.fc4(hidden), dim=1)


class CONV_DCD(BasicModule):
    def __init__(self, opt):
        super(CONV_DCD, self).__init__()
        hid_dim = opt['encoder_hid_dim'] * 2
        z_dim = opt['encoder_z_dim']
        h_features = 64
        self.conv1 = conv_block(hid_dim, hid_dim)
        self.conv2 = conv_block(hid_dim, z_dim)
        self.flatten = Flatten()
        self.fc1 = nn.Linear(opt['classifier_input_dim'], h_features)
        # self.fc2=nn.Linear(h_features,4)

    def forward(self, input):
        hidden = self.conv1(input)
        out = self.flatten(self.conv2(hidden))
        # hidden=F.relu(self.fc1(hidden))
        # out=self.fc2(hidden)
        # return F.softmax(out,dim=1)
        return self.fc1(out)


class Classifier(BasicModule):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(opt['classifier_input_dim'], opt['classes_num'])

    def forward(self, input):
        return F.softmax(self.fc(input), dim=1)


class Encoder2(BasicModule):
    def __init__(self, opt):
        super(Encoder2, self).__init__()
        hid_dim = opt['encoder_hid_dim']
        z_dim = opt['encoder_z_dim']
        self.conv1 = conv_block(hid_dim, hid_dim)
        self.conv2 = conv_block(hid_dim, z_dim)
        self.flatten = Flatten()

    def forward(self, input):
        hidden = self.conv1(input)
        return hidden, self.flatten(self.conv2(hidden))


class Encoder1(BasicModule):
    def __init__(self, opt):
        super(Encoder1, self).__init__()
        hid_dim = opt['encoder_hid_dim']
        z_dim = opt['encoder_z_dim']

        self.encoder = nn.Sequential(
            conv_block(1, hid_dim),
            conv_block(hid_dim, hid_dim),
        )

    def forward(self, input):
        return self.encoder(input)


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(),
        nn.MaxPool3d(2)
    )


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def prototype_loss(zq, z_proto):
    n_class = z_proto.size(0)
    n_query = zq.size(0)

    dists = euclidean_dist(zq, z_proto.squeeze(1))

    log_p_y = F.log_softmax(-dists, dim=1).view(n_query, n_class)
    return log_p_y
    # loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    # return loss_val


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):  # [75,6,256]
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head  # 256,256,1
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)  # [75,6,1,256]
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv
        # [75,6,256]
        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn, log_attn
