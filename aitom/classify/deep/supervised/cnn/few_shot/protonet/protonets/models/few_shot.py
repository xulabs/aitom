import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from .factory import register_model
from .feat import MultiHeadAttention
from .utils import euclidean_dist
import numpy as np
import os


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()
        self.encoder = encoder
        self.slf_attn = MultiHeadAttention(1, 512, 512, 512, dropout=0)

    def loss(self, sample, stage, eval=False):
        # support
        xs = Variable(sample['xs'])
        # query
        xq = Variable(sample['xq'])
        classes = sample['class']

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)
        z = self.encoder.forward(x)

        z_dim = z.size(-1)
        z_proto = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class * n_support:]
        dists = euclidean_dist(zq, z_proto)
        # print(self.encoder)
        if stage == 'feat':
            # get mean of the support
            proto = z_proto.detach()
            num_proto = proto.shape[0]
            # for query set
            # [75, 256]
            # combine all query set with the proto
            query = zq.detach()
            num_query = query.shape[0]
            proto = proto.unsqueeze(0).repeat([num_query, 1, 1])  # NK x N x d[75,5,256]
            query = query.unsqueeze(1)  # NK x 1 x d
            combined = torch.cat([proto, query], 1)  # Nk x (N + 1) x d, batch_size = NK
            # [75, 6, 64] 75 query, 6=5 support+1 query
            # refine by Transformer
            combined, enc_slf_attn, enc_slf_log_attn = self.slf_attn(combined, combined, combined)
            # [75, 6, 64] [75, 6, 6] [75, 6, 6]
            # compute distance for all batches

            # [75,5,64] [75,1,64]
            refined_support, refined_query = combined.split(n_class, 1)
            # [75,5]
            logitis = -torch.sum((refined_support - refined_query) ** 2, 2)
            dists = -dists
            # logitis = -10*torch.transpose(logitis.transpose(0,1)/torch.min(logitis,1)[0],0,1)
            # dists = -10*torch.transpose(dists.transpose(0,1)/torch.min(dists,1)[0],0,1)
            if not eval:
                log_p_y = F.log_softmax(logitis + dists, dim=1).view(n_class, n_query, -1)
            else:
                log_p_y = F.log_softmax(logitis + dists, dim=1).view(n_class, n_query, -1)
            loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
            _, y_hat = log_p_y.max(2)
            acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

            y_hat = y_hat.numpy()
            target_inds = target_inds.squeeze().numpy()
            class_acc = {}
            class_prec = {}
            class_count = {}

            for i in range(len(target_inds)):
                ind = classes[i]
                class_acc[ind] = (y_hat[i] == target_inds[i]).sum() / len(y_hat[i])
                dest = np.where(y_hat == i)
                class_count[ind] = len(dest[0])
                class_prec[ind] = (y_hat[dest] == target_inds[dest]).sum()
            prec_macro = 0
            prec_micro = 0
            count_micro = 0
            for k in class_prec.keys():
                if class_count[k] == 0:
                    prec_macro += 0
                else:
                    prec_macro += class_prec[k] / class_count[k]
                prec_micro += class_prec[k]
                count_micro += class_count[k]
            prec_macro = prec_macro / len(class_prec.keys())
            prec_micro = prec_micro / count_micro

            return loss_val, {
                'loss': loss_val.item(),
                'acc': acc_val.item(),
                'prec_macro': prec_macro,
                'prec_micro': prec_micro
            }, enc_slf_attn, class_acc, class_count, class_prec

        elif stage == 'protonet':
            dists = -dists

            log_p_y = F.log_softmax(dists, dim=1).view(n_class, n_query, -1)
            loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
            _, y_hat = log_p_y.max(2)
            acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
            y_hat = y_hat.numpy()
            target_inds = target_inds.squeeze().numpy()

            class_acc = {}
            class_prec = {}
            class_count = {}

            for i in range(len(target_inds)):
                ind = classes[i]
                class_acc[ind] = (y_hat[i] == target_inds[i]).sum() / len(y_hat[i])
                dest = np.where(y_hat == i)
                class_count[ind] = len(dest[0])
                class_prec[ind] = (y_hat[dest] == target_inds[dest]).sum()
            prec_macro = 0
            prec_micro = 0
            count_micro = 0
            for k in class_prec.keys():
                if class_count[k] == 0:
                    prec_macro += 0
                else:
                    prec_macro += class_prec[k] / class_count[k]
                prec_micro += class_prec[k]
                count_micro += class_count[k]
            prec_macro = prec_macro / len(class_prec.keys())
            prec_micro = prec_micro / count_micro

            return loss_val, {
                'loss': loss_val.item(),
                'acc': acc_val.item(),
                'prec_macro': prec_macro,
                'prec_micro': prec_micro
            }, class_acc, class_count, class_prec


@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )

    return Protonet(encoder)
