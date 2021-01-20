import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        # [75,6,1,256]
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # (n*b) x lq x dk
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        # (n*b) x lk x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        # (n*b) x lv x dv
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)
        # [75,6,256]
        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        # b x lq x (n*dv)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn, log_attn


class FEAT(nn.Module):
    def __init__(self, args, dropout=0.1):
        super().__init__()
        if args.model_type == 'ConvNet':
            from feat.networks.convnet import ConvNet
            self.encoder = ConvNet()
            z_dim = 512
            self.softmax = nn.Softmax(dim=1)
        elif args.model_type == 'ResNet':
            from feat.networks.resnet import ResNet
            self.encoder = ResNet()
            z_dim = 640
        else:
            raise ValueError('')

        self.slf_attn = MultiHeadAttention(1, z_dim, z_dim, z_dim, dropout=dropout)
        self.args = args

    def forward(self, support, query):
        # feature extraction
        # support [5,3,84,84]
        support = self.encoder(support)  # [5,256]
        support_proto = support.reshape(self.args.shot, self.args.way, -1).mean(dim=0)
        # get mean of the support
        proto = support.reshape(self.args.shot, -1, support.shape[-1]).mean(dim=0)  # N x d [5,256]
        num_proto = proto.shape[0]
        # for query set
        query = self.encoder(query)
        query_protp = query
        # [75, 256]
        # combine all query set with the proto
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
        refined_support, refined_query = combined.split(self.args.way, 1)
        # logitis = self.softmax(-torch.sum((refined_support - refined_query) ** 2, 2) / self.args.temperature + euclidean_metric(query_protp, support_proto) / self.args.temperature)
        # [75,5]
        logitis = -torch.sum((refined_support - refined_query) ** 2, 2) / self.args.temperature
        # logitis = euclidean_metric(query_protp, support_proto) / self.args.temperature
        # logitis = -torch.sum((refined_support - refined_query) ** 2, 2) / self.args.temperature

        return logitis, enc_slf_log_attn
