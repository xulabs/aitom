import torch
from torch import nn
from torch.nn import functional as F

from gaussian import gaussian3D
from utils import deform, Conv_block, decoder_general

# bottle = 64
# bottle=125

ns_lim = 0.1
alpha_m = 1
ds_lim = 0.01  ##Change carefully....Very sensitive
kernel = 7
bl_lim = 0.5

# ns_lim=0.0


device = "cuda"


class decoder_ds(nn.Module):
    def __init__(self, in_channels=128, bottle=125):
        super(decoder_ds, self).__init__()

        self.c1 = nn.Conv3d(
            in_channels=in_channels, out_channels=1, kernel_size=1, stride=1
        )
        self.f1 = nn.Sequential(nn.Linear(bottle, 1), nn.Sigmoid())

    def forward(self, x, orig, mask):

        global alpha_m, ds_lim, device
        alpha = torch.flatten(self.c1(x["encode_ds"]), start_dim=1)
        # print(alpha.shape)
        alpha = self.f1(alpha) * alpha_m
        # print(alpha)
        def_, mask = deform(orig.shape, kernel_size=kernel, device=device).to(device)(
            x, orig, alpha, mask, ds_lim=ds_lim
        )
        # def_ = def_.repeat(1, 3, 1, 1)
        # print(def_.shape)
        return def_, mask


class decoder_ns(nn.Module):
    def __init__(self):

        super(decoder_ns, self).__init__()
        # channel_list_ns = [128, 128, 128, 128, 128, 1]#, 128, 64, 64, 64, 64, 1]
        # channel_list_ns = [128, 128, 64, 64, 64, 1]#, 128, 64, 64, 64, 64, 1]
        channel_list_ns = [128, 128, 128, 128, 128, 128, 128, 64, 64, 64, 64, 1]
        self.dec = nn.Sequential(
            decoder_general(channel_list_ns, name="ns"),
            nn.Tanh(),
        )
        # self.up=nn.Upsample(scale_factor=2,mode="nearest")

    def forward(self, x):
        global ns_lim
        x = self.dec(x) * ns_lim

        # x=(x-0.5)*2.0*ns_lim
        # x=self.up(x*ns_lim)

        return x


class decoder_bl(nn.Module):
    def __init__(self, channels=1):
        super(decoder_bl, self).__init__()
        global device, bl_lim
        self.bl_lim = bl_lim
        channel_list_ns = [128, 128, 128, 128, 128, 128, 128, 64, 64, 64, 32, 1]
        self.dec = nn.Sequential(
            decoder_general(channel_list_ns, name="bl"),
            nn.Sigmoid(),
        )
        kernel_size = 5
        self.padding = (kernel_size - 1) // 2
        self.stride = 1
        kernel = torch.tensor(gaussian3D(1, 0, 1, kernel_size)).to(device)
        self.device = device
        kernel = kernel.view(1, 1, *kernel.size()).to(device)
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)).to(device).float()
        # print(kernel.shape)
        # self.register_buffer('weight', kernel)
        self.weight = kernel
        self.conv = F.conv3d

    def forward(self, x, orig):

        x = self.dec(x)
        x = x * self.bl_lim

        dx = self.conv(
            orig.to(device),
            weight=self.weight,
            stride=self.stride,
            padding=self.padding,
        )
        return orig * (1 - x) + (dx * (x))


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        global padding_, kernel_

        self.d0 = Conv_block(1, 64)
        self.d1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            Conv_block(64, 64),
            Conv_block(64, 128),
        )

        self.d2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            Conv_block(128, 128),
            Conv_block(128, 128),
        )

        self.d3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2), Conv_block(128, 128 * 3)
        )

    def forward(self, x):
        x_0 = self.d0(x)
        x_1 = self.d1(x_0)
        x_2 = self.d2(x_1)
        x_3 = self.d3(x_2)

        len_ = x_3.shape[1]
        store = {
            "encode_ds": x_3[:, : len_ // 3, :, :, :],
            "encode_ns": x_3[:, len_ // 3 : len_ * 2 // 3, :, :, :],
            "encode_bl": x_3[:, len_ * 2 // 3 :, :, :, :],
            # "encode_ds": x_3[:, : len_ // 2, :, :, :],
            # "encode_ns": x_3[:, len_// 2 :, :, :, :],
            "0": x_0,
            "1": x_1,
            "2": x_2,
        }

        return store


class encode_warp(nn.Module):
    def __init__(self, shape=False):
        super(encode_warp, self).__init__()
        self.encoder = encoder()
        # channel_list_ns = [256, 128, 128, 128, 128, 128, 128, 64, 64, 64, 64, 1]
        if not shape:
            self.decoder_1 = decoder_ds()
        else:
            self.decoder_1 = decoder_ds(bottle=64)

        self.decoder_3 = decoder_bl()
        self.decoder_2 = decoder_ns()
        self.re = nn.Hardtanh(0, 1)

    def forward(self, x, mask):
        # print(x.size())
        x_en = self.encoder(x)  ## [2,128,5,5,5]
        dec0 = self.decoder_3(x_en, x)
        dec2 = self.re(self.decoder_2(x_en) + x)
        dec1, mask = self.decoder_1(x_en, dec2, mask)
        return dec1, mask
        # return out_avg,dec2['loss'],dec2
        # return dec2


if __name__ == "__main__":
    from sim_Demo import get_sample_data

    from mrc_np import wrap

    ar, ar_2 = get_sample_data(1)
    from scipy.ndimage import zoom

    # print(ar.shape)
    # ar=zoom(ar,(1,0.8,0.8,0.8))
    # ar_2=zoom(ar_2,(0.8,0.8,0.8))
    in_ = torch.tensor(ar).repeat(2, 1, 1, 1).unsqueeze(1).float().to(device)

    mask_ = torch.tensor(ar_2).repeat(2, 1, 1, 1).float().to(device)
    enc = encode_warp().to(device)

    encoded, mask = enc(in_, mask_)

    # wrap(encoded[0][0].detach().cpu().numpy(), "out.jpg")
    # wrap(mask_[0].detach().cpu().numpy(), "in_mask.jpg")
    # wrap(in_[0][0].detach().cpu().numpy(), "in.jpg")
    # wrap(mask[0][0].detach().cpu().numpy(), "out_mask.jpg")
    print(encoded.shape, mask.shape)
