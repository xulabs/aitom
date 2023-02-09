import numpy as np
import torch
import kornia
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, latent_dims, pixel):
        self.pixel = pixel
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)  # 7 x 7
        self.conv3 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)  # 4 x 4
        self.conv4 = nn.Conv3d(128, 512, kernel_size=4, stride=1, padding=0)  # 1 x 1
        self.conv5 = nn.Conv3d(512, 2 * latent_dims + 6, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = F.tanh(self.conv4(x))
        x = self.conv5(x)
        phi = x.view(x.size(0), -1)
        return phi


class ConvDecoder(nn.Module):
    def __init__(self, latent_dims, pixel):
        super(ConvDecoder, self).__init__()
        self.activation = nn.Sigmoid()
        self.pixel = pixel
        self.lt = latent_dims
        self.conv = nn.ConvTranspose3d(latent_dims, 512, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.ConvTranspose3d(512, 128, kernel_size=4, stride=1, padding=0)
        self.conv2 = nn.ConvTranspose3d(128, 64, kernel_size=4, padding=1, stride=2)
        self.conv3 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = z.view(z.size(0), self.lt, 1, 1, 1)
        x = F.tanh(self.conv(x))
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = self.conv4(x)
        x = x.view(x.size(0), 1, self.pixel, self.pixel, self.pixel)
        return x


class Transformer(object):
    def __call__(self, image, theta, translations):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        B, C, D, H, W = image.size()
        center = torch.tensor([D / 2, H / 2, W / 2]).repeat(B, 1).to(device=device)
        scale = torch.ones(B, 1).to(device=device)
        angle = torch.rad2deg(theta)
        no_trans = torch.zeros(B, 3).to(device=device)
        no_rot = torch.zeros(B, 3).to(device=device)
        M = kornia.get_affine_matrix3d(translations=no_trans, center=center, scale=scale, angles=angle)
        affine_matrix = M[:, :3, :]
        rotated_image = kornia.warp_affine3d(image, affine_matrix, dsize=(D, H, W), align_corners=False,
                                             padding_mode='zeros')
        N = kornia.get_affine_matrix3d(translations=translations, center=center, scale=scale, angles=no_rot)
        affine_matrix_tran = N[:, :3, :]
        transformed_image = kornia.warp_affine3d(rotated_image, affine_matrix_tran, dsize=(D, H, W),
                                                 align_corners=False, padding_mode='zeros')
        return transformed_image


class AutoEncoder(nn.Module):
    def __init__(self, latent_dims, pixel):
        super(AutoEncoder, self).__init__()
        self.encoder = ConvEncoder(latent_dims, pixel)
        self.decoder = ConvDecoder(latent_dims, pixel)
        self.transform = Transformer()
        self.latent_dims = latent_dims

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x):
        phi = self.encoder(x)
        theta = phi[:, :3]
        trans = phi[:, 3:6]
        z_mu = phi[:, 6:6 + self.latent_dims]
        z_var = phi[:, -self.latent_dims:]
        z = self.reparametrize(z_mu, z_var)
        image_z = self.decoder.forward(z)
        image_x_theta = self.transform(x, theta, trans)
        return image_z, image_x_theta, phi


class Siamese(nn.Module):
    def __init__(self, latent_dims, pixel):
        super(Siamese, self).__init__()
        self.autoencoder = AutoEncoder(latent_dims, pixel)
        self.transform = Transformer()

    def forward(self, image_z):
        with torch.no_grad():
            angles = torch.FloatTensor(image_z.size(0), 3).uniform_(-np.pi / 2, np.pi / 2).to(device=image_z.device)
            translations = torch.FloatTensor(image_z.size(0), 3).uniform_(-4, 4).to(device=image_z.device)
            transformed_image = self.transform(image_z, angles, translations)
        image_z1, image_x_theta1, phi1 = self.autoencoder(image_z)
        image_z2, image_x_theta2, phi2 = self.autoencoder(transformed_image)
        return image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2


def load_ckp(model, optimizer=None, f_path='./best_model.pt'):
    checkpoint = torch.load(f_path)

    model.load_state_dict(checkpoint['state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer'])

    valid_loss_min = checkpoint['valid_loss_min']
    epoch_train_loss = checkpoint['epoch_train_loss']
    epoch_valid_loss = checkpoint['epoch_valid_loss']

    return model, optimizer, checkpoint['epoch'], epoch_train_loss, epoch_valid_loss, valid_loss_min


def save_ckp(state, f_path='./best_model.pt'):
    torch.save(state, f_path)


def get_instance_model_optimizer(device, learning_rate=0.0001, z_dims=2, pixel=64):
    model = Siamese(latent_dims=z_dims, pixel=pixel).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer
