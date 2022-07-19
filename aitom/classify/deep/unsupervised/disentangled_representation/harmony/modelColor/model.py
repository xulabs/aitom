import numpy as np
import kornia  # version 0.4.0
import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dims, pixel):
        super(Encoder, self).__init__()
        self.conv1 = self.my_conv_layer(3, 32, 4, 2, 1)
        self.conv2 = self.my_conv_layer(32, 32, 4, 2, 1)
        self.conv3 = self.my_conv_layer(32, 64, 4, 2, 1)
        self.conv4 = self.my_conv_layer(64, 64, 4, 2, 1)
        self.conv5 = self.my_conv_layer(64, 128, 4, 2, 1)
        self.conv6 = self.my_conv_layer(128, 512, 4, 1, 0)
        self.conv7 = nn.Conv2d(512, 1 + 2 * latent_dims, 1, 1, 0)
        self.latent_dims = latent_dims

    def my_conv_layer(self, in_c, out_c, kernel, stride, padding):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_c),
        )
        return conv_layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        phi = x.view(x.size(0), 1 + 2 * self.latent_dims)
        return phi


class Transformer(object):
    def __call__(self, image, contrast_factor, flag=0):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        B, C, H, W = image.size()
        cf = contrast_factor.squeeze()
        if flag == 0:
            cf = F.hardtanh(cf, 0.8, 1.5)
        contrast_adjusted = kornia.adjust_contrast(image, contrast_factor=cf)
        return contrast_adjusted


class Decoder(nn.Module):
    def __init__(self, latent_dims, pixel):
        super(Decoder, self).__init__()
        self.rev_conv1 = self.my_rev_conv_layer(latent_dims, 512, 1, 1, 0)
        self.rev_conv2 = self.my_rev_conv_layer(512, 128, 4, 1, 0)
        self.rev_conv3 = self.my_rev_conv_layer(128, 64, 4, 2, 1)
        self.rev_conv4 = self.my_rev_conv_layer(64, 64, 4, 2, 1)
        self.rev_conv5 = self.my_rev_conv_layer(64, 32, 4, 2, 1)
        self.rev_conv6 = self.my_rev_conv_layer(32, 32, 4, 2, 1)
        self.rev_conv = nn.ConvTranspose2d(32, 3, 4, 2, 1)
        self.activation = nn.Sigmoid()
        self.pixel = pixel

    def my_rev_conv_layer(self, in_c, out_c, kernel, stride, padding):
        rev_conv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_c)
        )
        return rev_conv_layer

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        x = self.rev_conv1(z)
        x = self.rev_conv2(x)
        x = self.rev_conv3(x)
        x = self.rev_conv4(x)
        x = self.rev_conv5(x)
        x = self.rev_conv6(x)
        x = self.rev_conv(x)
        x = self.activation(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, latent_dims, pixel):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(latent_dims, pixel)
        self.decoder = Decoder(latent_dims, pixel)
        self.transform = Transformer()
        self.latent_dims = latent_dims

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x):
        phi = self.encoder(x)
        cf = phi[:, 0]
        z_mu = phi[:, 1:1 + self.latent_dims]
        z_var = phi[:, -self.latent_dims:]
        z = self.reparametrize(z_mu, z_var)
        image_z = self.decoder.forward(z)
        image_x_theta = self.transform(x, cf, flag=0)
        return image_z, image_x_theta, phi


class Siamese(nn.Module):
    def __init__(self, latent_dims, pixel):
        super(Siamese, self).__init__()
        self.autoencoder = AutoEncoder(latent_dims, pixel)
        self.transform = Transformer()

    def forward(self, image, z_scale=1):
        with torch.no_grad():
            contrast_factor = torch.FloatTensor(image.size(0), 1).uniform_(1.0, 1.5).to(device=image.device)
            transformed_image = self.transform(image, contrast_factor, flag=1)
        image_z1, image_x_theta1, phi1 = self.autoencoder(image)
        image_z2, image_x_theta2, phi2 = self.autoencoder(transformed_image)
        return image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2


def load_ckp(model, optimizer=None, f_path='./best_model.pt'):
    # load check point
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
