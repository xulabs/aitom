# Pumpkin: PU learning-based Macromolecule PicKINg in cryo-electron tomograms

# Import Statements
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler

import numpy as np
import scipy.stats as S
from sklearn.metrics import average_precision_score, matthews_corrcoef
import pandas as pd

import math
import sys
import os
import shutil
import mrcfile
import gc
import random

from abc import ABC, abstractmethod
from tqdm import tqdm
import argparse

# Method Architecture Implementation
class BasicEncoder(nn.Module):
    def __init__(self, subtomogram_size):
        super().__init__()
        self.encoder, self.latent_dim = self.__get_encoder(subtomogram_size)

    def __get_encoder(self, subtomogram_size):
        kernel_dims = None
        in_channels, out_channels, channel_scaling = 1, 32, 2
        layers = []

        if subtomogram_size == 16:
            kernel_dims = [8, 5]
        elif subtomogram_size == 32:
            kernel_dims = [8, 5, 5]

        for kernel_dim in kernel_dims[:-1]:
            layers += [nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_dim, stride=2, bias=False)]
            layers += [nn.BatchNorm3d(num_features=out_channels)]
            layers += [nn.PReLU()]

            in_channels = out_channels
            out_channels = out_channels * channel_scaling

        layers += [nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_dims[-1], bias=False)]
        layers += [nn.BatchNorm3d(num_features=out_channels)]
        layers += [nn.PReLU()]
        return nn.Sequential(*layers), out_channels

    def forward(self, x):
        return self.encoder(x)

class GaussianDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        assert 0 <= dropout < 1

        self.dropout = dropout
        self.std = math.sqrt(dropout / (1 - dropout))

    def forward(self, x):
        if self.training and self.dropout > 0:
            means = torch.ones_like(input=x)
            gaussian_noises = torch.normal(mean=means, std=self.std)
            x = torch.mul(input=x, other=gaussian_noises)
        return x

class BasicDecoder(nn.Module):
    def __init__(self, subtomogram_size, latent_dim, add_bias=True):
        super().__init__()
        self.decoder = self.__get_decoder(subtomogram_size, latent_dim, add_bias)

    def __get_decoder(self, subtomogram_size, latent_dim, add_bias):
        kernel_dims = None
        in_channels, out_channels, channel_scaling = latent_dim, latent_dim, 2
        layers = []

        if subtomogram_size == 16:
            kernel_dims = [4, 4, 4]
        elif subtomogram_size == 32:
            kernel_dims = [4, 4, 4, 4]

        layers += [nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_dims[0], bias=add_bias)]
        layers += [nn.BatchNorm3d(num_features=out_channels)]
        layers += [nn.LeakyReLU()]

        for kernel_dim in kernel_dims[1:-1]:
            in_channels = out_channels
            out_channels = out_channels // channel_scaling

            layers += [nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_dim, stride=2, padding=1, bias=add_bias)]
            layers += [nn.BatchNorm3d(num_features=out_channels)]
            layers += [nn.LeakyReLU()]
        else:
            in_channels = out_channels
            out_channels = 1

        layers += [nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_dims[-1], stride=2, padding=1, bias=add_bias)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convolution_unit1 = self.__get_convolution_unit(in_channels=in_channels, out_channels=out_channels)
        self.convolution_unit2 = self.__get_convolution_unit(in_channels=out_channels, out_channels=out_channels)

    def __get_convolution_unit(self, in_channels, out_channels):
        layers = [nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm3d(num_features=out_channels)]
        layers += [nn.ReLU()]
        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.convolution_unit1(x)
        return self.convolution_unit2(y)

class ResUNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convolution_unit1 = self.__get_convolution_unit(in_channels=in_channels, out_channels=out_channels)
        self.convolution_unit2 = self.__get_convolution_unit(in_channels=out_channels, out_channels=out_channels)
        self.conv3d_layer = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.batch_norm3d_layer = nn.BatchNorm3d(num_features=out_channels)
        self.relu_layer = nn.ReLU()

    def __get_convolution_unit(self, in_channels, out_channels):
        layers = [nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm3d(num_features=out_channels)]
        layers += [nn.ReLU()]
        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.convolution_unit1(x)
        x = self.convolution_unit2(y)
        x = self.conv3d_layer(x)
        x = self.batch_norm3d_layer(x)
        return self.relu_layer(x + y)

class UNetCoder(nn.Module):
    def __init__(self, subtomogram_size, use_resunet=True, use_dropout=False):
        super().__init__()
        self.maxpool3d_layer, self.dropout_layer = nn.MaxPool3d(kernel_size=2, stride=2), nn.Dropout() if use_dropout else None
        self.down_unet_block1 = ResUNetBlock(in_channels=1, out_channels=32) if use_resunet else UNetBlock(in_channels=1, out_channels=32)
        self.down_unet_block2 = ResUNetBlock(in_channels=32, out_channels=64) if use_resunet else UNetBlock(in_channels=32, out_channels=64)
        self.down_unet_block3 = ResUNetBlock(in_channels=64, out_channels=128) if use_resunet else UNetBlock(in_channels=64, out_channels=128)
        self.down_unet_block4, self.deconvolution_unit3, self.up_unet_block3 = None, None, None

        if subtomogram_size == 32:
            self.down_unet_block4 = ResUNetBlock(in_channels=128, out_channels=256) if use_resunet else UNetBlock(in_channels=128, out_channels=256)
            self.deconvolution_unit3 = self.__get_deconvolution_unit(in_channels=256, out_channels=128)
            self.up_unet_block3 = ResUNetBlock(in_channels=256, out_channels=128) if use_resunet else UNetBlock(in_channels=256, out_channels=128)

        self.deconvolution_unit2 = self.__get_deconvolution_unit(in_channels=128, out_channels=64)
        self.up_unet_block2 = ResUNetBlock(in_channels=128, out_channels=64) if use_resunet else UNetBlock(in_channels=128, out_channels=64)
        self.deconvolution_unit1 = self.__get_deconvolution_unit(in_channels=64, out_channels=32)
        self.up_unet_block1 = ResUNetBlock(in_channels=64, out_channels=32) if use_resunet else UNetBlock(in_channels=64, out_channels=32)
        self.conv3d_layer = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1)

    def __get_deconvolution_unit(self, in_channels, out_channels):
        layers = [nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)]
        layers += [nn.BatchNorm3d(num_features=out_channels)]
        layers += [nn.ReLU()]
        return nn.Sequential(*layers)

    def forward(self, x):
        down_unet_block_output1 = self.down_unet_block1(x)
        down_unet_maxpool_output1 = self.maxpool3d_layer(down_unet_block_output1)
        down_unet_block_output2 = self.down_unet_block2(down_unet_maxpool_output1)
        down_unet_maxpool_output2 = self.maxpool3d_layer(down_unet_block_output2)
        down_unet_block_output3 = self.down_unet_block3(down_unet_maxpool_output2)

        if self.down_unet_block4 is None:
            down_unet_output = self.dropout_layer(down_unet_block_output3) if self.dropout_layer is not None else down_unet_block_output3
        else:
            down_unet_maxpool_output3 = self.maxpool3d_layer(down_unet_block_output3)
            down_unet_block_output4 = self.down_unet_block4(down_unet_maxpool_output3)
            down_unet_output = self.dropout_layer(down_unet_block_output4) if self.dropout_layer is not None else down_unet_block_output4
            up_unet_deconvolution_output3 = self.deconvolution_unit3(down_unet_output)
            up_unet_concat_output3 = torch.cat(tensors=(down_unet_block_output3, up_unet_deconvolution_output3), dim=1)
            down_unet_output = self.up_unet_block3(up_unet_concat_output3)

        up_unet_deconvolution_output2 = self.deconvolution_unit2(down_unet_output)
        up_unet_concat_output2 = torch.cat(tensors=(down_unet_block_output2, up_unet_deconvolution_output2), dim=1)
        up_unet_block_output2 = self.up_unet_block2(up_unet_concat_output2)
        up_unet_deconvolution_output1 = self.deconvolution_unit1(up_unet_block_output2)
        up_unet_concat_output1 = torch.cat(tensors=(down_unet_block_output1, up_unet_deconvolution_output1), dim=1)
        up_unet_block_output1 = self.up_unet_block1(up_unet_concat_output1)
        return self.conv3d_layer(up_unet_block_output1)


class Pumpkin(nn.Module):
    def __init__(self, encoder_mode, subtomogram_size, use_decoder):
        super().__init__()

        if encoder_mode == "unet":
            self.sample_coder = UNetCoder(subtomogram_size=subtomogram_size)
        else:
            self.sample_coder = None

            if encoder_mode == "basic":
                self.sample_encoder = BasicEncoder(subtomogram_size)
            
            self.sample_decoder = BasicDecoder(subtomogram_size, self.sample_encoder.latent_dim, add_bias=False) if use_decoder else None
            self.sample_classifier = BasicDecoder(subtomogram_size, self.sample_encoder.latent_dim)

    def forward(self, x):
        if self.sample_coder is not None:
            return self.sample_coder(x), None
        else:
            # Features extraction from subtomogram samples using encoder
            z = self.sample_encoder(x)
            # Sample reconstruction from extracted features using decoder
            y = self.sample_decoder(z) if self.sample_decoder is not None else None
            # Sample classification with extracted features using classifier
            x = self.sample_classifier(z)
        return x, y

# Training Objectives Implementation
class Objective(ABC):
    def __init__(self, model, criteria, optimizer, pi, reconstruction_weight, l2_coefficient):
        self.model = model
        self.criteria = criteria
        self.optimizer = optimizer
        self.pi = pi
        self.recon_weight = reconstruction_weight
        self.l2_coefficient = l2_coefficient
        self.stats_headers = ["Loss", "Recon Error", "Precision", "TPR", "FPR"]

    def compute_score(self, features):
        features = torch.unsqueeze(features, dim=1)
        if self.model.sample_coder is not None:
            score = self.model.sample_coder(features).view(-1)
        else:
            extracted_features = self.model.sample_encoder(features)
            score = self.model.sample_classifier(extracted_features).view(-1)
        return score

    def compute_recon_loss(self, features):
        features = torch.unsqueeze(features, dim=1)
        extracted_features = self.model.sample_encoder(features)
        recon_features = self.model.sample_decoder(extracted_features)
        recon_loss = (features - recon_features) ** 2
        recon_loss = torch.mean(torch.sum(recon_loss.view(recon_loss.size(0), -1), dim=1))
        return recon_loss

    def compute_performance_metrics(self, score, labels):
        p_hat = torch.sigmoid(score)
        precision = torch.sum(p_hat[labels == 1]).item() / torch.sum(p_hat).item()
        tpr = torch.mean(p_hat[labels == 1]).item()
        fpr = torch.mean(p_hat[labels == 0]).item()
        return precision, tpr, fpr

    def compute_regularization_loss(self):
        regularization_loss = sum([torch.sum((weights ** 2)) for weights in self.model.sample_encoder.parameters()])
        regularization_loss += sum([torch.sum((weights ** 2)) for weights in self.model.sample_classifier.parameters()])
        regularization_loss = 0.5 * self.l2_coefficient * regularization_loss
        return regularization_loss

    @abstractmethod
    def step(self, features, labels):
        pass

class PN(Objective):
    def __init__(self, model, criteria, optimizer, pi, reconstruction_weight, l2_coefficient):
        super().__init__(model, criteria, optimizer, pi, reconstruction_weight, l2_coefficient)

    def step(self, features, labels):
        score = self.compute_score(features)
        labels = labels.view(-1)
        recon_loss = None

        if self.recon_weight > 0:
            recon_loss = self.compute_recon_loss(features)

        self.optimizer.zero_grad()

        if self.pi is not None:
            factor = 10*torch.sum(labels[labels == 1].float())/labels.size(0)
            positive_loss = factor*self.criteria(score[labels == 1], labels[labels == 1].float())
            negative_loss = self.criteria(score[labels == 0], labels[labels == 0].float())
            if not math.isnan(positive_loss):
                loss = positive_loss * self.pi + negative_loss * (1 - self.pi)
            else:
                loss = negative_loss * (1 - self.pi)
        else:
            loss = self.criteria(score, labels.float())

        if self.recon_weight > 0:
            loss = loss + recon_loss * self.recon_weight

        loss.backward()

        precision, tpr, fpr = self.compute_performance_metrics(score, labels)

        if self.l2_coefficient > 0:
            regularization_loss = self.compute_regularization_loss()
            regularization_loss.backward()

        self.optimizer.step()

        return loss.item(), recon_loss.item() if self.recon_weight > 0 else None, precision, tpr, fpr

class PU(Objective):
    def __init__(self, model, criteria, optimizer, pi, reconstruction_weight, l2_coefficient):
        super().__init__(model, criteria, optimizer, pi, reconstruction_weight, l2_coefficient)
        self.beta = 0

    def step(self, features, labels):
        score = self.compute_score(features)
        labels = labels.view(-1)
        recon_loss = None

        if self.recon_weight > 0:
            recon_loss = self.compute_recon_loss(features)

        self.optimizer.zero_grad()

        factor = 10*torch.sum(labels[labels == 1].float())/labels.size(0)
        loss_pp = factor*self.criteria(score[labels == 1], labels[labels == 1].float())
        loss_pn = factor*self.criteria(score[labels == 1], 0 * labels[labels == 1].float())
        loss_un = self.criteria(score[labels == 0], labels[labels == 0].float())
        if not math.isnan(loss_pn.item()):
            loss_u = loss_un - loss_pn * self.pi
        else:
            loss_u = loss_un 

        if loss_u.item() < -self.beta:
            loss = -loss_u
            backprop_loss = loss
            loss_u = -self.beta
            if not math.isnan(loss_pp.item()):
                loss = loss_pp * self.pi + loss_u
            else:
                loss = loss_u
        else:
            if not math.isnan(loss_pp):
                loss = loss_pp * self.pi + loss_u
            else:
                loss = loss_u
            backprop_loss = loss

        if self.recon_weight > 0:
            backprop_loss = backprop_loss + recon_loss * self.recon_weight

        backprop_loss.backward()

        precision, tpr, fpr = self.compute_performance_metrics(score, labels)

        if self.l2_coefficient > 0:
            regularization_loss = self.compute_regularization_loss()
            regularization_loss.backward()

        self.optimizer.step()

        return loss.item(), recon_loss.item() if self.recon_weight > 0 else None, precision, tpr, fpr

class GE_KL(Objective):
    def __init__(self, model, criteria, optimizer, pi, reconstruction_weight, l2_coefficient, slack):
        super().__init__(model, criteria, optimizer, pi, reconstruction_weight, l2_coefficient)
        self.slack = slack
        self.running_expectation = pi
        self.stats_headers += ["GE Penalty"]
        self.momentum = 1
        self.entropy_penalty = 0

    def step(self, features, labels):
        score = self.compute_score(features)
        labels = labels.view(-1)
        recon_loss = None

        if self.recon_weight > 0:
            recon_loss = self.compute_recon_loss(features)

        self.optimizer.zero_grad()

        factor = 100*torch.sum(labels[labels == 1].float())/labels.size(0)
        classifier_loss = factor*self.criteria(score[labels == 1], labels[labels == 1].float())

        p_hat = torch.mean(torch.sigmoid(score[labels == 0]))
        #if torch.sum(labels[labels == 1].float())>0:
        #    print(torch.sum(labels[labels == 1].float()), torch.mean(labels[labels == 1].float()), labels.size())
        if self.momentum < 1:
            p_hat = p_hat * self.momentum + self.running_expectation * (1 - self.momentum)
            self.running_expectation = p_hat.item()

        entropy = np.log(self.pi) * self.pi + np.log1p(-self.pi) * (1 - self.pi)
        ge_penalty = -torch.log(p_hat) * self.pi - torch.log1p(-p_hat) * (1 - self.pi) + entropy
        ge_penalty = ge_penalty * self.slack / self.momentum
        #print(classifier_loss.item())
        if not math.isnan(classifier_loss.item()):
            loss = classifier_loss + ge_penalty
        else:
            loss = ge_penalty
        if self.recon_weight > 0:
            loss = loss + recon_loss * self.recon_weight

        loss.backward()

        precision, tpr, fpr = self.compute_performance_metrics(score, labels)

        if self.l2_coefficient > 0:
            regularization_loss = self.compute_regularization_loss()
            regularization_loss.backward()

        self.optimizer.step()

        return loss.item(), recon_loss.item() if self.recon_weight > 0 else None, precision, tpr, fpr, ge_penalty.item()

# Utilities Implementation
def make_model(args):
    pumpkin = Pumpkin(encoder_mode=args.encoder_mode, subtomogram_size=args.subtomogram_size, use_decoder=args.use_decoder)
    pumpkin.train()
    return pumpkin

def load_model(args):
    pumpkin = Pumpkin(encoder_mode=args.encoder_mode, subtomogram_size=args.subtomogram_size, use_decoder=args.use_decoder)
    pumpkin.load_state_dict(torch.load(f=f"{args.model_path}/{args.model_name}.pt"))
    pumpkin.eval()
    return pumpkin

def load_tomogram(tomogram_path, clip_value=3):
    #assert mrcfile.validate(tomogram_path)
    with mrcfile.open(tomogram_path) as mrc_file:
        tomogram = mrc_file.data

    tomogram = (tomogram - np.mean(tomogram)) / (np.std(tomogram) + sys.float_info.epsilon)
    tomogram = np.clip(tomogram, a_min=-clip_value - sys.float_info.epsilon, a_max=clip_value + sys.float_info.epsilon)
    tomogram = (tomogram - np.mean(tomogram)) / (np.std(tomogram) + sys.float_info.epsilon)
    return tomogram

def make_particles_mask(mask_shape, coords, particle_radius):
    particles_mask = np.zeros(shape=mask_shape, dtype=np.uint8)
    threshold = particle_radius ** 2

    z_grid = np.arange(start=0, stop=mask_shape[0], dtype=np.int32)
    y_grid = np.arange(start=0, stop=mask_shape[1], dtype=np.int32)
    x_grid = np.arange(start=0, stop=mask_shape[2], dtype=np.int32)
    z_grid, y_grid, x_grid = np.meshgrid(z_grid, y_grid, x_grid, indexing='ij')

    for i in range(coords.shape[0]):
        z_coord, y_coord, x_coord = coords[i, 0], coords[i, 1], coords[i, 2]
        squared_distances = (z_grid - z_coord) ** 2 + (y_grid - y_coord) ** 2 + (x_grid - x_coord) ** 2
        particles_mask = particles_mask + (squared_distances <= threshold).astype(np.uint8)

    return np.clip(particles_mask, a_min=0, a_max=1)


def make_data(args):
    """
    Process a single tomogram, split it into training and validation subtomograms, 
    generate particle masks, and save the stats to CSV files.

    Args:
        tomograms_dir (str): Path to the directory containing tomogram files.
        coordinates_path (str): Path to the CSV file containing particle coordinates.
        data_dir (str): Path to the directory where the output data will be saved.
        subtomogram_size (int): Size of each subtomogram.
        particle_radius (int): Radius for particle masks.
    
    Returns:
        pd.DataFrame: DataFrame containing training subtomogram stats.
        pd.DataFrame: DataFrame containing validation subtomogram stats.
    """
    
    # Load tomogram file and its path
    tomogram_names, tomogram_paths = [], []
    for tomogram_file_name in os.listdir(args.tomograms_dir):
        tomogram_name, _ = os.path.splitext(tomogram_file_name)
        tomogram_names.append(tomogram_name)
        tomogram_paths.append(args.tomograms_dir + os.sep + tomogram_file_name)

    # Assuming there is only one tomogram
    tomogram_name = tomogram_names[0]
    tomogram_path = tomogram_paths[0]

    # Load tomogram data (you'll need a function to load the tomogram, this is a placeholder)
    tomogram = load_tomogram(tomogram_path=tomogram_path)

    # Load coordinates
    coordinates = pd.read_csv(args.coordinates_path)

    # Filter coordinates for the current tomogram
    coords = coordinates.loc[coordinates["tomogram_name"] == int(tomogram_name.split('_')[0])]

    num_train_particles = args.num_train_particles

    # Split coordinates into training (first 100) and validation (remaining)
    train_coords = coords.iloc[:num_train_particles]
    test_coords = coords.iloc[num_train_particles:]

    # Prepare directories for saving data
    if os.path.exists(args.data_dir):
        shutil.rmtree(args.data_dir)

    train_subtomograms_dir = args.data_dir + os.sep + "Train" + os.sep + "Subtomograms"
    train_submasks_dir = args.data_dir + os.sep + "Train" + os.sep + "Labels"
    test_subtomograms_dir = args.data_dir + os.sep + "Test" + os.sep + "Subtomograms"
    test_submasks_dir = args.data_dir + os.sep + "Test" + os.sep + "Labels"

    os.makedirs(train_subtomograms_dir)
    os.makedirs(train_submasks_dir)
    os.makedirs(test_subtomograms_dir)
    os.makedirs(test_submasks_dir)

    # Prepare tomogram shape and padding
    z_dim = math.ceil(tomogram.shape[0] / args.subtomogram_size)
    y_dim = math.ceil(tomogram.shape[1] / args.subtomogram_size)
    x_dim = math.ceil(tomogram.shape[2] / args.subtomogram_size)
    tomogram_shape = (z_dim * args.subtomogram_size, y_dim * args.subtomogram_size, x_dim * args.subtomogram_size)

    padded_tomogram = np.zeros(shape=tomogram_shape, dtype=np.float32)
    padded_tomogram[:tomogram.shape[0], :tomogram.shape[1], :tomogram.shape[2]] = tomogram

    del tomogram
    gc.collect()

    tomogram = padded_tomogram

    # Create particle masks for training
    train_coords = train_coords[(0 < train_coords.z_coord) & (train_coords.z_coord < tomogram_shape[0]) & 
                                (0 < train_coords.y_coord) & (train_coords.y_coord < tomogram_shape[1]) & 
                                (0 < train_coords.x_coord) & (train_coords.x_coord < tomogram_shape[2])]
    train_coords = train_coords[["z_coord", "y_coord", "x_coord"]].to_numpy(dtype=np.int32)
    train_particles_mask = make_particles_mask(mask_shape=tomogram_shape, coords=train_coords, particle_radius=args.particle_radius)

    # Create particle masks for validation
    test_coords = test_coords[(0 < test_coords.z_coord) & (test_coords.z_coord < tomogram_shape[0]) & 
                            (0 < test_coords.y_coord) & (test_coords.y_coord < tomogram_shape[1]) & 
                            (0 < test_coords.x_coord) & (test_coords.x_coord < tomogram_shape[2])]
    test_coords = test_coords[["z_coord", "y_coord", "x_coord"]].to_numpy(dtype=np.int32)
    test_particles_mask = make_particles_mask(mask_shape=tomogram_shape, coords=test_coords, particle_radius=args.particle_radius)

    # Initialize lists for subtomogram stats
    train_subtomogram_names = []
    test_subtomogram_names = []
    train_positive_fractions = []
    test_positive_fractions = []

    # Process and save training subtomograms
    for k in range(z_dim):
        for j in range(y_dim):
            for i in range(x_dim):
                subtomogram = tomogram[k * args.subtomogram_size:(k + 1) * args.subtomogram_size, 
                                       j * args.subtomogram_size:(j + 1) * args.subtomogram_size, 
                                       i * args.subtomogram_size:(i + 1) * args.subtomogram_size]
                subtomogram_name = f"{tomogram_name}-subtomo-{k}_{j}_{i}"
                
                submask_train = train_particles_mask[k * args.subtomogram_size:(k + 1) * args.subtomogram_size, 
                                                     j * args.subtomogram_size:(j + 1) * args.subtomogram_size, 
                                                     i * args.subtomogram_size:(i + 1) * args.subtomogram_size]
                submask_name_train = f"{tomogram_name}-labels-{k}_{j}_{i}"
                
                # Calculate positive fraction for training data
                train_positive_fraction = np.sum(submask_train) / submask_train.size
                train_positive_fractions.append(train_positive_fraction)
                train_subtomogram_names.append(subtomogram_name)
                
                # Save train subtomogram and mask
                with open(f"{train_subtomograms_dir}{os.sep}{subtomogram_name}.npy", 'wb') as npy_file:
                    np.save(file=npy_file, arr=subtomogram)

                with open(f"{train_submasks_dir}{os.sep}{submask_name_train}.npy", 'wb') as npy_file:
                    np.save(file=npy_file, arr=submask_train)

                # Validation subtomogram and mask
                submask_test = test_particles_mask[k * args.subtomogram_size:(k + 1) * args.subtomogram_size, 
                                                 j * args.subtomogram_size:(j + 1) * args.subtomogram_size, 
                                                 i * args.subtomogram_size:(i + 1) * args.subtomogram_size]
                submask_name_test = f"{tomogram_name}-labels-{k}_{j}_{i}"
                
                # Calculate positive fraction for validation data
                test_positive_fraction = np.sum(submask_test) / submask_test.size
                if test_positive_fraction>0.0:
                    test_positive_fractions.append(test_positive_fraction)
                    test_subtomogram_names.append(subtomogram_name)
                
                    # Save validation subtomogram and mask
                    with open(f"{test_subtomograms_dir}{os.sep}{subtomogram_name}.npy", 'wb') as npy_file:
                        np.save(file=npy_file, arr=subtomogram)

                    with open(f"{test_submasks_dir}{os.sep}{submask_name_test}.npy", 'wb') as npy_file:
                        np.save(file=npy_file, arr=submask_test)

    # Create DataFrames to save subtomogram statistics
    train_subtomogram_stats = pd.DataFrame({
        "subtomogram_name": train_subtomogram_names, 
        "subtomogram_size": [args.subtomogram_size] * len(train_subtomogram_names), 
        "particle_radius": [args.particle_radius] * len(train_subtomogram_names), 
        "positive_fraction": train_positive_fractions
    })

    test_subtomogram_stats = pd.DataFrame({
        "subtomogram_name": test_subtomogram_names, 
        "subtomogram_size": [args.subtomogram_size] * len(test_subtomogram_names), 
        "particle_radius": [args.particle_radius] * len(test_subtomogram_names), 
        "positive_fraction": test_positive_fractions
    })

    # Save stats to CSV
    train_subtomogram_stats.to_csv(args.data_dir + os.sep + "Train" + os.sep + "train_subtomogram_stats.csv", index=False)
    test_subtomogram_stats.to_csv(args.data_dir + os.sep + "Test" + os.sep + "test_subtomogram_stats.csv", index=False)

    # Clean up
    del tomogram, train_particles_mask
    gc.collect()

    return train_subtomogram_stats, test_subtomogram_stats


def load_data(args):
    train_sample_stats = pd.read_csv(args.data_dir + os.sep + "Train" + os.sep + "train_subtomogram_stats.csv")
    train_sample_stats = train_sample_stats[["subtomogram_name", "positive_fraction"]].values.tolist()
    train_sample_stats = [[s_s[0].split('-')[0], s_s[0].split('-')[2], float(s_s[1])] for s_s in train_sample_stats]

    test_sample_stats = pd.read_csv(args.data_dir + os.sep + "Test" + os.sep + "test_subtomogram_stats.csv")
    test_sample_stats = test_sample_stats[["subtomogram_name", "positive_fraction"]].values.tolist()
    test_sample_stats = [[s_s[0].split('-')[0], s_s[0].split('-')[2], float(s_s[1])] for s_s in test_sample_stats]

    random.shuffle(train_sample_stats)
    random.shuffle(test_sample_stats)

    train_tomogram_names = [s_s[0] for s_s in train_sample_stats]
    train_tomogram_names = set(train_tomogram_names)
    num_train_tomograms = len(train_tomogram_names)

    
    num_total_particles = args.num_expected_particles * num_train_tomograms
    grid_line = np.linspace(start=-args.particle_radius, stop=args.particle_radius, num=2 * args.particle_radius + 1)

    z_grid = np.zeros(shape=(2 * args.particle_radius + 1, 2 * args.particle_radius + 1, 2 * args.particle_radius + 1)) + grid_line[:, np.newaxis, np.newaxis]
    y_grid = np.zeros(shape=(2 * args.particle_radius + 1, 2 * args.particle_radius + 1, 2 * args.particle_radius + 1)) + grid_line[np.newaxis, :, np.newaxis]
    x_grid = np.zeros(shape=(2 * args.particle_radius + 1, 2 * args.particle_radius + 1, 2 * args.particle_radius + 1)) + grid_line[np.newaxis, np.newaxis, :]

    grid_space = z_grid ** 2 + y_grid ** 2 + x_grid ** 2
    grid_mask = (grid_space <= args.particle_radius ** 2).astype(np.uint8)

    positive_cells_per_particle = np.sum(grid_mask)
    total_positive_cells = num_total_particles * positive_cells_per_particle
    # Divide the total number of expected positive voxels by the total number of voxels
    expected_pi = total_positive_cells / (args.subtomogram_size ** 3 * len(train_sample_stats))
    observed_pi = np.mean([s_s[2] for s_s in train_sample_stats])

    pi = expected_pi

    if expected_pi <= observed_pi:
        args.objective_type = "PN"
        pi = observed_pi
    elif args.objective_type in ["GE-KL"]:
        pi = expected_pi - observed_pi

    return train_sample_stats, test_sample_stats, num_train_tomograms, pi

def make_objective(args, model, pi):
    criteria = nn.BCEWithLogitsLoss()
    optimizer = Adam(params=model.parameters(), lr=args.init_lr)
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="max", factor=args.factor, patience=args.patience, min_lr=args.min_lr)

    if args.objective_type == "PN":
        objective = PN(model=model, criteria=criteria, optimizer=optimizer, pi=pi, reconstruction_weight=args.reconstruction_weight, l2_coefficient=args.l2_coefficient)
    elif args.objective_type == "PU":
        objective = PU(model=model, criteria=criteria, optimizer=optimizer, pi=pi, reconstruction_weight=args.reconstruction_weight, l2_coefficient=args.l2_coefficient)
    elif args.objective_type == "GE-KL":
        objective = GE_KL(model=model, criteria=criteria, optimizer=optimizer, pi=pi, reconstruction_weight=args.reconstruction_weight, l2_coefficient=args.l2_coefficient, slack=args.slack)
    else:
        objective = None

    return objective, criteria, lr_scheduler


def fit_epoch(epoch, objective, train_dataloader, output):
    for iteration, (features, labels) in enumerate(train_dataloader):
        if torch.cuda.is_available():
            features, labels = features.cuda(), labels.cuda()

        metrics = objective.step(features, labels)
        #print('Epoch: ', epoch, ' Loss: ',metrics[0], 'GE loss: ', metrics[-1], torch.sum(features), torch.sum(labels))
        if iteration%500==0:
            output += '\n' + '\t'.join([str(epoch + 1), str(iteration + 1), "train"] + [str(metric) if metric is None else f"{metric:.5f}" for metric in metrics] + ['-'] * 2)
    return output

def test_model(model, criteria, test_dataloader):
    if model.training:
        model.eval()

    num_sample_points = loss = 0
    y_score, y_true = [], []

    with torch.no_grad():
        for features, labels in test_dataloader:
            features = torch.unsqueeze(features, dim=1)
            labels = labels.view(-1)
            y_true.append(labels.numpy())

            if torch.cuda.is_available():
                features, labels = features.cuda(), labels.cuda()

            score, _ = model(features)
            score = score.view(-1)

            y_score.append(score.cpu().numpy())
            running_loss = criteria(score, labels.float()).item()
            num_sample_points = num_sample_points + labels.size(0)
            delta = labels.size(0) * (running_loss - loss)
            loss = loss + delta / num_sample_points

    y_score = np.concatenate(y_score, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    y_hat = 1 / (1 + np.exp(-y_score))
    y_predicted = (y_hat >= 0.5).astype(y_true.dtype)

    precision = np.sum(y_hat[y_true == 1]) / np.sum(y_hat)
    tpr = np.mean(y_hat[y_true == 1]) #tpr is recall 
    fpr = np.mean(y_hat[y_true == 0])
    f1_score = (2*precision*tpr)/(precision+tpr)
    auprc = average_precision_score(y_true=y_true, y_score=y_hat)
    #mcc = matthews_corrcoef(y_true=y_true, y_pred=y_predicted)

    return loss, precision, tpr, fpr, f1_score, auprc

def train_test_model(args, model, objective, criteria, lr_scheduler, train_dataloader, test_dataloader):
    stats_headers, args_dict = '\t'.join(["Epoch", "Iteration", "Split"] + objective.stats_headers + ["F1 Score","AUPRC"]), vars(args)
    logs = "-- Pumpkin Training & Testing --\n\n" + '\n'.join([f"{key}: {args_dict[key]}" for key in args_dict]) + '\n'

    max_test_f1 = -np.inf
    max_auprc = -np.inf
    for epoch in tqdm(iterable=range(args.num_epochs), desc="Training Progress", ncols=100, unit="epoch"):
        if not model.training:
            model.train()
            

        logs += '\n' + stats_headers

        logs = fit_epoch(epoch=epoch, objective=objective, train_dataloader=train_dataloader, output=logs)
        if epoch%1==0:
            loss, precision, tpr, fpr, f1_score, auprc = test_model(model=model, criteria=criteria, test_dataloader=test_dataloader)

            test_stats = '\t'.join([str(epoch + 1), '-', "test", f"{loss:.5f}", '-', f"{precision:.5f}", f"{tpr:.5f}", f"{fpr:.5f}"] + (['-'] if args.objective_type in ["GE-KL", "GE-binomial"] else []) + [f"{f1_score:.5f}", f"{auprc:.5f}"])
            logs += "\n\n" + stats_headers + '\n' + test_stats + '\n'

            if max_test_f1 < f1_score:
                logs += f"\nTest auprc score improved from {max_test_f1:.5f} to {f1_score:.5f}.\n"
                #max_auprc = auprc
                max_test_f1 = f1_score

                if not os.path.exists(args.model_path):
                    os.makedirs(args.model_path)

                torch.save(obj=model.state_dict(), f=f"{args.model_path}{os.sep}{args.model_name}.pt")
                logs += f"Updated model {args.model_name}.pt saved.\n"
            else:
                logs += f"\nTest AUPRC score did not improve from {max_test_f1:.5f}.\n"

            lr_scheduler.step(max_test_f1)
            logs += f"Current learning rate is {lr_scheduler._last_lr[0]}.\n"

    logs += "\nDone!"

    if not os.path.exists(args.logs_path):
        os.makedirs(args.logs_path)

    with open(args.logs_path + os.sep + args.model_name + "_train_test_logs.txt", 'w') as logs_file:
        logs_file.write(logs)

    return logs


class CustomSampleDataset(Dataset):
    def __init__(self, args, samples_stats, is_train=True):
        self.features, self.labels = [], []

        for sample_stats in samples_stats:
            self.features.append(sample_stats[0] + "-subtomo-" + sample_stats[1])
            self.labels.append(sample_stats[0] + "-labels-" + sample_stats[1])

        self.data_dir, self.augment_data = args.data_dir + os.sep + ("Train" if is_train else "Test"), args.augment_data

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        with open(f"{self.data_dir}{os.sep}Subtomograms{os.sep}{self.features[idx]}.npy", 'rb') as npy_file:
            subtomogram = np.load(file=npy_file)

        with open(f"{self.data_dir}{os.sep}Labels{os.sep}{self.labels[idx]}.npy", 'rb') as npy_file:
            submask = np.load(file=npy_file)

        if self.augment_data:
            choice_value = np.random.uniform()

            if choice_value < 0.3:
                num_rotations = np.random.randint(low=1, high=5)
                rotation_axes = np.random.choice(a=[0, 1, 2], size=(2,), replace=False)
                subtomogram = np.rot90(m=subtomogram, k=num_rotations, axes=rotation_axes).copy()
                submask = np.rot90(m=submask, k=num_rotations, axes=rotation_axes).copy()
            elif choice_value < 0.6:
                flip_axis = np.random.choice(a=[0, 1, 2])
                subtomogram = np.flip(m=subtomogram, axis=flip_axis).copy()
                submask = np.flip(m=submask, axis=flip_axis).copy()

        return subtomogram, submask


def make_train_dataloader(args, samples_stats):
    dataset = CustomSampleDataset(args=args, samples_stats=samples_stats)

    dataloader = DataLoader(dataset=dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)

    return dataloader

def make_test_dataloader(args, samples_stats):
    dataset = CustomSampleDataset(args=args, samples_stats=samples_stats, is_train=False)
    dataloader = DataLoader(dataset=dataset, batch_size=args.test_batch_size)
    return dataloader


def get_args():
    parser = argparse.ArgumentParser(description="Python script for training a Pumpkin model to pick macromolecules from cryo-electron tomograms.")
    metavar = 'X'

    parser.add_argument("--input", default=None, type=str, metavar=metavar, help="Path to a folder containing input subtomograms and submasks (Default: None)", dest="data_dir")
    parser.add_argument("--tomogram", default=None, type=str, metavar=metavar, help="Path to a folder containing sample tomograms for data generation (Default: None)", dest="tomograms_dir")
    parser.add_argument("--coord", default=None, type=str, metavar=metavar, help="Path to the particle coordinates file for data generation (Default: None)", dest="coordinates_path")

    parser.add_argument("--encoder", default="basic", type=str, metavar=metavar, help="Type of feature extractor (either basic or yopo) to use in network (Default: basic)", dest="encoder_mode")
    parser.add_argument("--decoder", action="store_true", help="Whether to use sample reconstructor in network (Default: False)", dest="use_decoder")

    parser.add_argument("--size", default=32, type=int, metavar=metavar, help="Size of subtomograms and submasks (either 16 or 32) in each dimension (Default: 16)", dest="subtomogram_size")
    parser.add_argument("--radius", default=12, type=int, metavar=metavar, help="Radius of a particle (in pixel) in sample tomograms (Default: 7)", dest="particle_radius")
    # parser.add_argument("--random", default=0.25, type=float, metavar=metavar, help="Percentage of randomly generated subtomograms and submasks (Default: 0.25)", dest="random_subdata_percentage")

    parser.add_argument("--split", default=1, type=int, metavar=metavar, help="Number of tomograms to randomly pick for testing (Default: 1)", dest="num_test_tomograms")
    parser.add_argument("--fraction", default=1.0, type=float, metavar=metavar, help="Percentage of samples with proper labeling for training (Default: 1.0)", dest="particles_fraction")
    parser.add_argument("--expect", default=2500, type=int, metavar=metavar, help="Expected average number of particles in each tomogram (Default: 1000)", dest="num_expected_particles")
    parser.add_argument("--train_particles", default=100, type=int, metavar=metavar, help="Number of particles to use for training (Default: 100)", dest="num_train_particles")

    parser.add_argument("--epoch", default=100, type=int, metavar=metavar, help="Number of training epochs (Default: 10)", dest="num_epochs")
    # We may no longer need --iter
    parser.add_argument("--iter", default=100, type=int, metavar=metavar, help="Number of weight updates in each training epoch (Default: 100)", dest="num_iterations")
    parser.add_argument("--train_batch", default=256, type=int, metavar=metavar, help="Number of samples in a training batch (Default: 256)", dest="train_batch_size")
    parser.add_argument("--test_batch", default=1, type=int, metavar=metavar, help="Number of samples in a testing batch (Default: 1)", dest="test_batch_size")

    parser.add_argument("--init_lr", default=1e-3, type=float, metavar=metavar, help="Initial lr used in training (Default: 2e-5)", dest="init_lr")
    parser.add_argument("--factor", default=0.5, type=float, metavar=metavar, help="Factor used in lr scheduling to adjust lr (Default: 0.5)", dest="factor")
    parser.add_argument("--patience", default=4, type=int, metavar=metavar, help="Patience used in lr scheduling to adjust lr (Default: 0)", dest="patience")
    parser.add_argument("--min_lr", default=1e-5, type=float, metavar=metavar, help="Minimum lr allowed in training (Default: 1e-7)", dest="min_lr")

    parser.add_argument("--name", default="tomopicker", type=str, metavar=metavar, help="Name of the model (Default: tomopicker)", dest="model_name")
    parser.add_argument("--save_weight", default=None, type=str, metavar=metavar, help="Path to a folder for saving model weights (Default: None)", dest="model_path")
    parser.add_argument("--save_log", default=None, type=str, metavar=metavar, help="Path to a folder for saving training and testing logs (Default: None)", dest="logs_path")

    parser.add_argument("--recon_weight", default=0.0, type=float, metavar=metavar, help="Weight on sample reconstruction error in loss calculation (Default: 0.0)", dest="reconstruction_weight")
    parser.add_argument("--l2_coeff", default=0.0, type=float, metavar=metavar, help="Weight on L2 regularization term (Default: 0.0)", dest="l2_coefficient")

    # We may need to omit GE-binomial for this work
    parser.add_argument("--objective", default="PU", type=str, metavar=metavar, help="Type of objective (any one of PN, PU, GE-KL or GE-binomial) to use in training (Default: PU)", dest="objective_type")
    # We may need to omit GE-binomial for this work
    parser.add_argument("--slack", default=None, type=float, metavar=metavar, help="Value of slack to use in GE-KL objective (Default: None)", dest="slack")

    parser.add_argument("--make", action="store_true", help="Whether to generate input dataset before training (Default: False)", dest="make_dataset")
    # We may no longer need --augment
    parser.add_argument("--augment", action="store_true", help="Whether to augment input dataset during training (Default: False)", dest="augment_data")

    parser.set_defaults(use_decoder=False)
    parser.set_defaults(make_dataset=False)
    parser.set_defaults(augment_data=False)
    # We may no longer need --augment
    
    args = parser.parse_args()

    encoder_modes = ["basic", "unet"]
    # We may need to omit GE-binomial for this work
    objective_types = ["PN", "PU", "GE-KL"]
    # We may need to omit GE-binomial for this work
    slack_configs = {"GE-KL": 10}

    assert args.encoder_mode in encoder_modes, "Invalid encoder_mode provided!"
    args.use_decoder = args.reconstruction_weight > 0

    #assert args.subtomogram_size == 16 or args.subtomogram_size == 32, "Invalid subtomogram_size provided!"
    assert args.particle_radius > 0, "Invalid particle_radius provided!"
    # assert args.random_subdata_percentage >= 0, "Invalid random_subdata_percentage provided!"

    assert args.num_test_tomograms > 0, "Invalid num_test_tomograms provided!"
    assert 0 < args.particles_fraction <= 1, "Invalid particles_fraction provided!"
    assert args.num_expected_particles > 0, "Invalid num_expected_particles provided!"

    assert args.reconstruction_weight >= 0, "Invalid reconstruction_weight provided!"
    assert args.l2_coefficient >= 0, "Invalid l2_coefficient provided!"

    assert args.objective_type in objective_types, "Invalid objective_type provided!"
    args.slack = slack_configs[args.objective_type] if args.objective_type in slack_configs else None

    return args

if __name__ == "__main__":
    # Processing command line arguments
    args = get_args()
    
    # Producing sample subtomograms and submasks (if necessary)
    if args.make_dataset:
        make_data(args=args)
    
    # Making an instance of Pumpkin class for training and testing
    pumpkin = make_model(args=args)
    
    if torch.cuda.is_available():
        pumpkin = pumpkin.cuda()
    
    # Loading samples_stats and fraction of positive regions in unlabeled samples (pi)
    train_samples_stats, test_samples_stats, num_train_tomograms, pi = load_data(args=args)
    print(f"\n#Train Samples = {len(train_samples_stats)}, #Test Samples = {len(test_samples_stats)}, #Train Tomograms = {num_train_tomograms}, Pi = {pi}\n")
    
    # Making objective function for training pipeline
    objective, criteria, lr_scheduler = make_objective(args=args, model=pumpkin, pi=pi)
    
    # Making training and testing dataloaders for training pipeline
    train_dataloader = make_train_dataloader(args=args, samples_stats=train_samples_stats)
    test_dataloader = make_test_dataloader(args=args, samples_stats=test_samples_stats)
    
    # Training and testing a model for picking particles from tomograms
    train_test_model(args=args, model=pumpkin, objective=objective, criteria=criteria, lr_scheduler=lr_scheduler, train_dataloader=train_dataloader, test_dataloader=test_dataloader)