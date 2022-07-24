import torch
import numpy as np
from modelColor.model import get_instance_model_optimizer, load_ckp
from modelColor.data import data_loader
from modelColor.train import train_model
from modelColor.evaluate import evaluate_model
from modelColor.utils import *
import argparse


def train_and_evaluate(dataset_name, batch_size=100, n_epochs=5, learning_rate=0.0001, z_dim=2, pixel=64,
                       load_model=False, w=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    siamese, optimizer = get_instance_model_optimizer(device, learning_rate, z_dim, pixel)
    train_loader, test_loader = data_loader(dataset_name, pixel, batch_size)
    if load_model:
        siamese, optimizer, start_epoch, epoch_train_loss, epoch_valid_loss, valid_loss_min = load_ckp(siamese, optimizer, 'best_model_Harmony_' + dataset_name + '_z_dim_{}.pt'.format(z_dim))
    else:
        valid_loss_min = np.inf
        start_epoch = 0
        epoch_train_loss = []
        epoch_valid_loss = []

    train_model(siamese, optimizer, train_loader, test_loader, device, start_epoch, n_epochs, epoch_train_loss,
                epoch_valid_loss, valid_loss_min, z_dim, pixel, batch_size, w)

    evaluate_vae_model(dataset_name, siamese, z_dim, pixel, batch_size, device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train and Test Harmony')
    parser.add_argument('-z', '--z-dim', type=int, default=1)
    parser.add_argument('-bs', '--batch-size', type=int, default=100)
    parser.add_argument('-ep', '--num-epochs', type=int, default=2)
    parser.add_argument('-l', '--learning-rate', type=float, default=0.0001)
    parser.add_argument('--load-model', action='store_true')
    parser.add_argument('-dat', '--dataset', type=str)
    parser.add_argument('-p', '--pixel', type=int, default=64)
    parser.add_argument('-w', '--gamma', type=int)
    args = parser.parse_args()
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    z_dim = args.z_dim
    dataset_name = args.dataset
    pixel = args.pixel
    load_model = False
    pixel = 128
    if args.load_model:
        load_model = True

    if args.gamma:
        w = args.gamma
    else:
        w = estimate_optimal_gamma(dataset_name, batch_size)

    train_and_evaluate(dataset_name=dataset_name, batch_size=batch_size, n_epochs=num_epochs,
                       learning_rate=learning_rate, z_dim=z_dim, pixel=pixel, load_model=load_model, w=w)