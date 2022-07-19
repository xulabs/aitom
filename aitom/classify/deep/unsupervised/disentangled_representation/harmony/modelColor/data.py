import pickle
import torch
import numpy as np


def loadpickle(fname):
    with open(fname, 'rb') as f:
        array = pickle.load(f)
    f.close()
    return array


def estimate_optimal_gamma(dataset_name, batch_size):
    X_train = loadpickle('data/' + dataset_name + '_train.pkl')
    M = X_train.shape[0]
    N = batch_size * 1000
    return math.ceil(M / N)


def data_loader(dataset_name, pixel, batch_size=100, shuffle=True, normalize=False):
    X_train = loadpickle('data/' + dataset_name + '_train.pkl')
    X_test = loadpickle('data/' + dataset_name + '_test.pkl')
    if normalize:
        mu = X_train.reshape(-1, pixel * pixel * 3).mean(1)
        std = X_train.reshape(-1, pixel * pixel * 3).std(1)
        X_train = (X_train - mu[:, np.newaxis, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis, np.newaxis]
        mu = X_test.reshape(-1, pixel * pixel * 3).mean(1)
        std = X_test.reshape(-1, pixel * pixel * 3).std(1)
        X_test = (X_test - mu[:, np.newaxis, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis, np.newaxis]
    X_train = X_train.reshape(X_train.shape[0], 3, pixel, pixel)
    X_test = X_test.reshape(X_test.shape[0], 3, pixel, pixel)
    train_x = torch.from_numpy(X_train).float()
    test_x = torch.from_numpy(X_test).float()

    train_loader = torch.utils.data.DataLoader(train_x, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_x, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return train_loader, test_loader
