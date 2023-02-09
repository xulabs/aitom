from .data import data_loader, loadpickle
from .utils import *
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


def evaluate_model(dataset_name, model, z_dim, pixel, batch_size, device, scale):
    train_loader, test_loader, mu, std = data_loader(dataset_name, pixel, batch_size, shuffle=False)
    plot_sample_images(dataset_name, test_loader, model, pixel, batch_size, device, mu, std)
    generate_manifold_images(dataset_name, model, pixel, z_dim, batch_size, device, mu, std)
    save_output_images(dataset_name, test_loader, model, pixel, 'test', batch_size, device=device)
    save_latent_variables(dataset_name, test_loader, model, 'test', pixel, w)
    # score = calculate_score(dataset_name,model,z_dim,pixel,scale, batch_size,device)
    # print('DIC score for semnatic content',score)


def discrete_factor_prediction_score(latent_factor_train, ground_truth_train, latent_factor_test, ground_truth_test,
                                     cls):
    if cls == 'KNN' or cls == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    elif cls == 'RF' or cls == 'rf':
        model = RandomForestClassifier()
    else:
        model = LinearSVC()

    model.fit(latent_factor_train, ground_truth_train)
    return model.score(latent_factor_test, ground_truth_test)


def continuos_factor_prediction_score(latent_factor_train, ground_truth_train, latent_factor_test, ground_truth_test,
                                      cls):
    if cls == 'LR' or cls == 'lr':
        model = LinearRegression()
    else:
        model = SVR()
    model.fit(latent_factor_train, ground_truth_train)
    return model.score(latent_factor_test, ground_truth_test)


def calculate_score(dataset_name, model, z_dim, pixel, scale, batch_size, device):
    train_loader, test_loader = data_loader(dataset_name, pixel, batch_size, shuffle=False)
    save_latent_variables(dataset_name, train_loader, model, 'train', pixel, batch_size, device=device)
    save_latent_variables(dataset_name, test_loader, model, 'test', pixel, batch_size, device=device)
    train_phi = np.loadtxt('Harmony_latent_factors_' + dataset_name + '_train.np')
    test_phi = np.loadtxt('Harmony_latent_factors_' + dataset_name + '_test.np')
    if scale:
        trans_dim = 4
    else:
        trans_dim = 3
    train_semantic_z = train_phi[:, trans_dim:trans_dim + z_dim]
    test_semantic_z = test_phi[:, trans_dim:trans_dim + z_dim]

    train_labels = loadpickle('data/' + dataset_name + '_train_labels.pkl')
    test_labels = loadpickle('data/' + dataset_name + '_test_labels.pkl')
    return discrete_factor_prediction_score(train_semantic_z.reshape(-1, z_dim), train_labels,
                                            test_semantic_z.reshape(-1, z_dim), test_labels, 'knn')
