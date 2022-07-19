from data import data_loader
from utils import *
import numpy as np
from scipy.stats import pearsonr


def evaluate_model(dataset_name, model, z_dim, pixel, batch_size, device):
    train_loader, test_loader = data_loader(dataset_name, batch_size, shuffle=False, normalize=False)
    plot_sample_images(dataset_name, test_loader, model, pixel, batch_size, device=device)
    generate_manifold_images(dataset_name, model, pixel, z_dim, batch_size, device=device)
    save_latent_variables(dataset_name, test_loader, model, 'test', batch_size, device=device)
    #score = calculate_score(dataset_name,model,z_dim,pixel,batch_size,device)
    #print('SAP score for semnatic content',score)

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


def calculate_score(dataset_name, model, z_dim, pixel, batch_size, device):
    train_loader, test_loader = data_loader(dataset_name, pixel, batch_size, shuffle=False)
    save_latent_variables(dataset_name, train_loader, model, 'train', pixel, batch_size, device=device)
    save_latent_variables(dataset_name, test_loader, model, 'test', pixel, batch_size, device=device)
    train_phi = np.loadtxt('Harmony_latent_factors_' + dataset_name + '_train.np')
    test_phi = np.loadtxt('Harmony_latent_factors_' + dataset_name + '_test.np')
    train_semantic_z = train_phi[:, -z_dim:]
    test_semantic_z = test_phi[:, -z_dim:]
    train_labels = loadpickle('data/' + dataset_name + '_train_label.pkl')
    test_labels = loadpickle('data/' + dataset_name + '_test_label.pkl')
    return discrete_factor_prediction_score(train_semantic_z.reshape(-1, z_dim), train_labels,
                                            test_semantic_z.reshape(-1, z_dim), test_labels, 'knn')
