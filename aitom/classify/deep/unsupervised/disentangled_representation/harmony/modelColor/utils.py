import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
import numpy as np
from model import load_ckp
from scipy.stats import norm


def loss_fn(image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2, dim=1, w=1):
    n = image_x_theta1.size(0)
    recon_loss1 = F.mse_loss(image_z1, image_x_theta1, reduction="sum").div(n)
    recon_loss2 = F.mse_loss(image_z2, image_x_theta2, reduction="sum").div(n)
    branch_loss = F.mse_loss(image_x_theta1, image_x_theta2, reduction="sum").div(n)
    z1_mean = phi1[:, 1:1 + dim]
    z1_var = phi1[:, -dim:]
    z2_mean = phi2[:, 1:1 + dim]
    z2_var = phi2[:, -dim:]
    dist_z1 = torch.distributions.multivariate_normal.MultivariateNormal(z1_mean, torch.diag_embed(z1_var.exp()))
    dist_z2 = torch.distributions.multivariate_normal.MultivariateNormal(z2_mean, torch.diag_embed(z2_var.exp()))
    z_loss = torch.mean(torch.distributions.kl.kl_divergence(dist_z1, dist_z2)).div(dim)
    loss = w * (recon_loss1 + recon_loss2) + branch_loss + z_loss
    return loss


def plot_loss(epoch_train_loss, epoch_valid_loss):
    fig, ax = plt.subplots(dpi=150)
    train_loss_list = [x for x in epoch_train_loss]
    valid_loss_list = [x for x in epoch_valid_loss]
    line1, = ax.plot([i for i in range(len(train_loss_list))], train_loss_list)
    line2, = ax.plot([i for i in range(len(valid_loss_list))], valid_loss_list)
    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('Loss')
    ax.legend((line1, line2), ('Train Loss', 'Validation Loss'))
    plt.savefig("Harmony_loss_curves.png", bbox_inches="tight")


def _save_sample_images(dataset_name, batch_size, recon_image, image, pixel):
    sample_out = recon_image.reshape(batch_size, pixel, pixel, 3)
    plt.clf()
    fig = plt.figure(figsize=(8, 8))  # Notice the equal aspect ratio
    plot_per_row = int(batch_size / 10)
    plot_per_col = int(batch_size / 10)
    ax = [fig.add_subplot(plot_per_row, plot_per_col, i + 1) for i in range(batch_size)]

    i = 0
    for a in ax:
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
        a.set_aspect('equal')
        a.imshow(sample_out[i])
        i += 1

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("Harmony_decoded_image_sample_" + dataset_name + ".png", bbox_inches="tight")

    sample_in = image.reshape(batch_size, pixel, pixel, 3)
    plt.clf()
    fig = plt.figure(figsize=(8, 8))  # Notice the equal aspect ratio
    plot_per_row = int(batch_size / 10)
    plot_per_col = int(batch_size / 10)
    ax = [fig.add_subplot(plot_per_row, plot_per_col, i + 1) for i in range(batch_size)]

    i = 0
    for a in ax:
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
        a.set_aspect('equal')
        a.imshow(sample_in[i])
        i += 1

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("Harmony_input_image_sample" + dataset_name + ".png", bbox_inches="tight")


def generate_manifold_images(dataset_name, trained_vae, pixel, z_dim=1, batch_size=100, device='cuda'):
    trained_vae.eval()
    decoder = trained_vae.autoencoder.decoder
    if z_dim>2:
        print("Generation of manifold image for higher than 2-dimension is not implemented in this version")
        print("Manifold images not saved")
        return
    elif z_dim == 1:
        z_arr = norm.ppf(np.linspace(0.05, 0.95, batch_size))
    else:
        n = int(np.sqrt(batch_size))
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
        z_list = []
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                z_list.append(z_sample)
        z_arr = np.array(z_list).squeeze()

    z = torch.from_numpy(z_arr).float().to(device=device)
    if z_dim == 1:
        z = torch.unsqueeze(z, 1)
    image_z = decoder(z)
    manifold = image_z.cpu().detach()
    sample_out = manifold.reshape(batch_size, pixel, pixel, 3)
    plt.clf()
    fig = plt.figure(figsize=(8, 8))  # Notice the equal aspect ratio
    plot_per_row = int(batch_size / 10)
    plot_per_col = int(batch_size / 10)
    ax = [fig.add_subplot(plot_per_row, plot_per_col, i + 1) for i in range(batch_size)]

    i = 0
    for a in ax:
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
        a.set_aspect('equal')
        a.imshow(sample_out[i])
        i += 1

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("Harmony_manifold_image_" + dataset_name + ".png", bbox_inches="tight")


def save_latent_variables(dataset_name, data_loader, siamese, type, batch_size=100, device='cuda'):
    Allphi = []
    siamese.eval()
    count = 0
    for batch_idx, images in enumerate(data_loader):
        count += 1
        with torch.no_grad():
            images = images.to(device=device)
            data = images.reshape(batch_size, 3, 128, 128)
            image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2 = siamese(data)
            phi_np = phi1.cpu().detach().numpy()
            Allphi.append(phi_np)
    PhiArr = np.array(Allphi).reshape(count * batch_size, -1)
    filepath = 'Harmony_latent_factors_' + dataset_name + '_' + type + '.pkl'
    np.savetxt(filepath, PhiArr)
