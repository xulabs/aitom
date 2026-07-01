import torch
from torch import Tensor
from numpy import clip

from utils.utils import save_image
import logging
import os

logging.basicConfig(level=logging.INFO)

def generate_gaussian_noise(shape: torch.Size, device: torch.device = "cpu") -> Tensor:
    return torch.randn(shape, dtype=torch.float32, device=device)


def train_data_process_dsm(batch, cfg, device) -> dict:
    noisy_images = batch["noisy_image"]
    noisy_images = noisy_images.to(torch.float32).to(device)
    noisy_images = noisy_images.permute(0, 3, 1, 2).contiguous() # [B,C,H,W]

    added_noise = generate_gaussian_noise(shape=noisy_images.size(), device=device)

    noiser_images = noisy_images + added_noise * cfg.additional_sigmas[0]

    return dict(noiser_images=noiser_images, added_noise=added_noise)

def train_data_process_tsm(batch, cfg, device) -> dict:
    """
    Preprocess data for TSM training.
    Generates inputs based on:
        x' = x / sqrt(sigma^2 + sigma_a^2) + sigma_a
    and random noise I ~ N(0,1).
    """
    # --- Load and normalize input image ---
    noisy_images = batch["noisy_image"]
    noisy_images = noisy_images.to(torch.float32).to(device)
    noisy_images = noisy_images.permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]

    # --- Gaussian noise I ~ N(0,1) ---
    added_noise = generate_gaussian_noise(shape=noisy_images.size(), device=device)

    # --- Extract σ_a and σ (clean sigma) ---
    sigma_a = cfg.additional_sigmas[0]
    sigma_clean = getattr(cfg, "clean_sigma", 1.0)  # fallback to 1.0 if not computed

    # --- Generate noiser_images ---
    denom = torch.sqrt(torch.tensor(sigma_a**2 + sigma_clean**2, device=device))
    noiser_images = noisy_images / denom + sigma_a * added_noise

    return dict(
        noiser_images=noiser_images,
        added_noise=added_noise,
        sigma_a=sigma_a,
        sigma_clean=sigma_clean
    )

def test_data_process(batch,cfg,device) -> dict:
    images = batch["image"]
    noisy_images = batch["noisy_image"]

    noisy_images_batch = noisy_images.to(torch.float32).to(device)
    noisy_images_batch = noisy_images_batch.permute(0, 3, 1, 2).contiguous() # to (b, c, h, w)

    images = images.cpu().numpy() # current format (b, h, w, c)
    noisy_images = noisy_images.cpu().numpy() # current format (b, h, w, c)

    return dict(
        images=images, noisy_images=noisy_images,
        noiser_images=noisy_images_batch, file_names=batch["file_name"]
    )

def postprocess(batch,cfg, model):
    x0 = batch["noiser_images"]              # [B,C,H,W]
    out = x0.clone()

    a = cfg.a
    b = cfg.b
    num_iteration = cfg.num_iteration
    step_size = getattr(cfg, "postprocess_step_size", 0.7)

    # reuse the dict, but overwrite the image each iter
    batch_iter = dict(batch)

    for i in range(num_iteration):
        sigma_map = (a + b * out).clamp(min=1e-6)

        batch_iter["noiser_images"] = out
        batch_iter["sigma_map"] = sigma_map   # only if forward uses it; otherwise harmless

        scores = model(batch_iter) 

        out = out + step_size * (sigma_map ** 2) * scores

    out = out.permute(0, 2, 3, 1).contiguous()
    return out.cpu().numpy()

@torch.no_grad()
def compute_clean_stats(dataloader, device="cpu"):
    """
    Compute global mean (mu_y) and standard deviation (sigma_y) 
    from clean image batches in a dataloader.
    """
    ys = []
    for batch in dataloader:
        if "clean_images" not in batch:
            raise KeyError("[ERROR] dataloader batch must contain 'clean_images'")
        y = batch["clean_images"].to(torch.float32).to(device)
        if y.dim() == 3:
            y = y.unsqueeze(1) # add channel dim if missing, only for grayscale images
        ys.append(y.view(y.size(0), -1))  # flatten each image

    Y = torch.cat(ys, dim=0)  # [N, D]
    mu_y = Y.mean(dim=0)
    sigma_y = Y.std(dim=0)
    sigma_scalar = sigma_y.mean().item()

    return mu_y, sigma_scalar