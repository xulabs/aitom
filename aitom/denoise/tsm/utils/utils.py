import torch
import torch.nn as nn
import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import mrcfile
from torch.utils.data._utils.collate import default_collate

def collate_mrc(batch):
    """
    Custom collate_fn that handles both:
      - normal dict samples
      - list of dict samples (from .mrcs stacks)
    Flattens all into one batch dict so train.py doesn't need changes.
    """
    # Flatten any nested lists
    flat_batch = []
    for item in batch:
        if isinstance(item, list):
            flat_batch.extend(item)
        else:
            flat_batch.append(item)
    # Use PyTorch's default_collate on flattened list of dicts
    return default_collate(flat_batch)

def count_parameters(model, trainable_only: bool = True) -> float: 
    if trainable_only:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        num_params = sum(p.numel() for p in model.parameters())
    return num_params / 1e6  # Convert to millions

def tsm_lambda(global_step, warmup_steps, ramp_steps):
    """
    lambda(t): 0 during warmup, linear ramp to 1, then stays at 1
    """
    if global_step < warmup_steps:
        return 0.0
    elif global_step < warmup_steps + ramp_steps:
        return (global_step - warmup_steps) / ramp_steps
    else:
        return 1.0

def loss_func(train_data_dict_dsm, train_data_dict_tsm, outputs_dsm, outputs_tsm, cfg, global_step, return_components=False):
    """
    Combined DSM + TSM loss with per-sample weighting from CleanTargetProjector.
    """
    mse = nn.MSELoss(reduction="none")

    # --- Extract common tensors ---
    I_dsm = train_data_dict_dsm["added_noise"]        # [B,C,H,W]
    x_dsm = train_data_dict_dsm["noiser_images"]      # [B,C,H,W]
    x_tsm = train_data_dict_tsm["noiser_images"]      # [B,C,H,W]
    y_tsm = train_data_dict_tsm["clean_images"]      # [B,C,H,W]

    device = x_dsm.device
    sigma_a = torch.tensor(cfg.model_config.additional_sigmas[0], device=device)
    sigma   = torch.tensor(cfg.model_config.clean_sigma, device=device)

    if "similarity" in train_data_dict_tsm:
        w_t = train_data_dict_tsm["similarity"].to(device).clamp(0.0, 1.0)  # [B]
        w_t_threshold = getattr(cfg.model_config, "w_t_threshold", 0.93)
        picked_mask = (w_t >= w_t_threshold)
        # if w_t <= threshold, set w_t=0 to focus on DSM loss only
        w_t = torch.sigmoid(3.0 * (w_t - w_t_threshold)).detach()
        num_picked = picked_mask.sum().item()
        batch_size = w_t.numel()
        mean_sim = train_data_dict_tsm["similarity"].mean().item()
    else:
        w_t = torch.full((x_dsm.size(0),), 0.0, device=device) # if no similarity, set w_t=0 to auto-enable DSM loss only mode
        picked_mask = None
        num_picked = 0
        batch_size = x_dsm.size(0)
        mean_sim = 0.0

    L_DSM = mse(outputs_dsm, -I_dsm).mean(dim=(1, 2, 3))  # [B]

    grad_x = torch.abs(x_tsm[:, :, :, 1:] - x_tsm[:, :, :, :-1])   # (B,C,H,W-1)
    grad_y = torch.abs(x_tsm[:, :, 1:, :] - x_tsm[:, :, :-1, :])   # (B,C,H-1,W)

    # pad to (B,C,H,W)
    grad_x = F.pad(grad_x, (0, 1, 0, 0))   # pad right
    grad_y = F.pad(grad_y, (0, 0, 0, 1))   # pad bottom

    grad_mag = grad_x + grad_y
    grad_mag = grad_mag.mean(dim=1, keepdim=True)
    structure_mask = (grad_mag > grad_mag.mean()).float().detach()

    coef_front = (sigma**2 * sigma_a**2 + sigma**4) / (sigma**2 + 1e-12)
    coef_inner = sigma_a / (sigma * torch.sqrt(sigma_a**2 + sigma**2 + 1e-12))
    
    pred_term = coef_inner * outputs_tsm
    norm = torch.sqrt(sigma_a**2 + sigma**2 + 1e-12)
    target_term = (x_tsm - y_tsm / norm) / (sigma + 1e-12)

    # remove per-patch mean and variance
    pred_norm   = (pred_term - pred_term.mean(dim=(2,3), keepdim=True)) / (pred_term.std(dim=(2,3), keepdim=True) + 1e-6)
    target_norm = (target_term - target_term.mean(dim=(2,3), keepdim=True)) / (target_term.std(dim=(2,3), keepdim=True) + 1e-6)

    L_TSM = coef_front * (
        mse(pred_norm, target_norm) * structure_mask
    ).mean(dim=(1,2,3))

    lambda_tsm = tsm_lambda(global_step, warmup_steps=50, ramp_steps=30)

    alpha_tsm = getattr(cfg.model_config, "alpha_tsm", 0.1)

    # loss = L_DSM.mean()
    loss = L_DSM.mean() + alpha_tsm * lambda_tsm * (w_t * L_TSM).mean()

    if return_components:
        return {
            "loss": loss,
            "dsm": L_DSM.mean(),
            "tsm": L_TSM.mean(),
            "num_picked": num_picked,
            "batch_size": batch_size,
            "pick_ratio": num_picked / max(batch_size, 1),
            "mean_similarity": mean_sim,
        }
    return loss

def save_image(image: np.ndarray, image_path: str) -> None:

    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()

    # Remove singleton dimensions
    image = np.squeeze(image)

    # If shape is [H, W], fine. If still [1, 1], flatten more.
    if image.ndim == 0:
        image = np.array([[image]])
    elif image.ndim == 1:
        image = image[np.newaxis, :]
    elif image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))  # [C, H, W] → [H, W, C]

    #image = np.uint8((image - image.min()) / (image.max() - image.min() + 1e-5) * 255)
    with mrcfile.new(image_path, overwrite=True) as mrc:
        mrc.set_data(image.astype(np.float32))


def save_output(test_data_dict, output_dir):
    batch_size = len(test_data_dict["noisy_images"])

    for i in range(batch_size):
        output_subdir = os.path.join(output_dir, test_data_dict["file_names"][i])
        os.makedirs(output_subdir, exist_ok=True)

        image = test_data_dict["images"][i]
        noisy_image = test_data_dict["noisy_images"][i]
        denoised_image = test_data_dict['denoised_images'][i]

        save_image(image,
                    image_path=os.path.join(output_subdir, "image.mrc"))
        save_image(noisy_image,
                    image_path=os.path.join(output_subdir,
                                            "noisy_image.mrc"))
        save_image(denoised_image,
                    image_path=os.path.join(output_subdir,
                                            "denoised_image.mrc"))


def compute_image_metrics(ground_truth: np.ndarray,
                               comparison_obj: np.ndarray):
        psnr = compute_psnr(ground_truth, comparison_obj)
        ssim = compute_ssim(ground_truth, comparison_obj)
        nmse = compute_nmse(ground_truth, comparison_obj)
        mse = compute_mse(ground_truth, comparison_obj)
        return psnr, ssim, nmse, mse

def compute_metrics(test_data_dict):
    batch_size = len(test_data_dict["noisy_images"])
    metrics = {}
    for key in ["psnr", "ssim", "nmse", "mse",
                "n_psnr", "n_ssim", "n_nmse", "n_mse"]:
        metrics[key] = []
    for i in range(batch_size):
        file_name = test_data_dict["file_names"][i]
        image = test_data_dict["images"][i].astype(np.float32)
        noisy_image = test_data_dict["noisy_images"][i].astype(np.float32)
        denoised_image = test_data_dict["denoised_images"][i].astype(np.float32)

        psnr, ssim, nmse, mse = compute_image_metrics(
            ground_truth=image,
            comparison_obj=denoised_image
        )
        n_psnr, n_ssim, n_nmse, n_mse = compute_image_metrics(
            ground_truth=image,
            comparison_obj=noisy_image
        )

        metrics["psnr"].append(psnr)
        metrics["ssim"].append(ssim)
        metrics["nmse"].append(nmse)
        metrics["mse"].append(mse)
        metrics["n_psnr"].append(n_psnr)
        metrics["n_ssim"].append(n_ssim)
        metrics["n_nmse"].append(n_nmse)
        metrics["n_mse"].append(n_mse)

    return metrics

def compute_psnr(gt: np.ndarray, pred: np.ndarray):
    """
    Compute PSNR.
    """
    return peak_signal_noise_ratio(gt, pred, data_range=np.max(gt))

def compute_ssim(gt: np.ndarray, pred: np.ndarray):
    """
    Compute Structural Similarity (SSIM).
    """
    # gt and pred should be with shape of (c, h, w)
    gt = np.transpose(gt, [2, 0, 1])
    pred = np.transpose(pred, [2, 0, 1])
    return structural_similarity(
        gt, pred,
        data_range=np.max(gt),
        channel_axis=0
    )

def compute_nmse(gt: np.ndarray, pred: np.ndarray):
    """
    Compute Normalized Mean Squared Error (NMSE)
    """
    return np.linalg.norm(gt - pred) ** 2 / (np.linalg.norm(gt) ** 2 + 1e-8)

def compute_mse(gt: np.ndarray, pred: np.ndarray):
    """
    Compute MSE
    """
    return np.mean((gt - pred) ** 2)

import torch
import torch.nn.functional as F


def remap_to_raw_range(raw, den):
    """
    remap denoised image into the same numeric range as raw image
    so 3dmod displays particles correctly.
    """
    raw_min, raw_max = raw.min(), raw.max()
    den_min, den_max = den.min(), den.max()

    # Normalize den to [0,1]
    den_norm = (den - den_min) / (den_max - den_min + 1e-8)

    # Map to raw range
    den_rescaled = den_norm * (raw_max - raw_min) + raw_min
    return den_rescaled

class UNetMidExtractor(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.encoder = nn.Sequential(unet.input_blocks)
        self.middle = unet.middle_block  # adjust name if different

    def forward(self, x):
        # forward up to middle layer
        for module in self.encoder:
            x = module(x)
        x = self.middle(x)
        # flatten global features
        return torch.mean(x, dim=[2,3])  # [B, C] if 1024 channels