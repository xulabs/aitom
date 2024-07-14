import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm


@torch.inference_mode()
def preprocess_dino(x, mean, std, size=518, device="cuda"):
    normed = (x[..., None].repeat(3, axis=-1) - np.asarray(mean)) / np.asarray(std)
    tensor = torch.as_tensor(normed.transpose(2, 0, 1)[None], dtype=torch.float32, device=device)
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=True)

    return resized


@torch.inference_mode()
def extract_dino(model, voxel):
    (D, H, W), L = voxel.shape, int(np.sqrt(model.pos_embed.shape[1] - 1))
    mean, std = np.stack([voxel.mean()] * 3), np.stack([voxel.std()] * 3)
    features = {"z": [], "y": [], "x": []}

    for z in tqdm(range(D)):
        slice_z = preprocess_dino(voxel[z], mean=mean, std=std)
        feature_z = model.forward_features(slice_z)["x_norm_patchtokens"].reshape(1, L, L, -1)
        features["z"].append(feature_z)
    features["z"] = torch.cat(features["z"])
    for y in tqdm(range(H)):
        slice_y = preprocess_dino(voxel[:, y], mean=mean, std=std)
        feature_y = model.forward_features(slice_y)["x_norm_patchtokens"].reshape(1, L, L, -1)
        features["y"].append(feature_y)
    features["y"] = torch.cat(features["y"])
    for x in tqdm(range(W)):
        slice_x = preprocess_dino(voxel[:, :, x], mean=mean, std=std)
        feature_x = model.forward_features(slice_x)["x_norm_patchtokens"].reshape(1, L, L, -1)
        features["x"].append(feature_x)
    features["x"] = torch.cat(features["x"])

    return features
