import einops
import torch
import torch.nn.functional as F


@torch.inference_mode()
def sample(features, zyxs):
    D, H, W = [features[c].shape[0] for c in ["z", "y", "x"]]
    feature_zs = features["z"][zyxs[:, 0]].permute(0, 3, 1, 2)
    feature_ys = features["y"][zyxs[:, 1]].permute(0, 3, 1, 2)
    feature_xs = features["x"][zyxs[:, 2]].permute(0, 3, 1, 2)
    normed_zyxs = (2 * zyxs / torch.as_tensor([D - 1, H - 1, W - 1]).to(zyxs) - 1)[:, None, None]
    sampled_zs = F.grid_sample(feature_zs, normed_zyxs[..., [2, 1]], align_corners=True).flatten(1)
    sampled_ys = F.grid_sample(feature_ys, normed_zyxs[..., [2, 0]], align_corners=True).flatten(1)
    sampled_xs = F.grid_sample(feature_xs, normed_zyxs[..., [1, 0]], align_corners=True).flatten(1)
    sampled_features = torch.stack([sampled_zs, sampled_ys, sampled_xs], dim=2)

    return sampled_features


@torch.inference_mode()
def sample_z(features, zyxs):
    D, H, W = [features[c].shape[0] for c in ["z", "y", "x"]]
    feature_zs = features["z"][zyxs[:, 0]].permute(0, 3, 1, 2)
    normed_zyxs = (2 * zyxs / torch.as_tensor([D - 1, H - 1, W - 1]).to(zyxs) - 1)[:, None, None]
    sampled_zs = F.grid_sample(feature_zs, normed_zyxs[..., [2, 1]], align_corners=True).flatten(1)
    sampled_features = sampled_zs[..., None]

    return sampled_features


@torch.inference_mode()
def down_sample(features, ratio=1):
    if ratio == 1:
        return features
    (D, H, W), L = [features[c].shape[0] for c in ["z", "y", "x"]], features["z"].shape[1]
    feature_z, feature_y, feature_x = features["z"], features["y"], features["x"]

    feature_z = einops.rearrange(feature_z, "D h w c -> (h w) c D")
    down_feature_z = F.interpolate(feature_z, size=D // ratio, mode="linear", align_corners=True)
    down_feature_z = einops.rearrange(down_feature_z, "(h w) c D -> D h w c", h=L, w=L)
    feature_y = einops.rearrange(feature_y, "H d w c -> (d w) c H")
    down_feature_y = F.interpolate(feature_y, size=H // ratio, mode="linear", align_corners=True)
    down_feature_y = einops.rearrange(down_feature_y, "(d w) c H -> H d w c", d=L, w=L)
    feature_x = einops.rearrange(feature_x, "W d h c -> (d h) c W")
    down_feature_x = F.interpolate(feature_x, size=W // ratio, mode="linear", align_corners=True)
    down_feature_x = einops.rearrange(down_feature_x, "(d h) c W -> W d h c", d=L, h=L)
    down_features = {"z": down_feature_z, "y": down_feature_y, "x": down_feature_x}

    return down_features
