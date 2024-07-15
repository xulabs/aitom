import numpy as np
import torch
from tqdm import tqdm


@torch.inference_mode()
def preprocess_sam(x):
    return (x * 255).astype(np.uint8)[..., None].repeat(3, axis=-1)


@torch.inference_mode()
def extract_sam_axis(model, voxel, ax=0, zyx=None):
    features, original_sizes, input_sizes = list(), list(), list()
    for a in tqdm(range(voxel.shape[ax])) if zyx is None else [zyx[ax]]:
        idx = tuple(slice(None) if axis != ax else a for axis in range(voxel.ndim))
        slice_a = preprocess_sam(voxel[idx])
        model.set_image(slice_a)
        features.append(model.get_image_embedding())
        original_sizes = model.original_size
        input_sizes = model.input_size
    features = torch.cat(features) if zyx is None else {zyx[ax]: features[0]}
    return features, original_sizes, input_sizes


@torch.inference_mode()
def extract_sam_z(model, voxel, zyx=None):
    features, original_sizes, input_sizes = dict(), dict(), dict()
    features["z"], original_sizes["z"], input_sizes["z"] = extract_sam_axis(model, voxel, ax=0, zyx=zyx)
    original_sizes["x"] = voxel.shape[:2]  # trick to save D, H, W
    return features, original_sizes, input_sizes


@torch.inference_mode()
def set_sam_input(model, feature, original_size, input_size):
    model.features = feature
    model.original_size = original_size
    model.input_size = input_size
    model.is_image_set = True


@torch.inference_mode()
def prompt_sam(
    model,
    point_coords,
    area_threshold,
    iou_threshold,
    score_threshold,
    logit_threshold,
    mask=None,
    logit=None,
):
    masks, scores, logits = model.predict(
        point_coords=point_coords,
        point_labels=np.ones(point_coords.shape[0]),
        mask_input=logit[None] > logit_threshold if logit is not None else None,
    )
    areas = masks.sum((1, 2))
    if mask is None:
        condition = (areas < area_threshold) * (scores > score_threshold)
    else:
        ious = (masks * mask[None]).sum(1).sum(1) / (masks + mask[None]).sum(1).sum(1)
        condition = (areas < area_threshold) * (scores > score_threshold) * (ious > iou_threshold)
    indices = np.arange(len(masks))[condition]
    if len(indices) == 0:
        return None, None, None
    else:
        index = indices[np.argmax(scores[indices])]
        mask, score, logit = masks[index], scores[index], logits[index]
        return mask, score, logit


@torch.inference_mode()
def predict_sam_z(
    model,
    prompt,
    area_threshold=32 * 32,
    iou_threshold=0.5,
    score_threshold=0.0,
    logit_threshold=0.0,
    voxel=None,
    cache=None,
    return_voxel=False,
):
    if cache is None:
        assert voxel is not None
        D, H, W = voxel.shape
    else:
        features, original_sizes, input_sizes = cache
        D, H, W = list(original_sizes["x"]) + [original_sizes["z"][-1]]
    z_voxel = np.zeros((D, H, W), dtype=np.bool_)

    (z, y, x), z_count = prompt, 0
    if cache is None:
        model.set_image(preprocess_sam(voxel[z]))
    else:
        set_sam_input(model, features["z"][z], original_sizes["z"], input_sizes["z"])
    mask, score, logit = prompt_sam(
        model,
        np.asarray([[x, y]]),
        area_threshold,
        iou_threshold,
        score_threshold,
        logit_threshold,
    )
    if score is None:
        return None
    z_count += 1
    z_voxel[z] += mask
    z_center, mask_center, score_center, logit_center = z, mask, score, logit

    for direction in (-1, 1):
        z, mask, score, logit = z_center, mask_center, score_center, logit_center
        while score > score_threshold and -1 < z + direction < D:
            z = z + direction
            if cache is None:
                model.set_image(preprocess_sam(voxel[z]))
            else:
                set_sam_input(model, features["z"][z], original_sizes["z"], input_sizes["z"])
            mask, score, logit = prompt_sam(
                model,
                np.stack([np.argmax(logit) % 256 / 256 * W, np.argmax(logit) // 256 / 256 * H])[None],
                area_threshold,
                iou_threshold,
                score_threshold,
                logit_threshold,
                mask=mask,
                logit=logit,
            )
            if score is not None:
                z_count += 1
                z_voxel[z] += mask
            else:
                break

    if return_voxel:
        return z_voxel
    else:
        positive_zyxs = np.stack(z_voxel.nonzero(), axis=1)
        return positive_zyxs
