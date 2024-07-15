import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

from cryosam.config import CFG
from cryosam.dino_utils import extract_dino
from cryosam.sam_utils import extract_sam_z, predict_sam_z
from cryosam.utils import sample, down_sample


class CryoSAM:
    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg").eval().cuda()
        self.sam = SamPredictor(
            sam_model_registry["default"](checkpoint=Path(cfg.model_dir) / "sam_vit_h_4b8939.pth").eval().cuda()
        )
        self.dino_cache = None
        self.sam_cache = None

    def set_cache(self, voxel, key="unknown"):
        dino_cache_root = Path(self.cfg.cache_dir) / "dino"
        dino_cache_paths = {
            dino_cache_path.stem: dino_cache_path for dino_cache_path in list(dino_cache_root.glob("*.pth"))
        }
        sam_cache_root = Path(self.cfg.cache_dir) / "sam"
        sam_cache_paths = {sam_cache_path.stem: sam_cache_path for sam_cache_path in list(sam_cache_root.glob("*.pth"))}

        if key in dino_cache_paths.keys():
            self.dino_cache = torch.load(dino_cache_paths[key])
        else:
            self.dino_cache = extract_dino(self.dino, voxel)
            if self.cfg.save_cache and key != "unknown":
                dino_cache_root.mkdir(parents=True, exist_ok=True)
                torch.save(self.dino_cache, dino_cache_root / f"{key}.pth")

        if key in sam_cache_paths.keys():
            self.sam_cache = torch.load(sam_cache_paths[key])
        else:
            self.sam_cache = extract_sam_z(self.sam, voxel)
            if self.cfg.save_cache and key != "unknown":
                sam_cache_root.mkdir(parents=True, exist_ok=True)
                torch.save(self.sam_cache, sam_cache_root / f"{key}.pth")

    def prompt_points(self, points):
        segmentations = list()
        for point in tqdm(points):
            segmentation = predict_sam_z(
                self.sam,
                point,
                area_threshold=self.cfg.area_threshold,
                iou_threshold=self.cfg.iou_threshold,
                score_threshold=self.cfg.score_threshold,
                logit_threshold=self.cfg.logit_threshold,
                cache=self.sam_cache,
            )
            if segmentation is not None:
                segmentations.append(segmentation)
            else:
                print(f"{point} no mask")
        assert len(segmentations) > 0

        return segmentations

    def avg_pool(self, prompts):
        prompt_zyxs = np.stack([m.mean(0).round() for m in prompts])
        qs = sample(self.dino_cache, torch.as_tensor(prompt_zyxs).long().cuda())
        qs = qs.mean(0, keepdim=True)

        return F.normalize(qs.flatten(1))

    def match(self, normed_qs):
        masks, probs = list(), list()

        for level, ds in enumerate(self.cfg.down_sample_ratios):
            down_features = down_sample(self.dino_cache, ratio=ds)
            d, h, w = [down_features[c].shape[0] for c in ["z", "y", "x"]]
            k_zyxs = torch.meshgrid(torch.arange(d), torch.arange(h), torch.arange(w))
            k_zyxs = torch.stack(k_zyxs, dim=-1).flatten(end_dim=-2).cuda()
            if level == 0:
                mask = torch.ones((1, 1, d, h, w), dtype=torch.bool, device="cuda")
                prob = torch.zeros((1, 1, d, h, w), dtype=torch.float, device="cuda")
            else:
                print(f"level{level} ds{ds} resolution {(d, h, w)}")
                if ds < self.cfg.down_sample_ratios[level - 1]:
                    mask = F.interpolate(masks[-1].type(torch.float), (d, h, w)).type(torch.bool)
                    prob = F.interpolate(probs[-1], (d, h, w), mode="trilinear")
                    k_zyxs = k_zyxs[mask.flatten()]
                else:
                    mask = copy.deepcopy(masks[-1])
                    prob = copy.deepcopy(probs[-1])
                    k_zyxs = k_zyxs[mask.flatten()]
                print(f"{len(k_zyxs)} samples after pre-filtering")

            mask, prob = mask.flatten(), prob.flatten()
            for ki in tqdm(range(0, len(k_zyxs), self.cfg.chunk_k)):
                k_zyx = k_zyxs[ki : ki + self.cfg.chunk_k]
                k_idx = k_zyx[:, 2] + k_zyx[:, 1] * w + k_zyx[:, 0] * h * w
                ks = sample(down_features, k_zyx)
                normed_ks = F.normalize(ks.flatten(1))
                qk = normed_qs @ normed_ks.T
                prob[k_idx] = qk.T.max(1).values
            mask, prob = mask.reshape(1, 1, d, h, w), prob.reshape(1, 1, d, h, w)
            if level == len(self.cfg.down_sample_ratios) - 1:
                max_prob = F.max_pool3d(prob, 5, stride=1, padding=2)
                max_mask = prob == max_prob
                mask[~max_mask] = False
                if 0 < self.cfg.top_k < len(prob[mask]):
                    top_k_prob = prob[mask].topk(self.cfg.top_k).values.min()
                    mask[prob < top_k_prob] = False
            else:
                mask[prob < self.cfg.similarity_threshold] = False
            masks.append(mask)
            probs.append(prob)

        return masks, probs

    def detect(self, points):
        # extract query features
        query_masks = self.prompt_points(points)
        query_features = self.avg_pool(query_masks)

        # hierarchical feature matching
        detections, _ = self.match(query_features)

        return detections[-1]

    def segment(self, voxel, detections):
        D, H, W = voxel.shape
        detections = detections.squeeze().nonzero() * self.cfg.down_sample_ratios[-1]
        segmentations = np.zeros((D, H, W), dtype=np.bool_)
        for point in tqdm(detections):
            segmentation = predict_sam_z(
                self.sam,
                point.cpu().numpy(),
                area_threshold=self.cfg.area_threshold,
                iou_threshold=self.cfg.iou_threshold,
                score_threshold=self.cfg.score_threshold,
                logit_threshold=self.cfg.logit_threshold,
                cache=self.sam_cache,
                return_voxel=True,
            )
            if segmentation is not None:
                segmentations += segmentation

        return segmentations

    def infer(self, voxel, input_prompts, key="unknown"):
        self.set_cache(voxel, key=key)
        proposals = self.detect(input_prompts)
        segmentations = self.segment(voxel, proposals[-1])

        return segmentations
