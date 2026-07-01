import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader


class CleanTargetProjector(nn.Module):
    """
    Extract middle_block features from GuidedDDPMPlainUNet.
    Compare noisy mid-features to clean mid-features via cosine similarity.
    """

    def __init__(self, model, clean_dataloader, max_clean=None, device="cuda", mix=False):
        super().__init__()
        self.model = model.eval().to(device)
        self.device = device
        self.max_clean = max_clean
        self.mix = mix

        self._features = []

        def hook(module, inp, out):
            pooled = out.mean(dim=[2,3])   # [B, C]
            self._features.append(pooled.detach()) # calculate mean feature vector for each batch

        if not hasattr(self.model.network, "middle_block"):
            raise AttributeError("[ERROR] model.network does NOT have middle_block.")

        self.hook_handle = self.model.network.middle_block.register_forward_hook(hook)

        clean_feats, clean_imgs, filenames = self._extract_clean_features(clean_dataloader)

        if max_clean is not None and clean_feats.size(0) > max_clean:
            clean_feats = clean_feats[:max_clean]
            clean_imgs = clean_imgs[:max_clean]
            filenames = filenames[:max_clean]

        self.clean_feats = clean_feats.to(device)   # format: [K, C]
        self.clean_images = clean_imgs.to(device)   # format: [K, C, H, W]
        print(f"[INFO] Clean feature matrix: {self.clean_feats.size()}")

        if self.mix:
            # Separate 10A and 3A domains by folder name
            indices_10A = [i for i, fn in enumerate(filenames) if "10A" in str(fn)]
            indices_3A  = [i for i, fn in enumerate(filenames) if "3A"  in str(fn)]
            self.indices_10A = torch.tensor(indices_10A, device=device, dtype=torch.long)
            self.indices_3A  = torch.tensor(indices_3A,  device=device, dtype=torch.long)
            print(f"[INFO] Mix mode: {len(indices_10A)} samples @ 10A, {len(indices_3A)} samples @ 3A")
        else:
            print(f"[INFO] Single-domain mode: all {self.clean_feats.size(0)} clean samples used directly")

        '''
        # --- background features ---
        if bg_dataloader is not None:
            bg_feats, _ = self._extract_clean_features(bg_dataloader)
            self.background_feat = F.normalize(
                bg_feats.mean(dim=0, keepdim=True), dim=1
            )
        else:
            self.background_feat = None
        self.background_feat = self.background_feat.to(self.device)
        '''
    
    @torch.no_grad()
    def refresh_clean_features(self, clean_dataloader):
        """
        Recompute clean features using the CURRENT model weights.
        """
        self._features = []

        clean_feats, clean_images, filenames = self._extract_clean_features(clean_dataloader)

        if self.max_clean is not None and clean_feats.size(0) > self.max_clean:
            clean_feats  = clean_feats[:self.max_clean]
            clean_images = clean_images[:self.max_clean]
            filenames = filenames[:self.max_clean]

        self.clean_feats  = clean_feats.to(self.device)    # [K, C]
        self.clean_images = clean_images.to(self.device)   # [K, C, H, W]

        if self.mix:
            # Re-separate domains after refresh
            indices_10A = [i for i, fn in enumerate(filenames) if "10A" in str(fn)]
            indices_3A  = [i for i, fn in enumerate(filenames) if "3A"  in str(fn)]
            self.indices_10A = torch.tensor(indices_10A, device=self.device, dtype=torch.long)
            self.indices_3A  = torch.tensor(indices_3A,  device=self.device, dtype=torch.long)

    @torch.no_grad()
    def _extract_clean_features(self, dataloader):
        feat_list = []
        img_list  = []
        filename_list = []

        was_training = self.model.training
        for batch in dataloader:
            clean = batch.get("clean_images", batch.get("image"))
            clean = clean.to(self.device, dtype=torch.float32)

            if clean.ndim == 4 and clean.shape[1] != 1:
                clean = clean.permute(0, 3, 1, 2)  # HWC → CHW

            self._features.clear()
            _ = self.model.network(clean)          # triggers hook
            assert len(self._features) == 1
            mid = self._features[0]

            feat_list.append(mid.cpu())
            img_list.append(clean.cpu())
            
            # Track filenames for domain separation
            file_path = batch.get("file_path", "unknown")
            if isinstance(file_path, (list, tuple)):
                file_path = file_path[0] if len(file_path) > 0 else "unknown"
            filename_list.append(file_path)

            self._features.clear()

        if was_training:
            self.model.train()

        feats = torch.cat(feat_list, dim=0)        # [K, C]
        imgs  = torch.cat(img_list,  dim=0)        # [K, C, H, W]
        return feats, imgs, filename_list

    @torch.no_grad()
    def similarity_all_clean(self, noisy_images, return_all=False):
        if noisy_images.ndim == 4 and noisy_images.shape[1] != 1:
            noisy_images = noisy_images.permute(0, 3, 1, 2)

        noisy_images = noisy_images.to(self.device).float()
        
        was_training = self.model.training
        self.model.eval()

        self._features.clear()
        _ = self.model.network(noisy_images)
        assert len(self._features) == 1
        noisy_feat = F.normalize(self._features[0], dim=1)
        self._features.clear()

        clean_feat = F.normalize(self.clean_feats, dim=1)
        sims = noisy_feat @ clean_feat.T   # [B, K]

        if was_training:
            self.model.train()

        if return_all:
            return sims
        else:
            best_sim, best_idx = sims.max(dim=1)
            return best_idx, best_sim

    @torch.no_grad()
    def get_weighted_target(self, noisy_images, tau=0.5, alpha_10A=0.5, alpha_3A=0.5):
        """
        Unified entry point.
        - mix=True:  split into 10A/3A domains, softmax each separately, combine by alpha.
        - mix=False: softmax over all clean images at once (original behaviour).
        Returns (y_combined [B,C,H,W], best_sim [B])
        """
        if self.mix:
            return self.get_weighted_target_dual_domain(
                noisy_images, tau=tau, alpha_10A=alpha_10A, alpha_3A=alpha_3A
            )
        else:
            return self._get_weighted_target_single(noisy_images, tau=tau)

    @torch.no_grad()
    def _get_weighted_target_single(self, noisy_images, tau=0.5):
        """Original single-domain softmax-weighted target."""
        if noisy_images.ndim == 4 and noisy_images.shape[1] != 1:
            noisy_images = noisy_images.permute(0, 3, 1, 2)
        noisy_images = noisy_images.to(self.device).float()

        was_training = self.model.training
        self.model.eval()

        self._features.clear()
        _ = self.model.network(noisy_images)
        assert len(self._features) == 1
        noisy_feat = F.normalize(self._features[0], dim=1)
        self._features.clear()

        clean_feat = F.normalize(self.clean_feats, dim=1)
        sims = noisy_feat @ clean_feat.T          # [B, K]
        weights = torch.softmax(sims / tau, dim=1)  # [B, K]
        y = torch.einsum("bk,kchw->bchw", weights, self.clean_images)
        best_sim = sims.max(dim=1).values

        if was_training:
            self.model.train()

        return y, best_sim

    @torch.no_grad()
    def get_weighted_target_dual_domain(self, noisy_images, tau=0.5, alpha_10A=0.5, alpha_3A=0.5):
        """
        Compute weighted clean target by separately softmax-ing 10A and 3A domains.
        Then combine with alpha weights.
        
        Args:
            noisy_images: [B, C, H, W] or [B, H, W, C]
            tau: temperature for softmax
            alpha_10A, alpha_3A: weights for combining domains (should sum to 1.0)
        
        Returns:
            y_combined: [B, C, H, W] weighted combination of 10A and 3A targets
            best_sim: [B] max similarity across all domains (for gating)
        """
        if noisy_images.ndim == 4 and noisy_images.shape[1] != 1:
            noisy_images = noisy_images.permute(0, 3, 1, 2)

        noisy_images = noisy_images.to(self.device).float()
        
        was_training = self.model.training
        self.model.eval()

        self._features.clear()
        _ = self.model.network(noisy_images)
        assert len(self._features) == 1
        noisy_feat = F.normalize(self._features[0], dim=1)
        self._features.clear()

        clean_feat = F.normalize(self.clean_feats, dim=1)
        sims_all = noisy_feat @ clean_feat.T   # [B, K]
        
        # Extract domain-specific similarities
        if len(self.indices_10A) > 0:
            sims_10A = sims_all[:, self.indices_10A]  # [B, K_10A]
            weights_10A = torch.softmax(sims_10A / tau, dim=1)
            clean_imgs_10A = self.clean_images[self.indices_10A]  # [K_10A, C, H, W]
            y_10A = torch.einsum("bk,kchw->bchw", weights_10A, clean_imgs_10A)
        else:
            y_10A = None
        
        if len(self.indices_3A) > 0:
            sims_3A = sims_all[:, self.indices_3A]  # [B, K_3A]
            weights_3A = torch.softmax(sims_3A / tau, dim=1)
            clean_imgs_3A = self.clean_images[self.indices_3A]  # [K_3A, C, H, W]
            y_3A = torch.einsum("bk,kchw->bchw", weights_3A, clean_imgs_3A)
        else:
            y_3A = None
        
        # Combine domains
        if y_10A is not None and y_3A is not None:
            y_combined = alpha_10A * y_10A + alpha_3A * y_3A
        elif y_10A is not None:
            y_combined = y_10A
        elif y_3A is not None:
            y_combined = y_3A
        else:
            raise ValueError("No 10A or 3A samples found in clean dataset!")
        
        # Best similarity across all domains for gating
        best_sim = sims_all.max(dim=1).values

        if was_training:
            self.model.train()

        return y_combined, best_sim