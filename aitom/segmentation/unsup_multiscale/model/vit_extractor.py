import torch, math, random
from tqdm import tqdm
import pdb,os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import numpy as np
import timm
import time
import model.models_mae
import model.models_dino
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, DDPMScheduler

def get_attn_fn():
    attns = []
    @torch.no_grad()
    def attn_hook_fn(module, args, kwargs, output):
        self = module
        x = args[0]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        raw_attn = (q @ k.transpose(-2, -1)) * self.scale
        attns.append(raw_attn.detach()[:,:,1:,1:].softmax(dim = -1))
    return attns, attn_hook_fn

def resize_kq_atten(attn_list, ori_patch_h, ori_patch_w, patch_h, patch_w):
    q_raw, k_raw, scale = attn_list
    q_cls = q_raw[:,:,0,:]
    q_feat = q_raw[:,:,1:,:]
    q_feat = F.interpolate(q_feat.reshape(q_raw.shape[1], ori_patch_h, ori_patch_w, -1).permute(0,3,1,2),[patch_h, patch_w], mode = 'bicubic')
    q = torch.cat([q_cls[0][:,None,:], q_feat.flatten(2,-1).permute(0,2,1)], dim = 1)[None,...]

    k_cls = k_raw[:,:,:,0]
    k_feat = k_raw[:,:,:,1:]
    k_feat = F.interpolate(k_feat.reshape(k_raw.shape[1], -1, 14,14),[patch_h, patch_w], mode = 'bicubic')
    k = torch.cat([k_cls[0][:,:,None], k_feat.flatten(2,-1)], dim = -1)[None,...]
    raw_attn = (q @ k) * scale
    return raw_attn

class MAEFeatureExtractor(object):
    def __init__(self, args):
        device = args.gpu
        patch_size = 16
        model_arch = 'vit_base'
        self.model = model.models_mae .__dict__['mae_{}_patch{}'.format(model_arch, patch_size)]()
        ckpt = torch.load('../mae_pretrain_vit_base.pth')
        msg = self.model.load_state_dict(ckpt['model'], strict = False)
        self.model.to(device)
        self.data_norm_mean = torch.FloatTensor([0.485, 0.456, 0.406]).reshape(1,3,1,1)
        self.data_norm_std = torch.FloatTensor([0.229, 0.224, 0.225]).reshape(1,3,1,1)

        attn_hooks = [
            module.register_forward_hook(attn_hook_fn, with_kwargs=True)
            for module in self.model.modules() if isinstance(module, timm.models.vision_transformer.Attention)
        ]

    def __call__(self, img):
        norm_img = (img - self.data_norm_mean.to(img.device)) / self.data_norm_std.to(img.device)
        model_feat = self.model.forward_encoder(norm_img)
        return model_feat

class DINOFeaturizer(nn.Module):

    def __init__(self, patch_size = 8, arch = 'vit_base'):
        super().__init__()
        self.patch_size = patch_size
        self.model = model.models_dino.__dict__[arch](
            patch_size=patch_size,
            num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval().cuda()

        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError("Unknown arch and patch size")

        print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        self.model.load_state_dict(state_dict, strict=True)

        if arch == "vit_small":
            self.n_feats = 384
        else:
            self.n_feats = 768

    def forward(self, img, n=1, return_class_feat=False):
        self.model.eval()
        with torch.no_grad():
            assert (img.shape[2] % self.patch_size == 0)
            assert (img.shape[3] % self.patch_size == 0)

            # get selected layer activations
            feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
            feat, attn, qkv = feat[0], attn[0], qkv[0]

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
        return image_feat

class DINOFeatureExtractor(object):
    def __init__(self, args):
        device = args.gpu
        patch_size = args.vit_patch_size
        model_arch = args.vit_model_arch
        self.img_size = args.vit_resize_img_size
        self.feat_h = self.feat_w = self.img_size // int(patch_size)
        self.model = DINOFeaturizer(patch_size, model_arch)
        self.model.to(device)
        self.data_norm_mean = torch.FloatTensor([0.485, 0.456, 0.406]).reshape(1,3,1,1)
        self.data_norm_std = torch.FloatTensor([0.229, 0.224, 0.225]).reshape(1,3,1,1)
        self.attns, attn_hook_fn = get_attn_fn()
        attn_hooks = [
            module.register_forward_hook(attn_hook_fn, with_kwargs=True)
            for module in self.model.modules() if isinstance(module, model.models_dino.Attention)
        ]

    def __call__(self, img):
        norm_img = (img - self.data_norm_mean.to(img.device)) / self.data_norm_std.to(img.device)
        dino_feature = self.model(norm_img)
        return dino_feature

    def process_input(self, img):
        img = F.interpolate(img, size = (self.img_size, self.img_size), mode = 'bilinear')
        return img

    def collect_attention(self, img, i):
        if len(self.attns) == 0:
            self(img)
        return self.attns

    def clear_after_loop(self):
        self.attns.clear()
