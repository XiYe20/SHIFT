import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
import cv2
# from matplotlib import pyplot as plt
from sea_raft_core.raft import RAFT
import json
import argparse
from PIL import Image
import numpy as np

import timm

class SobelGradientLayer(nn.Module):
    def __init__(self, padding='valid', filter_size=3):
        super().__init__()
        
        self.filter_size = filter_size
        if self.filter_size == 3:
            # Define the convolution filters for gradients
            self.w_h = nn.Parameter(torch.tensor([[-1.0], [0.], [1.0]]).view(1, 1, 3, 1), requires_grad=False)
            self.w_w = nn.Parameter(torch.tensor([[-1.0, 0., 1.0]]).view(1, 1, 1, 3), requires_grad=False)
            self.out_zero_pad = 2
        elif self.filter_size == 2:
            self.w_h = nn.Parameter(torch.tensor([[-1.0], [1.0]]).view(1, 1, 2, 1), requires_grad=False)
            self.w_w = nn.Parameter(torch.tensor([[-1.0, 1.0]]).view(1, 1, 1, 2), requires_grad=False)
            self.out_zero_pad = 1
        else:
            raise NotImplementedError
        self.padding = padding
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        For padding = 'valid' (no padding), we don't have the gradient of first row and first column of x
        i.e.,
            grad_h: (B, 1, H-1, W), grad_h[i, j] = x[i+1, j] - x[i, j]
            grad_w: (B, 1, H, W-1), grad_w[i, j] = x[i, j+1] - x[i, j]
        We assume the boundary satisfy the brightness equation, thus we prepend a zero column and zero row to the grad_h and grad_w.
        Return:
            pad_grad_h: (B, 1, H, W), pad_grad_h[i,j] = x[i, j] - x[i-1, j], for i > 0
            pad_grad_w: (B, 1, H, W), pad_grad_w[i,j] = x[i, j] - x[i, j-1], for j > 0
        """
        # Convert RGB to grayscale
        B, C, H, W = x.shape
        if C == 3:
            r, g, b = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]
            gray = 0.2989 * r + 0.587 * g + 0.114 * b
            gray = gray.unsqueeze(1)  # Add channel dimension
        elif C == 1:
            gray = x
        else:
            raise NotImplementedError
        
        # Convolve with the filters to get gradients
        grad_h = F.conv2d(gray, self.w_h, padding=self.padding)
        grad_w = F.conv2d(gray, self.w_w, padding=self.padding)
        
        if self.out_zero_pad == 1:
            pad_grad_h = torch.cat([torch.zeros((B, 1, 1, W), dtype=grad_h.dtype, device=grad_h.device), grad_h], dim=2)
            pad_grad_w = torch.cat([torch.zeros((B, 1, H, 1), dtype=grad_w.dtype, device=grad_w.device), grad_w], dim=3)
        elif self.out_zero_pad == 2:
            pad_grad_h = torch.cat([torch.zeros((B, 1, 1, W), dtype=grad_h.dtype, device=grad_h.device), grad_h, torch.zeros((B, 1, 1, W), dtype=grad_h.dtype, device=grad_h.device)], dim=2)
            pad_grad_w = torch.cat([torch.zeros((B, 1, H, 1), dtype=grad_w.dtype, device=grad_w.device), grad_w, torch.zeros((B, 1, H, 1), dtype=grad_w.dtype, device=grad_w.device)], dim=3)

        return pad_grad_h, pad_grad_w

class TemporalGradientLayer(nn.Module):
    def __init__(self, sobel_filter_size=3):
        self.sobel_filter_size = sobel_filter_size
        super().__init__()
        
    def forward(self, image1, image2):
        """
        image1: (B, C, H, W)
        image2: (B, C, H, W)
        
        Because of grad_h and grad_w are "valid" padding, we assume the first row and first column of frames have a zero gradient
        We need to set the temporal gradient of first row and first column also to be 0
        Return:
            grad_t: (B, T-1, 1, H, W)
        """
        # Convert RGB to grayscale
        C = image1.shape[1]
        if C == 3:
            gray_image1 = 0.2989 * image1[:, 0, :, :] + 0.587 * image1[:, 1, :, :] + 0.114 * image1[:, 2, :, :]
            gray_image1 = gray_image1.unsqueeze(1)  # Add channel dimension
            gray_image2 = 0.2989 * image2[:, 0, :, :] + 0.587 * image2[:, 1, :, :] + 0.114 * image2[:, 2, :, :]
            gray_image2 = gray_image2.unsqueeze(1)  # Add channel dimension

        elif C == 1:
            gray_image1 = image1
            gray_image2 = image2
        else:
            raise NotImplementedError

        grad_t = gray_image2 - gray_image1
        if self.sobel_filter_size == 2:
            #set the first row and column of each frame to be 0
            grad_t[:, :, 0, :] *= 0.
            grad_t[:, :, :, 0] *= 0.
        elif self.sobel_filter_size == 3:
            #set the first and end row and column of each frame to be 0
            grad_t[:, :, 0, :] *= 0.
            grad_t[:, :, :, 0] *= 0.
            grad_t[:, :, -1, :] *= 0.
            grad_t[:, :, :, -1] *= 0.
            
        return grad_t

# ---------------------------
# Torchvision-ResNet encoder with temporal pooling + MLP head
# ---------------------------

def _build_resnet(backbone: str, pretrained: bool, pretrained_ckpt=None):
    """
    Returns a torchvision ResNet instance with global-pool output (fc kept for weight surgery).
    Supports resnet18/34/50/101/152.
    """
    name = backbone.lower()
    try:
        # Newer torchvision (weights enums)
        if name == "resnet18":
            if pretrained_ckpt is not None:
                model = torchvision.models.resnet18(weights=None)
                weights = torch.load(pretrained_ckpt)
                model.load_state_dict(weights)
            else:
                from torchvision.models import ResNet18_Weights
                weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                model = torchvision.models.resnet18(weights=weights)
        elif name == "resnet34":
            if pretrained_ckpt is not None:
                model = torchvision.models.resnet34(weights=None)
                weights = torch.load(pretrained_ckpt)
                model.load_state_dict(weights)
            else:
                from torchvision.models import ResNet34_Weights
                weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
                model = torchvision.models.resnet34(weights=weights)
        elif name == "resnet50":
            if pretrained_ckpt is not None:
                model = torchvision.models.resnet50(weights=None)
                weights = torch.load(pretrained_ckpt)
                model.load_state_dict(weights)
            else:
                from torchvision.models import ResNet50_Weights
                weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
                model = torchvision.models.resnet50(weights=weights)
        elif name == "resnet101":
            if pretrained_ckpt is not None:
                model = torchvision.models.resnet101(weights=None)
                weights = torch.load(pretrained_ckpt)
                model.load_state_dict(weights)
            else:
                from torchvision.models import ResNet101_Weights
                weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
                model = torchvision.models.resnet101(weights=weights)
        elif name == "resnet152":
            if pretrained_ckpt is not None:
                model = torchvision.models.resnet152(weights=None)
                weights = torch.load(pretrained_ckpt)
                model.load_state_dict(weights)
            else:
                from torchvision.models import ResNet152_Weights
                weights = ResNet152_Weights.IMAGENET1K_V1 if pretrained else None
                model = torchvision.models.resnet152(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    except Exception:
        # Fallback for older torchvision
        ctor = getattr(torchvision.models, name)
        model = ctor(pretrained=pretrained)
    return model

def _surgery_first_conv(resnet, in_ch: int):
    """
    Replace conv1 to accept in_ch. If pretrained and keep_init_from_rgb=True,
    initialize new weights by averaging the original RGB kernels.
    """
    old = resnet.conv1
    if old.in_channels == in_ch:
        return  # nothing to do
    new = nn.Conv2d(in_ch, old.out_channels,
                    kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=False)
    nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
    resnet.conv1 = new

def _disable_inplace_relu(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
        if isinstance(m, nn.GELU):
            m.inplace = False

class FlowResNetEncoder(nn.Module):
    """
    Torchvision ResNet backbone for flow-based motion discrimination.

    - Modifies conv1 to accept 'in_ch' (e.g., 2 for (u*C, v*C), or 3 if you also concat C).
    - Returns logits for BCE when given sequences (B,T,C,H,W) with internal temporal average.
    - Also exposes 'encode_pairs' to get per-pair embeddings from (N,C,H,W).

    Args:
      backbone: 'resnet18' | 'resnet34' | 'resnet50' | ...
      in_ch:    number of input channels (2 by default)
      out_dim:  embedding dimension after backbone (projected)
      mlp_hidden: tuple for the 3-layer head, e.g., (256, 64)
      pretrained_backbone: if True, load ImageNet weights and adapt conv1
      temporal_weighted: if True, forward accepts 'weights' for confidence-weighted mean over T
      dropout:  dropout in MLP
    """
    def __init__(self,
                 backbone: str = "resnet18",
                 in_ch: int = 3,
                 out_dim: int = 256,
                 mlp_hidden=(256, 64),
                 pretrained_backbone: bool = False,
                 temporal_weighted: bool = False,
                 dropout: float = 0.1,
                 pretrained_resnet_ckpt = None):
        super().__init__()
        self.temporal_weighted = temporal_weighted

        # 1) Build and adapt torchvision ResNet
        resnet = _build_resnet(backbone, pretrained=pretrained_backbone, pretrained_ckpt=pretrained_resnet_ckpt)
        _surgery_first_conv(resnet, in_ch=in_ch)
        _disable_inplace_relu(resnet)

        # Extract feature dim before the original 'fc'
        feat_dim = resnet.fc.in_features
        # Replace fc with Identity to output pooled features
        resnet.fc = nn.Identity()
        self.backbone = resnet
        # 2) Projection head to desired embedding size
        if out_dim != feat_dim:
            self.proj = nn.Linear(feat_dim, out_dim)
        else:
            self.proj = nn.Identity()

        # 3) 3-layer prediction head (MLP) → scalar logit for BCE
        mlp_layers = []
        d_in = out_dim
        for h in mlp_hidden:
            mlp_layers += [nn.Linear(d_in, h), nn.GELU(), nn.Dropout(dropout)]
            d_in = h
        mlp_layers += [nn.Linear(d_in, 1)]
        self.mlp = nn.Sequential(*mlp_layers)

    def aggregate_temporal(self, feats_bt: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        """
        feats_bt: (B, T, D)
        weights:  (B, T) optional weights (e.g., per-pair mean confidence)
        returns:  (B, D)
        """
        if weights is None or not self.temporal_weighted:
            return feats_bt.mean(dim=1)
        w = (weights / (weights.sum(dim=1, keepdim=True) + 1e-6)).unsqueeze(-1)  # (B,T,1)
        return (feats_bt * w).sum(dim=1)

    def forward(self, x, weights: torch.Tensor = None):
        """
        Args:
            x: (B,T,C,H,W)
            weights: (B,T) optional weights (e.g., per-pair mean confidence)
        Returns:
            logits: (N,) or (B,) logits
        """
        assert x.dim() == 5, "Expected (B,T,C,H,W)"
        B, T, C, H, W = x.shape
        x_flat = rearrange(x, 'b t c h w -> (b t) c h w')
        feat = self.backbone(x_flat)         # ((b t), feat_dim)
        feat = self.proj(feat)          # ((b t), out_dim)
        feat = rearrange(feat, '(b t) d -> b t d', b=B, t=T)

        feat = self.aggregate_temporal(feat, weights=weights)  # (B,D)
        logits = self.mlp(feat).squeeze(1)    # (B,)
        return logits

class FlowViTEncoder(nn.Module):
    """
    timm ViT backbone for flow-based motion discrimination.

    Matches FlowResNetEncoder behavior:
      - forward(x) expects (B,T,C,H,W)
      - returns logits (B,) for BCE
      - optional confidence-weighted temporal aggregation via weights (B,T)
      - encode_pairs(x2d) accepts (N,C,H,W) and returns (N,out_dim)

    Fixed resolution case:
      - default img_size=(320,576) and dynamic_img_size=True so pos-embed is handled safely.
    """

    _SIZE_TO_MODEL = {
        "small": "vit_small_patch16_224",
        "base":  "vit_base_patch16_224",
        "large": "vit_large_patch16_224",
    }

    def __init__(
        self,
        vit_size: str = "base",
        in_ch: int = 2,
        img_size: tuple[int, int] = (320, 576),  # (H,W), fixed in your setup
        temporal_weighted: bool = False,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        pool: str = "cls",  # "cls" or "mean"
    ):
        super().__init__()
        self.temporal_weighted = temporal_weighted
        assert pool in ("cls", "mean")
        self.pool = pool

        assert vit_size in self._SIZE_TO_MODEL, f"vit_size must be one of {list(self._SIZE_TO_MODEL.keys())}"
        model_name = self._SIZE_TO_MODEL[vit_size]

        # timm pooling names: 'token' (cls token) or 'avg' (mean pool)
        global_pool = "token" if pool == "cls" else "avg"

        create_kwargs = dict(
            pretrained=False,        # no pretrained weights
            in_chans=in_ch,          # adapt patch-embed input channels
            num_classes=0,           # remove classifier head
            global_pool=global_pool, # controls pooled feature output
            drop_rate=dropout,
            attn_drop_rate=attn_dropout,
            img_size=img_size,       # your fixed (H,W)
        )

        # enable if supported by your timm version (recommended for non-224)
        try:
            create_kwargs["dynamic_img_size"] = True
        except Exception:
            pass

        self.backbone = timm.create_model(model_name, **create_kwargs)

        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            feat_dim = getattr(self.backbone, "embed_dim")

        self.head = nn.Linear(feat_dim, 1)

    def aggregate_temporal(self, feats_bt: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        """
        feats_bt: (B, T, D)
        weights:  (B, T) optional weights (e.g., per-pair mean confidence)
        returns:  (B, D)
        """
        if weights is None or not self.temporal_weighted:
            return feats_bt.mean(dim=1)
        w = (weights / (weights.sum(dim=1, keepdim=True) + 1e-6)).unsqueeze(-1)  # (B,T,1)
        return (feats_bt * w).sum(dim=1)

    def encode_pairs(self, x2d: torch.Tensor) -> torch.Tensor:
        """
        x2d: (N, C, H, W)
        returns: (N, out_dim)
        """
        # Prefer forward_features + forward_head for compatibility across timm ViT variants
        feats = self.backbone.forward_features(x2d)
        feats = self.backbone.forward_head(feats, pre_logits=True)
        return feats

    def forward(self, x: torch.Tensor, weights: torch.Tensor = None):
        """
        x: (B,T,C,H,W)
        weights: (B,T) optional weights
        returns: logits (B,)
        """
        assert x.dim() == 5, "Expected (B,T,C,H,W)"
        B, T, C, H, W = x.shape
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")
        feat = self.encode_pairs(x_flat)  # ((b t), out_dim)
        feat = rearrange(feat, "(b t) d -> b t d", b=B, t=T)
        feat = self.aggregate_temporal(feat, weights=weights)  # (B, out_dim)
        logits = self.head(feat).squeeze(1)                     # (B,)
        return logits

class OpticalFlowLayer(nn.Module):
    def __init__(self, cfg_file, ckpt_file):
        super().__init__()
        # init SEA-RAFT model
        with open(cfg_file, 'r') as file:
            conf = json.load(file)
            model_args = argparse.Namespace()
            arg_dict = model_args.__dict__
            for key, value in conf.items():
                arg_dict[key] = value
        self.model_args = model_args
        self.model = RAFT(self.model_args)
        # load checkpoint
        state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.freeze_params()

    @torch.no_grad()
    def forward(self, image1, image2, normalize_flow=True):
        """
        image1: (B, C, H, W)
        image2: (B, C, H, W)
        Return:
            flow: (B, 2, H, W)
        """
        with torch.backends.cudnn.flags(enabled=False):
            output = self.model(image1, image2, iters=self.model_args.iters, test_mode=True)
        flow_final = output['flow'][-1]
        info_final = output['info'][-1]
        confidence_map = self.extract_sea_raft_confidence(info_final)
        if normalize_flow:
            flow_final = self.normalize_flow(flow_final)

        return flow_final, confidence_map
    
    def normalize_flow(self, flow_map, eps=1e-6):
        """
        Args:
            flow_map: (B, 2, H, W)
        Return:
            normalized_flow_map: (B, 2, H, W)
        """
        flow_mag = torch.sqrt(flow_map[:,0:1, ...]**2 + flow_map[:,1:2, ...]**2)
        denom = (flow_mag**2).mean(dim=(2,3), keepdim=True)  # (B,1,1,1)
        denom = torch.sqrt(denom)
        flow_map = flow_map / (denom + eps)
        
        return flow_map

    # @torch.no_grad()
    def extract_sea_raft_confidence(self, info_final,
                                    percentile=0.8, eps=1e-6):
        """
        info_final:  [B, 4, H, W] from SEA-RAFT (test_mode=True path)
            channels: [w_logit0, w_logit1, raw_log_b0, raw_log_b1]
        percentile: keep the most confident (1 - percentile) fraction as M=1
                    e.g., 0.8 -> keep ~80% lowest-uncertainty pixels
        Returns:
        M:   [B, 1, H, W] binary mask (1 = confident)
        C:   [B, 1, H, W] soft confidence in (0,1]
        unc: [B, 1, H, W] uncertainty proxy (mixture expected |ε|)
        """
        B, C, H, W = info_final.shape
        assert C == 4, f"Expected 4 channels in info, got {C}"
        w_logit = info_final[:, 0:2, ...]          # mixture logits
        raw_logb = info_final[:, 2:4, ...]         # raw log-scale params

        # Reproduce training-time clamping to valid ranges
        logb = torch.zeros_like(raw_logb)
        # Large-b component (index 0): clamp to [0, var_max]
        logb[:, 0] = torch.clamp(raw_logb[:, 0], min=0.0, max=self.model.args.var_max)
        # Small-b component (index 1): clamp to [var_min, 0]
        logb[:, 1] = torch.clamp(raw_logb[:, 1], min=self.model.args.var_min, max=0.0)

        pi = F.softmax(w_logit, dim=1)       # [B,2,H,W]
        b  = torch.exp(logb)                 # [B,2,H,W]

        # Mixture expected absolute deviation as uncertainty
        unc = (pi * b).sum(dim=1, keepdim=True)   # [B,1,H,W]

        # Soft confidence in (0,1]: scale-free using per-sample median
        s = unc.flatten(2).median(dim=2, keepdim=True).values.view(B, 1, 1, 1).clamp_min(eps)
        C = torch.exp(-unc / s).clamp(min=eps, max=1.0)  # higher = more confident
        # Binary mask by per-sample percentile on uncertainty (lower unc = keep)
        # q is the uncertainty threshold at the chosen percentile
        # q = torch.quantile(unc.flatten(2), q=percentile, dim=2, keepdim=True).view(B, 1, 1, 1)
        # M = (unc <= q).float()
        return C
    
    def freeze_params(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

class OpticalFlowDiscriminator(nn.Module):
    def __init__(self, raft_cfg_file='spring-M.json', raft_ckpt_file=None,
                 flow_resent_backbone='resnet50', flow_resent_in_ch=3, flow_resent_out_dim=256, flow_resent_mlp_hidden=(256, 64), 
                 flow_resent_pretrained_backbone=False, flow_resent_temporal_weighted=False, flow_resent_dropout=0.1,
                 pretrained_resnet_ckpt=None):
        super().__init__()
        self.flow_layer = OpticalFlowLayer(raft_cfg_file, raft_ckpt_file)
        self.flow_resent = FlowResNetEncoder(backbone=flow_resent_backbone, in_ch=flow_resent_in_ch, out_dim=flow_resent_out_dim, 
                                             mlp_hidden=flow_resent_mlp_hidden, pretrained_backbone=flow_resent_pretrained_backbone, 
                                             temporal_weighted=flow_resent_temporal_weighted, dropout=flow_resent_dropout, pretrained_resnet_ckpt=pretrained_resnet_ckpt)

    def forward(self, vid):
        """
        Args:
            vid: (B, T, C, H, W), pixel values are in [0, 255]
        Return:
            logits: (B,) logits
        """
        # detect the video values are in [0, 255]
        if vid.max() <= 1.:
            vid = vid * 255.
        assert vid.min() >= 0. and vid.max() <= 255., f"Pixel values should be in [0, 255], but got {vid.min()} and {vid.max()}"

        B, T, C, H, W = vid.shape
        vid1 = vid[:, :-1, ...] #(B, T-1, C, H, W)
        vid2 = vid[:, 1:, ...] #(B, T-1, C, H, W)
        vid1 = rearrange(vid1, 'b t c h w -> (b t) c h w')
        vid2 = rearrange(vid2, 'b t c h w -> (b t) c h w')
        flow, confidence_map = self.flow_layer(vid1, vid2) #(b*t, 2, h, w) and (b*t, 1, h, w) respectively

        x = torch.cat([flow, confidence_map], dim=1) #(b*t, 3, h, w)
        x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T-1)

        logits = self.flow_resent(x)
        return logits

class BrightnessConsistencyMapDiscriminator(nn.Module):
    def __init__(
        self,
        raft_cfg_file='spring-M.json',
        raft_ckpt_file=None,
        sobel_filter_size=3,

        # NEW: selector
        encoder_type: str = "vit",

        # ViT args (existing)
        vit_size='base',
        vit_ch_in=2,
        vit_in_img_size=(320, 576),
        vit_temporal_weighted=False,
        vit_dropout=0.1,
        vit_attn_dropout=0.0,
        vit_pool='cls',

        # ResNet args (old-style)
        flow_resent_backbone='resnet50',
        flow_resent_in_ch=2,
        flow_resent_out_dim=256,
        flow_resent_mlp_hidden=(256, 64),
        flow_resent_pretrained_backbone=False,
        flow_resent_temporal_weighted=False,
        flow_resent_dropout=0.1,
        pretrained_resnet_ckpt=None,
        in_img_downsample_ratio=1.,
    ):
        super().__init__()
        self.encoder_type = (encoder_type or "vit").lower()

        self.sobel_filter_size = sobel_filter_size
        self.sobel_layer = SobelGradientLayer(filter_size=sobel_filter_size)
        self.temporal_layer = TemporalGradientLayer(sobel_filter_size=sobel_filter_size)

        self.flow_layer = OpticalFlowLayer(raft_cfg_file, raft_ckpt_file)

        self.in_img_downsample_ratio = in_img_downsample_ratio
        assert self.in_img_downsample_ratio >= 1.0, "invalid input image resolution downsample ratio"
        if self.encoder_type == "vit":
            # OmegaConf may provide [H, W]
            # if isinstance(vit_in_img_size, (list, tuple)):
            self.vit_in_img_size = tuple(vit_in_img_size)
            if self.in_img_downsample_ratio > 1:
                self.vit_in_img_size = (int(vit_in_img_size[0]/self.in_img_downsample_ratio), int(vit_in_img_size[1]/self.in_img_downsample_ratio))
            # IMPORTANT: keep attribute name = flow_vit for old ViT checkpoints
            self.flow_vit = FlowViTEncoder(
                vit_size=vit_size,
                in_ch=vit_ch_in,
                img_size=self.vit_in_img_size,
                temporal_weighted=vit_temporal_weighted,
                dropout=vit_dropout,
                attn_dropout=vit_attn_dropout,
                pool=vit_pool,
            )

        elif self.encoder_type == "resnet":
            # IMPORTANT: keep attribute name = flow_resent for old ResNet checkpoints
            self.flow_resent = FlowResNetEncoder(
                backbone=flow_resent_backbone,
                in_ch=flow_resent_in_ch,
                out_dim=flow_resent_out_dim,
                mlp_hidden=flow_resent_mlp_hidden,
                pretrained_backbone=flow_resent_pretrained_backbone,
                temporal_weighted=flow_resent_temporal_weighted,
                dropout=flow_resent_dropout,
                pretrained_resnet_ckpt=pretrained_resnet_ckpt,
            )

        else:
            raise ValueError(f"Unknown encoder_type={encoder_type}. Use 'vit' or 'resnet'.")

    def forward(self, vid):
        """
        Args:
            vid: (B, T, C, H, W), pixel values are in [0, 255]
        Return:
            logits: (B,) logits
        """
        B, T, _, _, _ = vid.shape
        # detect the video values are in [0, 255]
        if vid.max() <= 1.:
            vid = vid * 255.
        assert vid.min() >= 0. and vid.max() <= 255., f"Pixel values should be in [0, 255], but got {vid.min()} and {vid.max()}"

        # resize the input video
        vid_bt = rearrange(vid, 'b t c h w -> (b t) c h w')
        vid_bt = F.interpolate(
            vid_bt,
            size=self.vit_in_img_size,
            mode='bilinear',
            align_corners=False,
        )
        vid = rearrange(vid_bt, '(b t) c h w -> b t c h w', b=B, t=T)

        vid1 = vid[:, :-1, ...] #(B, T-1, C, H, W)
        vid2 = vid[:, 1:, ...] #(B, T-1, C, H, W)
        vid1 = rearrange(vid1, 'b t c h w -> (b t) c h w')
        vid2 = rearrange(vid2, 'b t c h w -> (b t) c h w')
        flow, confidence_map = self.flow_layer(vid1, vid2) #(b*t, 2, h, w) and (b*t, 1, h, w) respectively

        # extract the spatial and temporal gradient
        grad_h, grad_w = self.sobel_layer(vid1)
        grad_t = self.temporal_layer(vid1, vid2)

        u, v = flow[:, 0:1, ...], flow[:, 1:, ...]
        brightness_consistency_map = grad_h * v + grad_w * u + grad_t
        x = torch.cat([brightness_consistency_map, confidence_map], dim=1) #(b*t, 2, h, w)
        x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T-1)

        if self.encoder_type == 'resnet':
            logits = self.flow_resent(x)
        elif self.encoder_type == 'vit':
            proj = self.flow_vit.backbone.patch_embed.proj
            x = x.to(dtype=proj.weight.dtype)
            logits = self.flow_vit(x)

        return logits

    def visualize_map(self, map, save_dir):
        """
        Args:
            map: (B, 1, H, W)
        """
        # normalize to range [0, 1] firstly
        map = (map - map.min()) / (map.max() - map.min())
        # convert to uint8
        map = (map * 255.).to(torch.uint8)
        # convert to numpy array
        map = map.cpu().numpy()
        # save the map
        for b in range(map.shape[0]):
            cv2.imwrite(f'{save_dir}/brightness_consistency_map_{b}.png', map[b, 0, ...])


if __name__ == '__main__':
    import os
    import cv2
    import numpy as np
    import torch

    def save_tensor_map_png(x: torch.Tensor, out_path: str, *,
                            normalize: str = "minmax",  # "minmax" | "absmax" | "none"
                            eps: float = 1e-8) -> None:
        """
        Save a tensor map to PNG. Accepts shapes [H,W], [1,H,W], [B,1,H,W] (uses first item).
        - minmax: scales x linearly to [0,255]
        - absmax: scales by max(|x|) to [0,255] with 0 mapped to 127 (good for signed flow/grad)
        - none: assumes x already in [0,255] or [0,1] (will clip)
        """
        t = x.detach()
        # pick first batch/channel if present
        if t.ndim == 4:   # [B,C,H,W]
            t = t[0, 0]
        elif t.ndim == 3: # [C,H,W] or [1,H,W]
            t = t[0]
        elif t.ndim != 2:
            raise ValueError(f"Unsupported shape: {tuple(t.shape)}")

        a = t.float().cpu().numpy()

        if normalize == "minmax":
            mn, mx = float(a.min()), float(a.max())
            a = (a - mn) / (mx - mn + eps)
            a = (a * 255.0).round()
        elif normalize == "absmax":
            m = float(np.max(np.abs(a)))
            a = a / (m + eps)              # [-1,1]
            a = (a * 127.5 + 127.5).round() # -> [0,255], 0 -> 127
        elif normalize == "none":
            # if looks like [0,1], upscale
            if a.max() <= 1.0:
                a = (a * 255.0).round()
        else:
            raise ValueError(f"Unknown normalize mode: {normalize}")

        img = np.clip(a, 0, 255).astype(np.uint8)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        ok = cv2.imwrite(out_path, img)  # grayscale PNG
        if not ok:
            raise IOError(f"cv2.imwrite failed for {out_path}")

    from matplotlib import pyplot as plt
    image1 = cv2.imread("/path/to/camel_frames/frame_000003.png")
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread("/path/to/camel_frames/frame_000006.png")
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    image1 = F.interpolate(image1, size=(320, 576), mode='bilinear', align_corners=False)
    image2 = F.interpolate(image2, size=(320, 576), mode='bilinear', align_corners=False)

    layer = SobelGradientLayer(filter_size=2)
    grad_h, grad_w = layer(image1)
    print(grad_h.shape, grad_w.shape)

    # fig, ax = plt.subplots()
    # ax.imshow(grad_h[0, 0, ...].cpu().numpy(), cmap = 'gray')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # fig.savefig(f'gradient_h.png', bbox_inches='tight', dpi=300)

    # fig, ax = plt.subplots()
    # ax.imshow(grad_w[0, 0, ...].cpu().numpy(), cmap = 'gray')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # fig.savefig(f'gradient_w.png', bbox_inches='tight', dpi=300)

    temporal_layer = TemporalGradientLayer(sobel_filter_size=3)
    grad_t = temporal_layer(image1, image2)
    print(grad_t.shape)
    # fig, ax = plt.subplots()
    # ax.imshow(grad_t[0, 0, ...].cpu().numpy(), cmap = 'gray')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # fig.savefig(f'gradient_t.png', bbox_inches='tight', dpi=300)

    flow_layer = OpticalFlowLayer(cfg_file='/path/to/sea_raft_checkpoints/spring-M.json', ckpt_file='/path/to/sea_raft_checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth')
    flow, info = flow_layer(image1, image2)
    print('flow', flow.shape, info.shape)

    # fig, ax = plt.subplots()
    # ax.imshow(flow[0, 0, ...].cpu().numpy(), cmap = 'gray')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # fig.savefig(f'u.png', bbox_inches='tight', dpi=300)
    # fig, ax = plt.subplots()
    # ax.imshow(flow[0, 1, ...].cpu().numpy(), cmap = 'gray')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # fig.savefig(f'v.png', bbox_inches='tight', dpi=300)

    save_tensor_map_png(grad_h, "gradient_h.png", normalize="minmax")
    save_tensor_map_png(grad_w, "gradient_w.png", normalize="minmax")
    save_tensor_map_png(grad_t, "gradient_t.png", normalize="minmax")

    save_tensor_map_png(flow[:, 0:1], "u.png", normalize="absmax")  # signed
    save_tensor_map_png(flow[:, 1:2], "v.png", normalize="absmax")  # signed

    # flow_discriminator = OpticalFlowDiscriminator(raft_cfg_file='/path/to/SEA-RAFT/config/eval/spring-M.json', raft_ckpt_file='/path/to/Tartan-C-T-TSKH-spring540x960-M.pth')
    # flow_discriminator = flow_discriminator.to('cuda:0')
    # vid = torch.randn(3, 16, 3, 256, 256).to('cuda:0')
    # vid = torch.clamp(vid, min=0., max=1.)
    # logits = flow_discriminator(vid)
    # print(logits.shape)

    # flow_resent = FlowResNetEncoder(backbone='resnet50', in_ch=3, out_dim=256, mlp_hidden=(256, 64), pretrained_backbone=True, temporal_weighted=False, dropout=0.1)
    # flow_resent = flow_resent.to('cuda:0')
    # logits = flow_resent(x)
    # print(logits.shape)

    brightness_consistency_discriminator = BrightnessConsistencyMapDiscriminator(raft_cfg_file='/path/to/sea_raft_checkpoints/spring-M.json', raft_ckpt_file='/path/to/sea_raft_checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth')
    brightness_consistency_discriminator = brightness_consistency_discriminator.to('cuda:0')
    vid = torch.randn(3, 16, 3, 320, 576).to('cuda:0')
    vid = torch.clamp(vid, min=0., max=1.)
    logits = brightness_consistency_discriminator(vid)
    print(logits.shape)