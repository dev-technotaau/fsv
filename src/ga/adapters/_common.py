"""Shared building blocks for adapters — dataset, decoders, training, eval.

Public API:
  Data:     FenceDataset, build_dataloaders
  Losses:   focal_loss, dice_loss, boundary_loss, lovasz_hinge, tversky_loss,
            ohem_ce_loss, combined_loss
  Decoders: build_upernet_head, build_m2f_lite_decoder, build_mask2former_decoder
  Train:    train_proxy (now with warmup+scheduler+accumulation+EMA)
  Eval:     _eval_iou_generic, dump_predictions, tta_predict
  Utils:    EMAHelper, file_logger, pick_device, _morph_close
"""
from __future__ import annotations

import copy
import math
import random as _random
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


# ============================================================
# DATASET
# ============================================================

class FenceDataset:
    """Image/mask pair dataset with 3 augmentation levels."""

    def __init__(self, images_dir: Path, masks_dir: Path, input_size: int,
                 augment: str = "medium", stride: int = 1):
        import torchvision.transforms.functional as TF
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.input_size = input_size
        self.augment = augment
        self.pairs: list[tuple[Path, Path]] = []
        for p in sorted(self.images_dir.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            mp = self.masks_dir / (p.stem + ".png")
            if mp.exists():
                self.pairs.append((p, mp))
        if stride > 1:
            self.pairs = self.pairs[::stride]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image
        img_p, mask_p = self.pairs[idx]
        img = Image.open(img_p).convert("RGB").resize((self.input_size, self.input_size), Image.BILINEAR)
        msk = Image.open(mask_p).convert("L").resize((self.input_size, self.input_size), Image.NEAREST)

        if self.augment != "none":
            if _random.random() < 0.5:
                img, msk = TF.hflip(img), TF.hflip(msk)
            if self.augment in ("medium", "aggressive"):
                if _random.random() < 0.3:
                    img = TF.adjust_brightness(img, _random.uniform(0.75, 1.25))
                if _random.random() < 0.3:
                    img = TF.adjust_contrast(img, _random.uniform(0.8, 1.2))
            if self.augment == "aggressive":
                if _random.random() < 0.3:
                    img = TF.adjust_saturation(img, _random.uniform(0.7, 1.3))
                if _random.random() < 0.3:
                    angle = _random.uniform(-10, 10)
                    img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
                    msk = TF.rotate(msk, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        msk_t = torch.from_numpy((np.array(msk) > 127).astype("int64"))
        return img_t, msk_t


def build_dataloaders(data_cfg, input_size, augment, batch_size=2, seed=42, num_workers=0):
    import torch
    from torch.utils.data import DataLoader, random_split
    ds = FenceDataset(data_cfg.images_dir, data_cfg.masks_dir, input_size, augment=augment)
    if len(ds) < 10:
        raise ValueError(f"Dataset too small: {len(ds)} pairs")
    val_n = max(2, int(len(ds) * data_cfg.val_split))
    train_n = len(ds) - val_n
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [train_n, val_n], generator=gen)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=False),
    )


# ============================================================
# LOSSES
# ============================================================

def focal_loss(logits, target, alpha: float = 0.25, gamma: float = 2.0):
    import torch
    import torch.nn.functional as F
    t = target.float()
    if logits.dim() == 4 and t.dim() == 3:
        t = t.unsqueeze(1)
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    pt = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()


def dice_loss(logits, target):
    import torch
    t = target.float()
    if logits.dim() == 4 and t.dim() == 3:
        t = t.unsqueeze(1)
    probs = torch.sigmoid(logits)
    inter = (probs * t).sum((2, 3))
    union = probs.sum((2, 3)) + t.sum((2, 3))
    return (1 - (2 * inter + 1) / (union + 1)).mean()


def boundary_loss(logits, target):
    import torch
    import torch.nn.functional as F
    t = target.float()
    if logits.dim() == 4 and t.dim() == 3:
        t = t.unsqueeze(1)
    kx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                      device=logits.device, dtype=logits.dtype)
    ky = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]],
                      device=logits.device, dtype=logits.dtype)
    def edges(x):
        return torch.sqrt(F.conv2d(x, kx, padding=1) ** 2
                          + F.conv2d(x, ky, padding=1) ** 2 + 1e-8)
    p_edge = edges(torch.sigmoid(logits))
    t_edge = edges(t)
    return F.l1_loss(p_edge, t_edge)


def lovasz_hinge(logits, target):
    import torch
    t = target.float()
    if logits.dim() == 4 and t.dim() == 3:
        t = t.unsqueeze(1)
    signs = 2.0 * t - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors.view(-1), descending=True)
    gt_sorted = t.view(-1)[perm]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(torch.nn.functional.relu(errors_sorted), grad)


def _lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def tversky_loss(logits, target, alpha: float = 0.5, beta: float = 0.5):
    import torch
    t = target.float()
    if logits.dim() == 4 and t.dim() == 3:
        t = t.unsqueeze(1)
    probs = torch.sigmoid(logits)
    tp = (probs * t).sum((2, 3))
    fp = (probs * (1 - t)).sum((2, 3))
    fn = ((1 - probs) * t).sum((2, 3))
    return (1 - (tp + 1) / (tp + alpha * fp + beta * fn + 1)).mean()


def ohem_ce_loss(logits, target, ratio: float = 0.7, num_classes: int = 2):
    import torch
    import torch.nn.functional as F
    if logits.shape[1] == 1:
        logits = torch.cat([-logits, logits], dim=1)
    t = target.long()
    ce = F.cross_entropy(logits, t, reduction="none").view(-1)
    k = max(1, int(ratio * ce.numel()))
    vals, _ = ce.topk(k)
    return vals.mean()


def ssim_loss(logits, target, window_size: int = 7):
    """Soft SSIM loss on sigmoid(logits) vs target. Useful for structural similarity."""
    import torch
    import torch.nn.functional as F
    t = target.float()
    if logits.dim() == 4 and t.dim() == 3:
        t = t.unsqueeze(1)
    pred = torch.sigmoid(logits)
    # simple gaussian window
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * (window_size / 6.0) ** 2))
    g = (g / g.sum()).view(1, 1, 1, -1)
    gy = g.transpose(-1, -2)
    kernel = (g * gy).expand(1, 1, window_size, window_size)
    mu_x = F.conv2d(pred, kernel, padding=window_size // 2)
    mu_y = F.conv2d(t, kernel, padding=window_size // 2)
    sig_x = F.conv2d(pred * pred, kernel, padding=window_size // 2) - mu_x * mu_x
    sig_y = F.conv2d(t * t, kernel, padding=window_size // 2) - mu_y * mu_y
    sig_xy = F.conv2d(pred * t, kernel, padding=window_size // 2) - mu_x * mu_y
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x + sig_y + C2) + 1e-8)
    return (1 - ssim.clamp(0, 1)).mean()


def combined_loss(logits, target, weights: dict[str, float]):
    """Weighted combination of any subset of the above losses."""
    import torch
    total = logits.sum() * 0.0
    if weights.get("focal", 0) > 0:
        total = total + weights["focal"] * focal_loss(logits, target)
    if weights.get("dice", 0) > 0:
        total = total + weights["dice"] * dice_loss(logits, target)
    if weights.get("boundary", 0) > 0:
        total = total + weights["boundary"] * boundary_loss(logits, target)
    if weights.get("lovasz", 0) > 0:
        total = total + weights["lovasz"] * lovasz_hinge(logits, target)
    if weights.get("tversky", 0) > 0:
        a = weights.get("tversky_alpha", 0.5)
        b = weights.get("tversky_beta", 0.5)
        total = total + weights["tversky"] * tversky_loss(logits, target, a, b)
    if weights.get("ohem", 0) > 0:
        total = total + weights["ohem"] * ohem_ce_loss(logits, target, weights.get("ohem_ratio", 0.7))
    if weights.get("ssim", 0) > 0:
        total = total + weights["ssim"] * ssim_loss(logits, target)
    return total


# ============================================================
# DECODERS
# ============================================================

def build_upernet_head(in_channels: list[int], hidden_channels: int = 384, num_classes: int = 1):
    """UPerNet: PPM on deepest feature + FPN lateral + fuse to single class map."""
    import torch
    import torch.nn as nn

    class PPM(nn.Module):
        def __init__(self, in_c, out_c, scales=(1, 2, 3, 6)):
            super().__init__()
            self.branches = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(s),
                    nn.Conv2d(in_c, out_c, 1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                ) for s in scales
            ])
            self.fuse = nn.Sequential(
                nn.Conv2d(in_c + out_c * len(scales), out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            )

        def forward(self, x):
            feats = [x]
            for b in self.branches:
                feats.append(torch.nn.functional.interpolate(
                    b(x), size=x.shape[-2:], mode="bilinear", align_corners=False))
            return self.fuse(torch.cat(feats, dim=1))

    class UPerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.ppm = PPM(in_channels[-1], hidden_channels)
            self.lateral = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(c, hidden_channels, 1, bias=False),
                    nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True),
                ) for c in in_channels[:-1]
            ])
            self.smooth = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True),
                ) for _ in in_channels[:-1]
            ])
            self.fuse = nn.Sequential(
                nn.Conv2d(hidden_channels * len(in_channels), hidden_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True),
            )
            self.cls = nn.Conv2d(hidden_channels, num_classes, 1)

        def forward(self, feats):
            x = self.ppm(feats[-1])
            multi = [x]
            cur = x
            for i in range(len(feats) - 2, -1, -1):
                lat = self.lateral[i](feats[i])
                cur = lat + torch.nn.functional.interpolate(
                    cur, size=lat.shape[-2:], mode="bilinear", align_corners=False)
                multi.insert(0, self.smooth[i](cur))
            target_size = multi[0].shape[-2:]
            fused = torch.cat([
                torch.nn.functional.interpolate(m, size=target_size, mode="bilinear", align_corners=False)
                for m in multi
            ], dim=1)
            return self.cls(self.fuse(fused))
    return UPerNet()


def build_m2f_lite_decoder(in_dim: int, num_queries: int = 100, mask_feat: int = 256, num_classes: int = 1):
    """Lightweight Mask2Former-ish decoder (single-scale, single-layer attention).
    Kept for backward compat with adapters that don't need the full decoder.
    """
    import torch
    import torch.nn as nn

    class M2FLite(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(in_dim, mask_feat, 1)
            self.attn = nn.MultiheadAttention(mask_feat, num_heads=8, batch_first=True)
            self.query_embed = nn.Embedding(num_queries, mask_feat)
            self.mlp = nn.Sequential(
                nn.Linear(mask_feat, mask_feat), nn.GELU(),
                nn.Linear(mask_feat, mask_feat),
            )
            self.mask_head = nn.Conv2d(mask_feat, num_classes, 1)

        def forward(self, feats):
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]
            B, C, H, W = feats.shape
            x = self.proj(feats)
            tokens = x.flatten(2).transpose(1, 2)
            q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
            attn_out, _ = self.attn(q, tokens, tokens)
            attn_out = self.mlp(attn_out) + attn_out
            gate = attn_out.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
            fused = x * torch.sigmoid(gate)
            return self.mask_head(fused)
    return M2FLite()


def build_mask2former_decoder(
    in_channels: list[int],
    num_queries: int = 100,
    mask_feat: int = 256,
    num_classes: int = 1,
    num_layers: int = 6,
    num_heads: int = 8,
):
    """Real Mask2Former decoder with:
      - Multi-scale pixel decoder (FPN from multi-scale backbone features)
      - N queries cross-attend to pixel features at each scale
      - Masked cross-attention (attn mask = previous layer's coarse mask prediction)
      - Iterative refinement: each layer predicts class_logits + mask_logits
      - Returns list of (class_logits, mask_logits) per layer for deep supervision
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MultiScalePixelDecoder(nn.Module):
        """FPN-style decoder that produces `mask_feat`-channel maps at multiple scales."""
        def __init__(self):
            super().__init__()
            self.laterals = nn.ModuleList([
                nn.Conv2d(c, mask_feat, 1) for c in in_channels
            ])
            self.fpn_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(mask_feat, mask_feat, 3, padding=1, bias=False),
                    nn.GroupNorm(32, mask_feat), nn.ReLU(inplace=True),
                ) for _ in in_channels
            ])
            self.mask_feature_conv = nn.Conv2d(mask_feat, mask_feat, 1)

        def forward(self, feats: list):
            """feats: list ordered shallow→deep."""
            laterals = [lat(f) for lat, f in zip(self.laterals, feats)]
            # top-down pathway
            for i in range(len(laterals) - 2, -1, -1):
                laterals[i] = laterals[i] + F.interpolate(
                    laterals[i + 1], size=laterals[i].shape[-2:],
                    mode="bilinear", align_corners=False)
            # FPN smoothing
            outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
            # mask_features = the shallowest (highest resolution) map
            mask_features = self.mask_feature_conv(outs[0])
            # multi-scale memory for cross-attention uses the coarser levels
            multi_scale = outs[1:] if len(outs) > 1 else [outs[0]]
            return mask_features, multi_scale

    class FFN(nn.Module):
        def __init__(self, d, mult=4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, d * mult), nn.GELU(),
                nn.Linear(d * mult, d),
            )

        def forward(self, x):
            return self.net(x)

    class M2FLayer(nn.Module):
        """One transformer decoder layer: masked cross-attn → self-attn → FFN."""
        def __init__(self):
            super().__init__()
            self.cross_attn = nn.MultiheadAttention(mask_feat, num_heads, batch_first=True, dropout=0.1)
            self.self_attn = nn.MultiheadAttention(mask_feat, num_heads, batch_first=True, dropout=0.1)
            self.ffn = FFN(mask_feat)
            self.norm1 = nn.LayerNorm(mask_feat)
            self.norm2 = nn.LayerNorm(mask_feat)
            self.norm3 = nn.LayerNorm(mask_feat)

        def forward(self, queries, kv_tokens, attn_mask=None):
            # masked cross-attn
            q2, _ = self.cross_attn(queries, kv_tokens, kv_tokens, attn_mask=attn_mask)
            queries = self.norm1(queries + q2)
            # self-attn between queries
            q3, _ = self.self_attn(queries, queries, queries)
            queries = self.norm2(queries + q3)
            # FFN
            queries = self.norm3(queries + self.ffn(queries))
            return queries

    class Mask2FormerDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.pixel_decoder = MultiScalePixelDecoder()
            self.query_embed = nn.Embedding(num_queries, mask_feat)
            self.query_feat  = nn.Embedding(num_queries, mask_feat)
            self.level_embed = nn.Embedding(max(1, len(in_channels) - 1), mask_feat)
            self.layers = nn.ModuleList([M2FLayer() for _ in range(num_layers)])
            self.class_head = nn.Linear(mask_feat, num_classes + 1)  # +1 for "no object"
            self.mask_mlp = nn.Sequential(
                nn.Linear(mask_feat, mask_feat), nn.ReLU(inplace=True),
                nn.Linear(mask_feat, mask_feat), nn.ReLU(inplace=True),
                nn.Linear(mask_feat, mask_feat),
            )

        def _prediction_heads(self, queries, mask_features):
            """Per-layer predict: class_logits [B,Q,C+1], mask_logits [B,Q,H,W]."""
            class_logits = self.class_head(queries)
            # mask_embed: project queries to mask_feat-dim; dot with pixel features
            mask_embed = self.mask_mlp(queries)          # [B, Q, mask_feat]
            B, Q, D = mask_embed.shape
            # mask_features: [B, mask_feat, H, W]
            mask_logits = torch.einsum("bqd,bdhw->bqhw", mask_embed, mask_features)
            return class_logits, mask_logits

        def _make_attn_mask(self, mask_logits, target_size):
            """Convert current mask prediction into cross-attention mask.
            attn_mask == True means "blocked"; False means "attend here".
            We attend to pixels where the query's mask is above 0.5.
            """
            import torch
            import torch.nn.functional as F
            # downsample to the kv spatial size
            m = F.interpolate(mask_logits, size=target_size, mode="bilinear", align_corners=False)
            m = (m.sigmoid() > 0.5).float()
            # [B, Q, H, W] -> [B, Q, H*W] -> [B*Q_heads, Q, H*W]
            B, Q, H, W = m.shape
            m = m.flatten(2)
            # block positions where mask is 0 (i.e., attn_mask = ~m)
            attn_mask = ~m.bool()
            # make sure each query has at least one attended position
            all_blocked = attn_mask.all(dim=-1, keepdim=True)
            attn_mask = attn_mask & ~all_blocked
            # expand to per-head format: [B*num_heads, Q, HW]
            attn_mask = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1).flatten(0, 1)
            return attn_mask

        def forward(self, feats: list):
            import torch
            mask_features, multi_scale = self.pixel_decoder(feats)
            B = mask_features.shape[0]
            queries = self.query_feat.weight.unsqueeze(0).expand(B, -1, -1).contiguous()
            pos_q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1).contiguous()
            queries = queries + pos_q

            # Predict an initial mask (layer 0) so layer 1 can use attn_mask
            class0, mask0 = self._prediction_heads(queries, mask_features)
            layer_outputs = [(class0, mask0)]

            for li, layer in enumerate(self.layers):
                # pick a pyramid level (round-robin)
                level = multi_scale[li % len(multi_scale)]
                B_, C_, H_, W_ = level.shape
                kv = level.flatten(2).transpose(1, 2)  # [B, HW, C]
                # add level embedding
                kv = kv + self.level_embed.weight[li % self.level_embed.num_embeddings].unsqueeze(0).unsqueeze(0)
                attn_mask = self._make_attn_mask(layer_outputs[-1][1], (H_, W_))
                queries = layer(queries, kv, attn_mask=attn_mask)
                cls, mask = self._prediction_heads(queries, mask_features)
                layer_outputs.append((cls, mask))

            # For single-class fence seg: take query with highest class prob, combine masks
            # Return final (class_logits, mask_logits) — adapters can also consume all layers.
            final_cls, final_mask = layer_outputs[-1]
            # Aggregate: weighted sum of mask predictions by fence-class probability
            probs = final_cls.softmax(dim=-1)[..., :num_classes]   # [B, Q, C]
            fused_mask = torch.einsum("bqc,bqhw->bchw", probs, final_mask)
            return {
                "fused_mask": fused_mask,
                "aux_outputs": layer_outputs,   # for deep supervision
                "query_masks": final_mask,      # raw per-query masks (for inference inspection)
            }

    return Mask2FormerDecoder()


# ============================================================
# EMA
# ============================================================

class EMAHelper:
    """Exponential Moving Average of model parameters.
    Applies parameter updates after each step; swap_in()/swap_out() for eval/save.
    """
    def __init__(self, model, decay: float = 0.9999):
        import torch
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self._backup = None

    def update(self, model):
        import torch
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def swap_in(self, model):
        """Copy EMA weights into model; back up original for later swap_out."""
        self._backup = {k: v.clone() for k, v in model.state_dict().items()
                        if k in self.shadow}
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v)

    def swap_out(self, model):
        if self._backup is None:
            return
        sd = model.state_dict()
        for k, v in self._backup.items():
            sd[k].copy_(v)
        self._backup = None


# ============================================================
# TRAIN LOOP — with warmup / scheduler / gradient accumulation / EMA
# ============================================================

def train_proxy(
    backbone,
    decoder,
    feature_extract_fn: Callable,
    train_loader,
    val_loader,
    *,
    device,
    proxy_epochs: int,
    optim,
    loss_weights: dict[str, float],
    input_size: int,
    fitness_cfg,
    logger_fn: Callable[[str], None] = print,
    # new knobs
    warmup_epochs: int = 0,
    scheduler: Optional[str] = None,         # None | "cosine" | "step"
    min_lr: float = 1e-7,
    accumulation_steps: int = 1,
    grad_clip: float = 1.0,
    ema_decay: Optional[float] = None,
    deep_sup_logits_fn: Optional[Callable] = None,
    deep_sup_weights: Optional[list[float]] = None,
    decoder_call: Optional[Callable] = None, # if decoder needs custom calling (e.g., returns dict)
    postprocess_logits_fn: Optional[Callable] = None,  # convert decoder output -> [B,1,H,W] logits
):
    """Generic training loop. Returns (best_val_iou, killed_early_flag)."""
    import torch
    import torch.nn.functional as F

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    early_kill_at = int(proxy_epochs * fitness_cfg.early_kill_at_fraction)
    best_iou = 0.0
    killed_early = False

    # Save base LRs for warmup scheduler
    base_lrs = [pg["lr"] for pg in optim.param_groups]

    # Build scheduler (only cosine/step supported; warmup is handled manually per-epoch)
    sched = None
    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max(1, proxy_epochs - warmup_epochs), eta_min=min_lr)
    elif scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(optim, step_size=max(1, proxy_epochs // 3), gamma=0.5)

    # EMA
    ema = EMAHelper(decoder._mod if hasattr(decoder, "_mod") else decoder,
                    decay=ema_decay) if ema_decay else None

    def _call_decoder(feats):
        if decoder_call is not None:
            return decoder_call(decoder, feats)
        return decoder(feats)

    def _logits_from_out(out):
        if postprocess_logits_fn is not None:
            return postprocess_logits_fn(out)
        return out  # assume tensor

    for ep in range(proxy_epochs):
        # Warmup: scale LRs linearly from 0 → base_lr over warmup_epochs
        if warmup_epochs and ep < warmup_epochs:
            warmup_factor = (ep + 1) / warmup_epochs
            for pg, base in zip(optim.param_groups, base_lrs):
                pg["lr"] = base * warmup_factor
        elif sched is not None and ep == warmup_epochs:
            # reset LRs to base before scheduler takes over
            for pg, base in zip(optim.param_groups, base_lrs):
                pg["lr"] = base

        decoder.train()
        if loss_weights.get("_train_backbone", False) and hasattr(backbone, "train"):
            backbone.train()
        else:
            if hasattr(backbone, "eval"):
                backbone.eval()

        ep_loss, n = 0.0, 0
        optim.zero_grad(set_to_none=True)
        for step, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                feats = feature_extract_fn(backbone, imgs)
                out = _call_decoder(feats)
                logits = _logits_from_out(out)
                logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
                loss = combined_loss(logits, masks, loss_weights)
                if deep_sup_logits_fn is not None and deep_sup_weights:
                    aux_list = deep_sup_logits_fn(feats, out)
                    for w, aux in zip(deep_sup_weights, aux_list):
                        aux = F.interpolate(aux, size=input_size, mode="bilinear", align_corners=False)
                        loss = loss + w * combined_loss(aux, masks, loss_weights)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
                scaler.step(optim); scaler.update()
                optim.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(decoder._mod if hasattr(decoder, "_mod") else decoder)

            ep_loss += loss.item() * accumulation_steps
            n += 1

        # step scheduler after warmup
        if sched is not None and ep >= warmup_epochs:
            sched.step()

        logger_fn(f"ep {ep} avg_loss={ep_loss / max(1, n):.4f} lr={optim.param_groups[0]['lr']:.2e}")

        # Periodic val
        if ep + 1 == early_kill_at or ep == proxy_epochs - 1 or ep % 3 == 0:
            if ema is not None:
                ema.swap_in(decoder._mod if hasattr(decoder, "_mod") else decoder)
            val_iou = _eval_iou_generic(backbone, decoder, feature_extract_fn,
                                        val_loader, device, input_size,
                                        decoder_call=_call_decoder,
                                        logits_fn=_logits_from_out)
            if ema is not None:
                ema.swap_out(decoder._mod if hasattr(decoder, "_mod") else decoder)
            logger_fn(f"ep {ep} val_iou={val_iou:.4f}")
            if ep + 1 == early_kill_at and val_iou < fitness_cfg.early_kill_iou:
                killed_early = True
                return val_iou, killed_early
            best_iou = max(best_iou, val_iou)

    # Swap EMA in permanently for final eval (don't swap back)
    if ema is not None:
        ema.swap_in(decoder._mod if hasattr(decoder, "_mod") else decoder)

    return best_iou, killed_early


def _eval_iou_generic(backbone, decoder, feature_extract_fn, loader, device, input_size,
                     decoder_call=None, logits_fn=None):
    import torch
    import torch.nn.functional as F
    backbone.eval(); decoder.eval()
    ious = []
    decoder_call = decoder_call or (lambda d, f: d(f))
    logits_fn = logits_fn or (lambda o: o)
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            feats = feature_extract_fn(backbone, imgs)
            out = decoder_call(decoder, feats)
            logits = logits_fn(out)
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
            pred = (torch.sigmoid(logits) > 0.5).squeeze(1)
            inter = (pred & masks.bool()).sum(dim=(1, 2))
            union = (pred | masks.bool()).sum(dim=(1, 2))
            iou = (inter.float() / union.clamp(min=1).float()).mean().item()
            ious.append(iou)
    return float(np.mean(ious)) if ious else 0.0


# ============================================================
# TTA
# ============================================================

def tta_predict(
    backbone, decoder, feature_extract_fn, img_tensor, *,
    device, input_size: int, mode: str = "hflip+vflip",
    decoder_call=None, logits_fn=None,
):
    """Test-Time Augmentation. img_tensor: [B,3,H,W]. Returns avg prob [B,1,H,W].
    mode: "none" | "hflip" | "hflip+vflip" | "hflip+vflip+rot90"
    """
    import torch
    import torch.nn.functional as F
    decoder_call = decoder_call or (lambda d, f: d(f))
    logits_fn = logits_fn or (lambda o: o)

    aug_fns: list[tuple[Callable, Callable]] = []   # (forward, inverse)
    aug_fns.append((lambda x: x, lambda x: x))
    if mode in ("hflip", "hflip+vflip", "hflip+vflip+rot90"):
        aug_fns.append((lambda x: torch.flip(x, dims=[-1]),
                        lambda x: torch.flip(x, dims=[-1])))
    if mode in ("hflip+vflip", "hflip+vflip+rot90"):
        aug_fns.append((lambda x: torch.flip(x, dims=[-2]),
                        lambda x: torch.flip(x, dims=[-2])))
    if mode == "hflip+vflip+rot90":
        aug_fns.append((lambda x: torch.rot90(x, 1, dims=(-2, -1)),
                        lambda x: torch.rot90(x, -1, dims=(-2, -1))))

    backbone.eval(); decoder.eval()
    probs_sum = None
    with torch.no_grad():
        for fwd, inv in aug_fns:
            x = fwd(img_tensor)
            feats = feature_extract_fn(backbone, x)
            out = decoder_call(decoder, feats)
            logits = logits_fn(out)
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
            prob = torch.sigmoid(logits)
            prob = inv(prob)
            probs_sum = prob if probs_sum is None else probs_sum + prob
    return probs_sum / len(aug_fns)


# ============================================================
# POST-PROCESSING / DUMP
# ============================================================

def _morph_close(binary, k: int):
    try:
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    except ImportError:
        from scipy.ndimage import binary_closing
        return binary_closing(binary, iterations=1).astype("uint8")


def dump_predictions(
    backbone, decoder, feature_extract_fn, loader,
    *, device, input_size: int, out_dir: Path,
    threshold: float = 0.5, morph_k: int = 0,
    decoder_call=None, logits_fn=None,
    tta_mode: str = "none",
) -> Path:
    import torch
    import torch.nn.functional as F
    from PIL import Image

    preds_dir = out_dir / "preds"
    gt_dir = out_dir / "gt"
    preds_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    decoder_call = decoder_call or (lambda d, f: d(f))
    logits_fn = logits_fn or (lambda o: o)

    backbone.eval(); decoder.eval()
    with torch.no_grad():
        for bi, (imgs, masks) in enumerate(loader):
            imgs = imgs.to(device)
            if tta_mode != "none":
                probs = tta_predict(backbone, decoder, feature_extract_fn, imgs,
                                    device=device, input_size=input_size, mode=tta_mode,
                                    decoder_call=decoder_call, logits_fn=logits_fn).cpu().numpy()
            else:
                feats = feature_extract_fn(backbone, imgs)
                out = decoder_call(decoder, feats)
                logits = logits_fn(out)
                logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
                probs = torch.sigmoid(logits).cpu().numpy()
            masks_np = masks.cpu().numpy()
            for j in range(probs.shape[0]):
                prob = probs[j, 0]
                binary = (prob > threshold).astype("uint8")
                if morph_k > 0:
                    binary = _morph_close(binary, morph_k)
                Image.fromarray((binary * 255).astype("uint8")).save(
                    preds_dir / f"val_{bi:04d}_{j}.png")
                Image.fromarray((masks_np[j] * 255).astype("uint8")).save(
                    gt_dir / f"val_{bi:04d}_{j}.png")
    return preds_dir


# ============================================================
# CONVENIENCE
# ============================================================

def file_logger(path: Path) -> Callable[[str], None]:
    def _log(msg: str) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    return _log


def pick_device(gpu_id: Optional[int]):
    import torch
    if gpu_id is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{min(gpu_id, torch.cuda.device_count() - 1)}")
    return torch.device("cpu")
