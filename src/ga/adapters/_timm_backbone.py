"""Helpers for loading backbones from timm + ViT-specific utilities.

Public API:
  load_timm_features(model_name, pretrained)  -> (model, channels, extract_fn)
  layer_decay_param_groups(model, base_lr, decay)  -> list[param_groups]
  vit_feature_pyramid(tokens, n_patches, taps)  -> list of feature maps at decreasing res
  count_parameters(model)  -> int
"""
from __future__ import annotations

from typing import Any, Callable


def load_timm_features(model_name: str, pretrained: bool = True):
    """Load timm backbone with features_only=True.
    Returns (model, channels_list, extract_fn(backbone, imgs) -> list_of_feature_maps).
    """
    import timm
    model = timm.create_model(model_name, pretrained=pretrained, features_only=True)
    try:
        channels = list(model.feature_info.channels())
    except Exception:
        channels = [info["num_chs"] for info in model.feature_info]  # type: ignore

    def extract(bb, imgs):
        return bb(imgs)

    return model, channels, extract


def layer_decay_param_groups(model, base_lr: float, decay: float = 0.85,
                              weight_decay: float = 0.01) -> list[dict[str, Any]]:
    """Produce param groups with layer-wise LR decay (deepest layer gets base_lr,
    shallower layers get progressively lower LRs).

    Works for ViT-style models by finding numeric suffixes in parameter names
    (`blocks.N.` or `layers.N.`). Falls back to a single group otherwise.
    """
    import re
    layers: dict[int, list[Any]] = {}
    other: list[Any] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = re.search(r"\b(?:blocks|layers|encoder\.layers)\.(\d+)\.", name)
        if m:
            layers.setdefault(int(m.group(1)), []).append(p)
        else:
            other.append(p)
    if not layers:
        return [{"params": list(model.parameters()), "lr": base_lr, "weight_decay": weight_decay}]

    max_idx = max(layers.keys())
    groups: list[dict[str, Any]] = []
    for idx, ps in sorted(layers.items()):
        scale = decay ** (max_idx - idx)
        groups.append({"params": ps, "lr": base_lr * scale, "weight_decay": weight_decay})
    if other:
        groups.append({"params": other, "lr": base_lr, "weight_decay": weight_decay})
    return groups


def vit_feature_pyramid(tokens, n_patches: int, taps: int = 4, backbone_dim: int = 1024):
    """Given ViT patch tokens [B, N, C], build a feature pyramid of `taps` levels
    via progressive average pooling. Returns list shallow→deep.
    """
    import torch
    import torch.nn.functional as F
    if tokens.dim() == 3:
        B, N, C = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, C, n_patches, n_patches)
    else:
        x = tokens
    out = []
    for i in range(taps):
        stride = 2 ** i
        out.append(x if stride == 1 else F.avg_pool2d(x, stride, stride))
    return out


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
