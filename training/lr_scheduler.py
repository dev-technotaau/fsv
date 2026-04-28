"""training/lr_scheduler.py — Cosine schedule with warmup + layer-wise LR decay
parameter-group builder for AdamW.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int,
                      base_lr: float, warmup_lr: float, lr_min: float) -> float:
    """Returns the LR at `step`. Smooth cosine schedule with linear warmup."""
    if step < warmup_steps:
        # Linear warmup from warmup_lr to base_lr
        if warmup_steps == 0:
            return base_lr
        return warmup_lr + (base_lr - warmup_lr) * (step / warmup_steps)
    # Cosine decay from base_lr to lr_min over the remaining steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + (base_lr - lr_min) * cos


class CosineWarmupScheduler:
    """Custom scheduler that supports per-group base LRs (used for backbone vs head)."""
    def __init__(self, optimizer: torch.optim.Optimizer,
                 total_steps: int, warmup_steps: int, lr_min: float,
                 warmup_lr: float = 1e-7) -> None:
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.lr_min = lr_min
        self.warmup_lr = warmup_lr
        # Capture each group's base LR from the optimizer's initial state
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self._step_count = 0

    def step(self) -> None:
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = cosine_warmup_lr(
                self._step_count, self.total_steps, self.warmup_steps,
                base_lr=base, warmup_lr=self.warmup_lr, lr_min=self.lr_min,
            )
        self._step_count += 1

    def state_dict(self) -> dict:
        return {"step_count": self._step_count, "base_lrs": self.base_lrs}

    def load_state_dict(self, sd: dict) -> None:
        self._step_count = sd["step_count"]
        self.base_lrs = sd["base_lrs"]


# ══════════════════════════════════════════════════════════════════════
# Parameter group builder with layer-wise LR decay
# ══════════════════════════════════════════════════════════════════════

# Both DINOv2 and DINOv3 nest transformer blocks under encoder, but use
# different attribute names ("layer" vs "layers"). Likewise the parameter-name
# prefix differs: 'backbone.model.encoder.layer.<i>.' or '...layers.<i>.'.
_LAYER_PREFIXES = (
    "backbone.model.encoder.layer.",
    "backbone.model.encoder.layers.",
)


def _detect_backbone_layers(model: nn.Module) -> int:
    """Return the number of transformer blocks in the backbone (0 if not found)."""
    backbone = getattr(model, "backbone", None)
    if backbone is None or not hasattr(backbone, "model"):
        return 0
    enc = getattr(backbone.model, "encoder", None)
    if enc is None:
        return 0
    layers = getattr(enc, "layer", None) or getattr(enc, "layers", None)
    return len(layers) if layers is not None else 0


def _backbone_layer_idx(name: str) -> Optional[int]:
    for pref in _LAYER_PREFIXES:
        if name.startswith(pref):
            try:
                return int(name[len(pref):].split(".", 1)[0])
            except (IndexError, ValueError):
                return None
    return None


def build_param_groups(model: nn.Module,
                        head_lr: float,
                        backbone_lr: float,
                        backbone_lr_decay: float = 1.0,
                        weight_decay: float = 0.05) -> list[dict]:
    """Build optimizer parameter groups with separate LRs for backbone vs head,
    plus optional layer-wise LR decay applied to backbone transformer layers.

    Layer-wise LR decay: each transformer layer i gets backbone_lr * decay^(N-i)
    where N is the total number of layers. Makes early layers (closer to input)
    have smaller LR — preserves the pretrained backbone features.

    Also: zero weight decay for biases, LayerNorm/GroupNorm params, and
    positional embeddings (standard practice).

    Works for DINOv2 ('encoder.layer.<i>') and DINOv3 ('encoder.layers.<i>').
    """
    n_layers = _detect_backbone_layers(model)

    no_decay_keywords = ("bias", "LayerNorm", "layer_norm", "layernorm",
                         "GroupNorm", "norm.", "position", "pos_embed",
                         "cls_token", "register_token")

    def _no_decay(name: str) -> bool:
        return any(k in name for k in no_decay_keywords)

    groups_dict: dict[tuple[float, float], list[nn.Parameter]] = {}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = name.startswith("backbone.")
        wd = 0.0 if _no_decay(name) else weight_decay
        if is_backbone:
            layer_idx = _backbone_layer_idx(name)
            if layer_idx is not None and n_layers > 0:
                # layer 0 = closest to input -> highest decay
                lr = backbone_lr * (backbone_lr_decay ** (n_layers - 1 - layer_idx))
            else:
                # Patch embedding / non-layer backbone params -> deepest decay
                lr = backbone_lr * (backbone_lr_decay ** n_layers) if n_layers > 0 else backbone_lr
        else:
            lr = head_lr
        bucket = (round(lr, 8), wd)
        groups_dict.setdefault(bucket, []).append(p)

    groups = []
    for (lr, wd), params in sorted(groups_dict.items(), key=lambda kv: -kv[0][0]):
        groups.append({"params": params, "lr": lr, "weight_decay": wd})
    return groups
