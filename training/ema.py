"""training/ema.py — Exponential Moving Average wrapper for the model.

EMA stabilizes training and yields a more accurate val/test model. Standard
practice for modern segmentation/detection training (DETR, Mask2Former, YOLO).

Usage:
    ema = ModelEMA(model, decay=0.9999, warmup_steps=1000)
    # ... in training loop ...
    optimizer.step()
    ema.update(model, step=global_step)
    # ... at val time ...
    ema.apply_shadow(model)         # swap in EMA weights for inference
    ... validate ...
    ema.restore(model)              # swap original weights back
"""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999,
                 warmup_steps: int = 1000) -> None:
        self.decay = decay
        self.warmup_steps = warmup_steps
        # EMA parameters live on CPU by default to save VRAM.
        # We move to GPU in apply_shadow / restore.
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()
        # Also track buffers (BatchNorm stats etc.)
        for name, buf in model.named_buffers():
            self.shadow[name] = buf.detach().clone()

    def _effective_decay(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup from 0.5 to self.decay
            warm_decay = 0.5 + (self.decay - 0.5) * (step / max(1, self.warmup_steps))
            return min(self.decay, warm_decay)
        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        d = self._effective_decay(step)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            ema_p = self.shadow[name]
            # Move EMA tensor to param's device on first update if needed
            if ema_p.device != param.device:
                ema_p = ema_p.to(param.device)
                self.shadow[name] = ema_p
            ema_p.mul_(d).add_(param.detach(), alpha=1.0 - d)
        # Buffers: just copy (BN stats etc. — no smoothing)
        for name, buf in model.named_buffers():
            self.shadow[name] = buf.detach().clone()

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> None:
        """Swap EMA weights into the model. Call restore() to get originals back."""
        self.backup.clear()
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.device))
        for name, buf in model.named_buffers():
            if name in self.shadow:
                self.backup[name] = buf.data.clone()
                buf.data.copy_(self.shadow[name].to(buf.device))

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        for name, buf in model.named_buffers():
            if name in self.backup:
                buf.data.copy_(self.backup[name])
        self.backup.clear()

    @torch.no_grad()
    def shadow_state_dict(self) -> dict[str, torch.Tensor]:
        """Return the shadow as a flat tensor dict on CPU — directly loadable
        into the model via `model.load_state_dict()`. Use this to ship
        EMA-best inference weights independently of training state."""
        return {k: v.detach().cpu().clone() for k, v in self.shadow.items()}

    def state_dict(self) -> dict[str, torch.Tensor]:
        # Always store shadows on CPU to keep checkpoints portable
        return {
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()},
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.shadow = {k: v.clone() for k, v in sd["shadow"].items()}
        self.decay = sd.get("decay", self.decay)
        self.warmup_steps = sd.get("warmup_steps", self.warmup_steps)
