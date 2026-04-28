"""training/checkpoint.py — Save / load training state.

Manages: model weights, optimizer state, LR scheduler state, EMA state,
AMP grad scaler state, RNG states (for reproducibility), training meta
(epoch, global step, best metric).

Layout under <output_dir>/checkpoints/:
    latest.pt              — overwritten every epoch (resume here)
    best.pt                — overwritten when val metric improves
    ema.pt                 — separate file for EMA weights (smaller, val-only use)
    epoch_<NNN>.pt         — periodic snapshots (kept N most recent)
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


@dataclass
class CheckpointState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("-inf")
    best_metric_name: str = "val_iou"


class CheckpointManager:
    def __init__(self, ckpt_dir: str | Path,
                 keep_last_n: int = 3,
                 save_optimizer_state: bool = True) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.save_optimizer_state = save_optimizer_state

    # ── Save ─────────────────────────────────────────────────────────

    def save(self,
             path: str | Path,
             model: nn.Module,
             optimizer: Optional[torch.optim.Optimizer] = None,
             scheduler: Optional[Any] = None,
             scaler: Optional[Any] = None,
             ema: Optional[Any] = None,
             state: Optional[CheckpointState] = None,
             extra: Optional[dict] = None,
             config_dict: Optional[dict] = None,
             provenance: Optional[dict] = None) -> None:
        """Atomic checkpoint save.

        `config_dict`: serialized TrainingConfig (cfg.to_dict()). Bundling it
            here makes the checkpoint SELF-DESCRIBING — anyone loading it can
            rebuild the exact model architecture without needing the YAML.
        `provenance`: who/what/when (git SHA, library versions, hostname,
            created_at). For audit + debugging post-launch.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "model": _safe_state_dict(model),
            "rng": {
                "torch": torch.get_rng_state(),
                "torch_cuda": (torch.cuda.get_rng_state_all()
                                if torch.cuda.is_available() else None),
            },
        }
        if state is not None:
            payload["state"] = state.__dict__
        if optimizer is not None and self.save_optimizer_state:
            payload["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        if scaler is not None:
            payload["scaler"] = scaler.state_dict()
        if ema is not None:
            payload["ema"] = ema.state_dict()
        if config_dict is not None:
            payload["config"] = config_dict
        if provenance is not None:
            payload["provenance"] = provenance
        if extra is not None:
            payload["extra"] = extra
        # Atomic write
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        # On Windows the destination may be read-only; clear that.
        if path.exists():
            try:
                import os
                os.chmod(path, 0o644)
            except OSError:
                pass
        tmp.replace(path)

    def save_latest(self, **kwargs) -> Path:
        out = self.ckpt_dir / "latest.pt"
        self.save(out, **kwargs)
        return out

    def save_best(self, **kwargs) -> Path:
        out = self.ckpt_dir / "best.pt"
        self.save(out, **kwargs)
        return out

    def save_inference_only(self,
                              path: str | Path,
                              model: nn.Module,
                              meta: Optional[dict] = None,
                              config_dict: Optional[dict] = None,
                              provenance: Optional[dict] = None) -> Path:
        """Save a STRIPPED-DOWN checkpoint: model weights + meta + config.
        No optimizer / scheduler / EMA / RNG.

        With `config_dict` bundled, the checkpoint is fully SELF-DESCRIBING:
        loading it produces the exact same architecture that was trained,
        no separate YAML required.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "model": _safe_state_dict(model),
            "inference_only": True,
        }
        if meta is not None:
            payload["meta"] = meta
        if config_dict is not None:
            payload["config"] = config_dict
        if provenance is not None:
            payload["provenance"] = provenance
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        if path.exists():
            try:
                import os
                os.chmod(path, 0o644)
            except OSError:
                pass
        tmp.replace(path)
        return path

    def save_best_with_ema_swap(self,
                                  model: nn.Module,
                                  ema: Optional[Any] = None,
                                  **kwargs) -> Path:
        """Save best.pt where the 'model' state IS the EMA shadow (if EMA is
        provided). Validation runs use EMA weights; this guarantees that
        the checkpoint we mark as "best" is actually the model that achieved
        the best metric — not the raw post-step weights, which can differ."""
        out = self.ckpt_dir / "best.pt"
        if ema is None:
            self.save(out, model=model, ema=ema, **kwargs)
            return out
        ema.apply_shadow(model)
        try:
            self.save(out, model=model, ema=ema, **kwargs)
        finally:
            ema.restore(model)
        return out

    def save_ema(self, ema, model: nn.Module, state: CheckpointState,
                 config_dict: Optional[dict] = None,
                 provenance: Optional[dict] = None) -> Path:
        out = self.ckpt_dir / "ema.pt"
        # Build a "model" state dict from EMA shadow params, so it loads
        # like a normal model checkpoint.
        ema.apply_shadow(model)
        try:
            payload: dict[str, Any] = {
                "model": _safe_state_dict(model),
                "state": state.__dict__,
                "is_ema": True,
            }
            if config_dict is not None:
                payload["config"] = config_dict
            if provenance is not None:
                payload["provenance"] = provenance
            torch.save(payload, out.with_suffix(".tmp"))
            out.with_suffix(".tmp").replace(out)
        finally:
            ema.restore(model)
        return out

    def save_periodic(self, epoch: int, **kwargs) -> Path:
        out = self.ckpt_dir / f"epoch_{epoch:03d}.pt"
        self.save(out, **kwargs)
        # Prune old periodic checkpoints
        periodics = sorted(self.ckpt_dir.glob("epoch_*.pt"))
        if self.keep_last_n > 0 and len(periodics) > self.keep_last_n:
            for old in periodics[:-self.keep_last_n]:
                try:
                    old.unlink()
                except OSError:
                    pass
        return out

    # ── Load ─────────────────────────────────────────────────────────

    @staticmethod
    def load(path: str | Path,
             model: Optional[nn.Module] = None,
             optimizer: Optional[torch.optim.Optimizer] = None,
             scheduler: Optional[Any] = None,
             scaler: Optional[Any] = None,
             ema: Optional[Any] = None,
             strict: bool = True,
             map_location: str = "cpu") -> dict:
        """Load checkpoint and restore the provided components in-place.
        Returns the raw payload dict so caller can extract `state` etc."""
        path = Path(path)
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if model is not None and "model" in payload:
            missing, unexpected = model.load_state_dict(payload["model"], strict=strict)
            if not strict:
                if missing:
                    print(f"[ckpt] missing keys: {len(missing)} (first 3: {missing[:3]})")
                if unexpected:
                    print(f"[ckpt] unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
        if optimizer is not None and "optimizer" in payload:
            try:
                optimizer.load_state_dict(payload["optimizer"])
            except (ValueError, KeyError) as e:
                print(f"[ckpt] WARN: could not restore optimizer state: {e}")
        if scheduler is not None and "scheduler" in payload:
            try:
                scheduler.load_state_dict(payload["scheduler"])
            except Exception as e:
                print(f"[ckpt] WARN: could not restore scheduler state: {e}")
        if scaler is not None and "scaler" in payload:
            try:
                scaler.load_state_dict(payload["scaler"])
            except Exception as e:
                print(f"[ckpt] WARN: could not restore scaler state: {e}")
        if ema is not None and "ema" in payload:
            try:
                ema.load_state_dict(payload["ema"])
            except Exception as e:
                print(f"[ckpt] WARN: could not restore EMA state: {e}")
        if "rng" in payload:
            try:
                torch.set_rng_state(payload["rng"]["torch"])
                if (payload["rng"].get("torch_cuda") is not None
                        and torch.cuda.is_available()):
                    torch.cuda.set_rng_state_all(payload["rng"]["torch_cuda"])
            except Exception as e:
                print(f"[ckpt] WARN: could not restore RNG state: {e}")
        return payload


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _safe_state_dict(model: nn.Module) -> dict:
    """Return state_dict, stripping any 'module.' prefix from DataParallel/DDP."""
    sd = model.state_dict()
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in sd.items()}
