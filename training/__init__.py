"""Production training pipeline (v2) for wood-fence segmentation.

DINOv2/v3 backbone + Mask2Former-style decoder + multi-loss + two-phase training.
Kept SEPARATE from any prior training code so we never share state by accident.

Entry point: `python -m training.train --config configs/phase1.yaml`

Modules:
  config         — TrainingConfig dataclass + YAML loader/saver
  model          — Backbone + decoder + optional refinement head
  losses         — Combined BCE+Dice+Boundary loss
  post_process   — CRF / guided filter / morphology (inference-time)
  ema            — EMA model wrapper
  metrics        — IoU, Dice, F1, Boundary IoU
  lr_scheduler   — Cosine with warmup + layer-wise decay groups
  checkpoint     — CheckpointManager (save/load latest/best/ema)
  train          — Main training loop (handles one phase per invocation)
"""
from __future__ import annotations

__version__ = "2.0.0"
