"""training/metrics.py — Validation metrics for binary segmentation.

Provides:
  - per-batch metric accumulation (running confusion matrix)
  - final aggregation: IoU (Jaccard), Dice (F1), Pixel Accuracy, Precision, Recall
  - Boundary IoU (IoU computed only within K pixels of GT boundary)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass
class SegMetricsAccumulator:
    """Accumulates pixel counts across batches; computes IoU/Dice/Acc at end."""
    threshold: float = 0.5
    boundary_kernel: int = 5

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    boundary_tp: int = 0
    boundary_fp: int = 0
    boundary_fn: int = 0

    # Per-image accumulators (for class-balanced metrics)
    per_image_iou: list[float] = field(default_factory=list)
    per_image_dice: list[float] = field(default_factory=list)

    # Per-subcategory accumulators — populated when update() receives a list
    # of subcategory strings alongside probs/targets. Used to surface which
    # fence styles are weakest (cedar vs general wood vs negative subclasses)
    # so we can target augmentation / sampling fixes.
    per_subcat_iou: dict[str, list[float]] = field(default_factory=dict)
    per_subcat_dice: dict[str, list[float]] = field(default_factory=dict)

    @torch.no_grad()
    def update(self, probs: torch.Tensor, targets: torch.Tensor,
                subcategories: Optional[Sequence[Optional[str]]] = None) -> None:
        """probs: (B, 1, H, W) sigmoid-space  OR  (B, H, W). targets: (B, H, W) int.
        subcategories: optional per-sample subcategory tag (len == B). When
            provided, also tracks IoU/Dice grouped by this tag."""
        if probs.dim() == 4 and probs.shape[1] == 1:
            probs = probs.squeeze(1)
        preds = (probs >= self.threshold).long()
        targets = targets.long()

        # Pixel-level confusion matrix
        tp = ((preds == 1) & (targets == 1)).sum().item()
        fp = ((preds == 1) & (targets == 0)).sum().item()
        fn = ((preds == 0) & (targets == 1)).sum().item()
        tn = ((preds == 0) & (targets == 0)).sum().item()
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn

        # Per-image IoU + Dice (better statistic than dataset-pooled IoU)
        for i in range(preds.shape[0]):
            p = preds[i]
            t = targets[i]
            i_tp = ((p == 1) & (t == 1)).sum().item()
            i_fp = ((p == 1) & (t == 0)).sum().item()
            i_fn = ((p == 0) & (t == 1)).sum().item()
            denom_iou = i_tp + i_fp + i_fn
            denom_dice = 2 * i_tp + i_fp + i_fn
            iou_i = (i_tp / denom_iou) if denom_iou > 0 else 1.0
            dice_i = (2 * i_tp / denom_dice) if denom_dice > 0 else 1.0
            self.per_image_iou.append(iou_i)
            self.per_image_dice.append(dice_i)
            # Bucket by subcategory if caller supplied one — same image-level
            # numerators, just collected per group for breakdown reporting.
            if subcategories is not None and i < len(subcategories):
                sc = subcategories[i] or "unknown"
                self.per_subcat_iou.setdefault(sc, []).append(iou_i)
                self.per_subcat_dice.setdefault(sc, []).append(dice_i)

        # Boundary IoU: extract boundary band from targets, compute IoU in band
        target_f = targets.float().unsqueeze(1)
        boundary = self._boundary_band(target_f, self.boundary_kernel)   # (B, 1, H, W) {0,1}
        boundary = boundary.squeeze(1).long()
        b_pred = preds * boundary
        b_targ = targets * boundary
        self.boundary_tp += ((b_pred == 1) & (b_targ == 1)).sum().item()
        self.boundary_fp += ((b_pred == 1) & (b_targ == 0) & (boundary == 1)).sum().item()
        self.boundary_fn += ((b_pred == 0) & (b_targ == 1) & (boundary == 1)).sum().item()

    @staticmethod
    def _boundary_band(mask: torch.Tensor, kernel: int) -> torch.Tensor:
        """mask: (B, 1, H, W) float. Returns (B, 1, H, W) ∈ {0,1}: 1 within `kernel`
        pixels of an edge in mask."""
        # Edge = | mask - dilated mask | OR | mask - eroded mask |
        dilated = F.max_pool2d(mask, kernel, stride=1, padding=kernel // 2)
        eroded = -F.max_pool2d(-mask, kernel, stride=1, padding=kernel // 2)
        edge = (dilated - eroded).clamp(0, 1)
        return (edge > 0).float()

    def reset(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0
        self.boundary_tp = self.boundary_fp = self.boundary_fn = 0
        self.per_image_iou.clear()
        self.per_image_dice.clear()
        self.per_subcat_iou.clear()
        self.per_subcat_dice.clear()

    def compute(self) -> dict[str, float]:
        eps = 1e-9
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        dice = 2 * self.tp / (2 * self.tp + self.fp + self.fn + eps)
        acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        biou = self.boundary_tp / (self.boundary_tp + self.boundary_fp + self.boundary_fn + eps)
        per_img_iou_mean = (sum(self.per_image_iou) / len(self.per_image_iou)) \
            if self.per_image_iou else 0.0
        per_img_dice_mean = (sum(self.per_image_dice) / len(self.per_image_dice)) \
            if self.per_image_dice else 0.0
        out = {
            "val_iou":               iou,                     # dataset-pooled IoU
            "val_dice":              dice,
            "val_pixel_acc":         acc,
            "val_precision":         precision,
            "val_recall":            recall,
            "val_f1":                f1,
            "val_boundary_iou":      biou,
            "val_per_image_iou":     per_img_iou_mean,        # mean per-image IoU
            "val_per_image_dice":    per_img_dice_mean,
        }
        # Per-subcategory breakdown (only if any subcategories were provided
        # to update()). Keys are val_iou_<subcat>, val_dice_<subcat>,
        # val_n_<subcat> for the sample count in that bucket.
        for sc, ious in self.per_subcat_iou.items():
            n = len(ious)
            if n == 0:
                continue
            out[f"val_iou_{sc}"] = sum(ious) / n
            dices = self.per_subcat_dice.get(sc, [])
            if dices:
                out[f"val_dice_{sc}"] = sum(dices) / len(dices)
            out[f"val_n_{sc}"] = float(n)
        return out
