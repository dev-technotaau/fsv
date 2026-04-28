"""Abstract ModelAdapter base class.

Each combo has one Adapter subclass. The adapter:
  1. Builds / loads the model from genome.params
  2. Runs a short proxy training (respecting fitness_cfg.proxy_budget)
  3. Evaluates on val set + hard_eval_dir (if present)
  4. Returns AdapterResult(metrics, status, ...)

Heavy lifting (actual training loops) typically delegates to existing training
scripts via subprocess OR uses a minimal trainer inlined per-adapter.

The adapter MUST be safe to run in a subprocess (no global state beyond what
it owns). It receives a work_dir where it can write checkpoints/logs/masks.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..config import DataConfig, FitnessConfig
from ..genome import Genome


@dataclass
class AdapterResult:
    metrics: dict[str, float]          # MUST contain "iou"; "boundary_f1", "tv_penalty" optional
    status: str = "ok"                 # "ok" | "killed_early" | "crashed"
    error: Optional[str] = None
    ckpt_path: Optional[Path] = None   # where the best checkpoint was saved
    extra: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(abc.ABC):
    """Base class every combo adapter subclasses."""

    def __init__(
        self,
        *,
        genome: Genome,
        data_cfg: DataConfig,
        fitness_cfg: FitnessConfig,
        work_dir: Path,
        gpu_id: Optional[int] = None,
    ):
        self.genome = genome
        self.data_cfg = data_cfg
        self.fitness_cfg = fitness_cfg
        self.work_dir = work_dir
        self.gpu_id = gpu_id
        self.work_dir.mkdir(parents=True, exist_ok=True)

    # ---------- helpers ----------
    @property
    def params(self) -> dict[str, Any]:
        return self.genome.params

    @property
    def combo_key(self) -> str:
        return self.genome.combo_key

    def device_str(self) -> str:
        if self.gpu_id is None:
            return "cpu"
        return f"cuda:{self.gpu_id}" if self.gpu_id >= 0 else "cpu"

    # ---------- to override ----------
    @abc.abstractmethod
    def run(self) -> AdapterResult:
        """Train (short proxy) + eval, return metrics."""

    # ---------- utility for subclasses ----------
    def compute_iou_and_bf1(
        self,
        preds_dir: Path,
        masks_dir: Optional[Path] = None,
    ) -> dict[str, float]:
        """Compute IoU, boundary F1, and TV penalty from saved prediction PNGs.

        preds_dir — directory of predicted binary PNG masks (same filenames as GT).
        masks_dir — ground-truth masks dir (defaults to self.data_cfg.masks_dir).
        """
        import numpy as np
        from PIL import Image

        gt_dir = masks_dir or self.data_cfg.masks_dir
        iou_list: list[float] = []
        bf1_list: list[float] = []
        tv_list:  list[float] = []

        for pred_path in sorted(preds_dir.glob("*.png")):
            gt_path = gt_dir / pred_path.name
            if not gt_path.exists():
                continue
            pred = np.array(Image.open(pred_path).convert("L")) > 127
            gt   = np.array(Image.open(gt_path).convert("L")) > 127
            inter = np.logical_and(pred, gt).sum()
            union = np.logical_or(pred, gt).sum()
            iou = float(inter / union) if union > 0 else 0.0
            iou_list.append(iou)
            bf1_list.append(_boundary_f1(pred, gt))
            tv_list.append(_total_variation(pred.astype(np.float32)))

        if not iou_list:
            return {"iou": 0.0, "boundary_f1": 0.0, "tv_penalty": 0.0, "n_eval": 0}
        return {
            "iou":         float(np.mean(iou_list)),
            "boundary_f1": float(np.mean(bf1_list)),
            "tv_penalty":  float(np.mean(tv_list)),
            "n_eval":      len(iou_list),
        }


# ---------- metric helpers ----------

def _boundary_f1(pred: "np.ndarray", gt: "np.ndarray", tol_px: int = 2) -> float:
    """Boundary F1 score: precision+recall of predicted edges within tol_px of GT edges."""
    import numpy as np
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
    except ImportError:
        # graceful fallback: just IoU on eroded masks
        return 0.0

    gt_edge   = gt   ^ binary_erosion(gt,   iterations=1)
    pred_edge = pred ^ binary_erosion(pred, iterations=1)
    if not pred_edge.any() or not gt_edge.any():
        return 0.0
    # distance from pred_edge to nearest gt_edge
    gt_dt   = distance_transform_edt(~gt_edge)
    pred_dt = distance_transform_edt(~pred_edge)
    prec = (gt_dt[pred_edge] <= tol_px).mean()
    rec  = (pred_dt[gt_edge] <= tol_px).mean()
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def _total_variation(mask: "np.ndarray") -> float:
    """Total variation — penalizes blocky/jagged boundaries. Normalized per-pixel."""
    import numpy as np
    dx = np.abs(np.diff(mask, axis=0)).sum()
    dy = np.abs(np.diff(mask, axis=1)).sum()
    return float((dx + dy) / max(1, mask.size))
