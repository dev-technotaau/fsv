"""training/losses.py — Combined segmentation loss with multiple components.

Components implemented:
  - BCEWithLogitsLoss          pixel-wise classification (with optional pos_weight)
  - DiceLoss                   region-overlap (1 - 2*intersection / union)
  - BoundaryLoss               weights pixels near GT boundary higher (signed-distance based)
  - LovaszSoftmaxLoss          IoU-surrogate (Berman et al., 2018)
  - ConnectivityLoss           experimental: encourages topologically connected masks
                                (a simple Euler-characteristic-style penalty)

The CombinedLoss sums weighted components. Set a weight to 0 to disable.

References:
  - Boundary loss: Kervadec et al., "Boundary loss for highly unbalanced segmentation"
                    (Medical Image Analysis 2021)
  - Lovasz: Berman et al., "The Lovasz-Softmax Loss" (CVPR 2018)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# BCE
# ══════════════════════════════════════════════════════════════════════

class BCELoss(nn.Module):
    """Binary cross-entropy with three optional fence-friendly modes:

      - `pos_weight`:        scale up the positive (fence) class loss.
      - `focal_gamma`:       focal-loss exponent — down-weights easy pixels
                              by (1 - p_t)^gamma. gamma=2 is the standard
                              choice (Lin et al., 2017). Concentrates loss
                              on hard pixels — directly attacks the
                              "cedar fence vs other wood" confusion.
      - `ohem_top_k_ratio`:  if > 0, keep only the top-K% highest-loss
                              pixels per image. Online Hard Example Mining.
                              Stacks with focal_gamma if both are set.
                              Independent of PointRend (PointRend selects
                              UNCERTAIN pixels by logit; OHEM selects
                              HIGH-LOSS pixels by current loss).
    """

    def __init__(self, pos_weight: Optional[float] = None,
                 focal_gamma: float = 0.0,
                 ohem_top_k_ratio: float = 0.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.focal_gamma = float(focal_gamma)
        self.ohem_top_k_ratio = float(ohem_top_k_ratio)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """logits: (B, 1, H, W) or (B, N) for PointRend.
        targets: (B, H, W) long or (B, N) float.
        sample_weight: (B,) optional per-sample weight."""
        targets_f = targets.float()
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits_flat = logits.squeeze(1)
        elif logits.dim() == 4:
            logits_flat = logits[:, 0]
        else:
            logits_flat = logits  # (B, N) — PointRend path
        pw = None
        if self.pos_weight is not None:
            pw = torch.tensor([self.pos_weight], device=logits.device,
                              dtype=logits.dtype)
        loss = F.binary_cross_entropy_with_logits(
            logits_flat, targets_f, pos_weight=pw, reduction="none",
        )   # (B, H, W) or (B, N)

        # Focal weighting: emphasize hard pixels by (1 - p_t)^gamma
        if self.focal_gamma > 0:
            with torch.no_grad():
                probs = torch.sigmoid(logits_flat)
                p_t = probs * targets_f + (1 - probs) * (1 - targets_f)
                focal_w = (1.0 - p_t).pow(self.focal_gamma)
            loss = loss * focal_w

        if sample_weight is not None:
            # Broadcast sample_weight across spatial / point dims
            shape = (-1,) + (1,) * (loss.dim() - 1)
            loss = loss * sample_weight.view(*shape)

        # OHEM: per-image, keep only the top-K% highest losses
        if self.ohem_top_k_ratio > 0:
            B = loss.shape[0]
            flat = loss.reshape(B, -1)
            n = flat.shape[1]
            k = max(1, int(n * self.ohem_top_k_ratio))
            top_losses, _ = flat.topk(k, dim=1, sorted=False)
            return top_losses.mean()

        return loss.mean()


# ══════════════════════════════════════════════════════════════════════
# Dice
# ══════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        targets = targets.float()
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits.squeeze(1))
        else:
            probs = torch.sigmoid(logits[:, 0])
        # Per-sample dice: (B,)
        intersection = (probs * targets).flatten(1).sum(dim=1)
        union = probs.flatten(1).sum(dim=1) + targets.flatten(1).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice                                    # (B,)
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()


class TverskyLoss(nn.Module):
    """Generalized Dice with separate FN and FP penalties:

        Tversky = TP / (TP + alpha*FN + beta*FP)

    - alpha = beta = 0.5  →  equivalent to Dice
    - alpha > beta        →  penalize FN more  (recall-focused)
                              For fence-staining we WANT high recall:
                              missing fence pixels means staining gaps
                              show through; over-coloring slightly outside
                              is much less visible. Default alpha=0.7, beta=0.3.

    Combine with Dice (not replace) — Dice optimizes balanced segmentation;
    Tversky tilts the model toward recall.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3,
                 smooth: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        targets = targets.float()
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits.squeeze(1))
        else:
            probs = torch.sigmoid(logits[:, 0])
        tp = (probs * targets).flatten(1).sum(dim=1)
        fn = ((1 - probs) * targets).flatten(1).sum(dim=1)
        fp = (probs * (1 - targets)).flatten(1).sum(dim=1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp
                                          + self.smooth)
        loss = 1.0 - tversky
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()


# ══════════════════════════════════════════════════════════════════════
# Boundary loss (edge-weighted BCE proxy)
# ══════════════════════════════════════════════════════════════════════

class BoundaryLoss(nn.Module):
    """Edge-weighted BCE: pixels within `kernel_size` of a GT boundary are
    weighted higher in the loss. This is a fast, well-behaved proxy for the
    full signed-distance boundary loss; no heavy SDF computation needed."""

    def __init__(self, kernel_size: int = 5, edge_weight: float = 5.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.edge_weight = edge_weight
        # Sobel-style edge extractor
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = kx.t()
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def _boundary_map(self, mask: torch.Tensor) -> torch.Tensor:
        """mask: (B, 1, H, W) float. Returns boundary heatmap (B, 1, H, W)."""
        gx = F.conv2d(mask, self.kx, padding=1)
        gy = F.conv2d(mask, self.ky, padding=1)
        edge = (gx.abs() + gy.abs()).clamp(0, 1)
        # Dilate edge by kernel_size to make a fat boundary band
        if self.kernel_size > 1:
            edge = F.max_pool2d(edge, kernel_size=self.kernel_size,
                                 stride=1, padding=self.kernel_size // 2)
        return edge

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        targets_f = targets.float().unsqueeze(1)             # (B, 1, H, W)
        edge = self._boundary_map(targets_f)                 # (B, 1, H, W)
        # Per-pixel weight: 1.0 baseline + edge_weight * edge
        weight = 1.0 + self.edge_weight * edge
        if logits.shape[1] == 1:
            logits_flat = logits
        else:
            logits_flat = logits[:, 0:1]
        loss = F.binary_cross_entropy_with_logits(
            logits_flat, targets_f, weight=weight, reduction="none",
        )
        loss = loss.flatten(2).mean(dim=2).squeeze(1)        # (B,)
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()


# ══════════════════════════════════════════════════════════════════════
# Lovasz-Softmax (IoU surrogate)
# ══════════════════════════════════════════════════════════════════════

class LovaszLoss(nn.Module):
    """Binary Lovasz hinge — IoU-surrogate that's differentiable.
    Reference: Berman et al., 'The Lovasz-Softmax Loss', CVPR 2018."""

    @staticmethod
    def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jaccard = 1.0 - intersection / union
        if len(gt_sorted) > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if logits.shape[1] == 1:
            logits_flat = logits.squeeze(1)
        else:
            logits_flat = logits[:, 0]
        targets_f = targets.float()
        signs = 2.0 * targets_f - 1.0
        errors = (1.0 - logits_flat * signs)                 # (B, H, W)
        # Per-image Lovasz
        losses = []
        for i in range(errors.shape[0]):
            err = errors[i].flatten()
            tgt = targets_f[i].flatten()
            sorted_idx = torch.argsort(err, descending=True)
            err_sorted = err[sorted_idx]
            gt_sorted = tgt[sorted_idx]
            grad = self._lovasz_grad(gt_sorted)
            li = torch.dot(F.relu(err_sorted), grad)
            if sample_weight is not None:
                li = li * sample_weight[i]
            losses.append(li)
        return torch.stack(losses).mean()


# ══════════════════════════════════════════════════════════════════════
# Connectivity (experimental — soft Euler-number penalty)
# ══════════════════════════════════════════════════════════════════════

class ConnectivityLoss(nn.Module):
    """Experimental: penalize the number of disconnected regions in the
    predicted mask via a soft, differentiable Euler-number proxy.

    Real topological losses require persistent homology (e.g., topologylayer);
    this is a much faster heuristic that pushes toward fewer holes / pieces.

    NOT a drop-in replacement for true topology loss — use sparingly with
    a small weight (1e-3 to 1e-2 range), and watch your val IoU."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits.squeeze(1))
        else:
            probs = torch.sigmoid(logits[:, 0])
        # Soft Euler approximation: count of approximate "vertices" minus
        # "edges" minus "faces" via small kernel convolutions on the soft mask.
        # Reduce to a scalar per image, then take mean.
        # 2x2 max-pool reductions act as a "vertex" detector for 1-pixel features.
        v = F.max_pool2d(probs.unsqueeze(1), kernel_size=2, stride=2)
        e = F.avg_pool2d(probs.unsqueeze(1), kernel_size=2, stride=2)
        loss = (v - e).abs().flatten(1).mean(dim=1)         # (B,)
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()


# ══════════════════════════════════════════════════════════════════════
# Edge-aware auxiliary loss (refinement head's edge prediction head)
# ══════════════════════════════════════════════════════════════════════

class EdgeLoss(nn.Module):
    """BCE between predicted edge mask and Sobel-derived GT edges.

    Forces the refinement head's intermediate features to be edge-aware,
    which in turn sharpens the residual mask predictions on boundaries.

    GT edges are computed on-the-fly from the binary GT mask via Sobel
    (no extra labels needed). The edge band is widened by `dilate` pixels
    via max_pool so the loss isn't a hairline (otherwise gradients vanish
    when the prediction is off by 1-2 pixels).
    """

    def __init__(self, dilate: int = 3, pos_weight: float = 8.0) -> None:
        super().__init__()
        self.dilate = max(1, int(dilate))
        # Edges are very sparse (~1-3% of pixels); reweight positives heavily.
        self.pos_weight = float(pos_weight)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = kx.t()
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    @torch.no_grad()
    def gt_edges(self, targets: torch.Tensor) -> torch.Tensor:
        """targets: (B, H, W) {0, 1}. Returns (B, 1, H, W) {0, 1} edge band."""
        m = targets.float().unsqueeze(1)
        gx = F.conv2d(m, self.kx, padding=1)
        gy = F.conv2d(m, self.ky, padding=1)
        edge = (gx.abs() + gy.abs()).clamp(0, 1)
        if self.dilate > 1:
            edge = F.max_pool2d(edge, kernel_size=self.dilate,
                                 stride=1, padding=self.dilate // 2)
        return (edge > 0).float()

    def forward(self, edge_logits: torch.Tensor, targets: torch.Tensor,
                sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """edge_logits: (B, 1, H, W) raw logits. targets: (B, H, W)."""
        edges = self.gt_edges(targets)                                # (B, 1, H, W)
        pw = torch.tensor([self.pos_weight], device=edge_logits.device,
                          dtype=edge_logits.dtype)
        loss = F.binary_cross_entropy_with_logits(
            edge_logits, edges, pos_weight=pw, reduction="none",
        )                                                              # (B, 1, H, W)
        if sample_weight is not None:
            loss = loss * sample_weight[:, None, None, None]
        return loss.mean()


# ══════════════════════════════════════════════════════════════════════
# PointRend-style importance sampling (Mask2Former + PointRend)
# ══════════════════════════════════════════════════════════════════════

class PointSampler:
    """Sample N informative points per image, with k-fold uncertainty
    oversampling on the boundaries — exactly the training-time scheme used by
    Mask2Former (lifted from Detectron2's PointRend implementation).

    Why this matters:
      - Dense per-pixel BCE+Dice on a 1024x1024 mask spends >99% of its compute
        on easy interior pixels. PointRend caps the loss at ~12K points/image
        (~10% of a 384^2 mask), spending those on ambiguous boundary pixels.
      - 20-30% faster training step + better boundary IoU.

    Usage:
        sampler = PointSampler(n_points=12544, oversample_ratio=3.0,
                                importance_ratio=0.75)
        coords = sampler.sample_uncertain_points(coarse_logits)  # (B, N, 2)
        pred_at_pts = sampler.sample(coarse_logits, coords)        # (B, N, 1)
        gt_at_pts   = sampler.sample(targets.float()[:, None], coords,
                                       mode="nearest")              # (B, N, 1)
    """

    def __init__(self, n_points: int = 12544,
                 oversample_ratio: float = 3.0,
                 importance_ratio: float = 0.75) -> None:
        if oversample_ratio < 1.0:
            raise ValueError("oversample_ratio must be >= 1.0")
        if not 0.0 <= importance_ratio <= 1.0:
            raise ValueError("importance_ratio must be in [0, 1]")
        self.n_points = int(n_points)
        self.oversample_ratio = float(oversample_ratio)
        self.importance_ratio = float(importance_ratio)

    @staticmethod
    @torch.no_grad()
    def sample(maps: torch.Tensor, coords: torch.Tensor,
                mode: str = "bilinear") -> torch.Tensor:
        """Sample (B, C, H, W) at coords (B, N, 2) in [0,1]^2.
        Returns (B, N, C). `mode`: 'bilinear' for logits, 'nearest' for masks."""
        # grid_sample expects coords in [-1, 1] and shape (B, H_out, W_out, 2)
        grid = (coords * 2 - 1).unsqueeze(2)                       # (B, N, 1, 2)
        sampled = F.grid_sample(maps, grid, mode=mode,
                                  padding_mode="border", align_corners=False)
        # sampled shape: (B, C, N, 1) -> (B, N, C)
        return sampled.squeeze(-1).transpose(1, 2)

    @staticmethod
    def sample_with_grad(maps: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Same as sample(bilinear) but DIFFERENTIABLE on `maps` (logits)."""
        grid = (coords * 2 - 1).unsqueeze(2)
        sampled = F.grid_sample(maps, grid, mode="bilinear",
                                  padding_mode="border", align_corners=False)
        return sampled.squeeze(-1).transpose(1, 2)

    @torch.no_grad()
    def sample_uncertain_points(self, logits: torch.Tensor) -> torch.Tensor:
        """Pick `n_points` points per image: k% by uncertainty (closeness of
        sigmoid(logit) to 0.5), the rest uniform. Returns (B, N, 2) in [0,1]."""
        B = logits.shape[0]
        n = self.n_points
        n_oversample = max(n, int(n * self.oversample_ratio))
        n_uncertain = int(n * self.importance_ratio)
        n_random = n - n_uncertain
        device = logits.device

        # Step 1: oversample uniformly
        cand = torch.rand(B, n_oversample, 2, device=device)        # (B, M, 2)
        # Step 2: get logits at those candidates
        sampled = self.sample(logits, cand, mode="bilinear")        # (B, M, 1)
        # Step 3: uncertainty = -|logit| (higher = more ambiguous)
        uncertainty = -sampled.squeeze(-1).abs()                    # (B, M)
        # Step 4: top-k uncertain
        if n_uncertain > 0:
            _, idx = torch.topk(uncertainty, k=n_uncertain, dim=1)
            idx_e = idx.unsqueeze(-1).expand(-1, -1, 2)
            kept = torch.gather(cand, 1, idx_e)                      # (B, n_uncertain, 2)
        else:
            kept = cand[:, :0]
        # Step 5: top up with uniform random
        if n_random > 0:
            rand_extra = torch.rand(B, n_random, 2, device=device)
            kept = torch.cat([kept, rand_extra], dim=1)              # (B, n, 2)
        return kept


def _bce_at_points(pred_logits: torch.Tensor, gt_at_pts: torch.Tensor,
                    sample_weight: Optional[torch.Tensor],
                    pos_weight: Optional[float]) -> torch.Tensor:
    """BCE on (B, N) logits/targets. pos_weight reweights positive class."""
    if pos_weight is not None:
        pw = torch.tensor([pos_weight], device=pred_logits.device,
                          dtype=pred_logits.dtype)
    else:
        pw = None
    loss = F.binary_cross_entropy_with_logits(
        pred_logits, gt_at_pts, pos_weight=pw, reduction="none",
    )                                                                # (B, N)
    if sample_weight is not None:
        loss = loss * sample_weight[:, None]
    return loss.mean()


def _dice_at_points(pred_logits: torch.Tensor, gt_at_pts: torch.Tensor,
                     sample_weight: Optional[torch.Tensor],
                     smooth: float) -> torch.Tensor:
    """Soft-Dice on (B, N) logits/targets. Per-image then mean."""
    probs = torch.sigmoid(pred_logits)
    inter = (probs * gt_at_pts).sum(dim=1)                          # (B,)
    union = probs.sum(dim=1) + gt_at_pts.sum(dim=1)                 # (B,)
    dice = (2 * inter + smooth) / (union + smooth)
    loss = 1.0 - dice                                                # (B,)
    if sample_weight is not None:
        loss = loss * sample_weight
    return loss.mean()


def pointrend_bce_dice(logits: torch.Tensor, targets: torch.Tensor,
                        sampler: PointSampler,
                        sample_weight: Optional[torch.Tensor] = None,
                        pos_weight: Optional[float] = None,
                        dice_smooth: float = 1.0,
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute BCE + Dice on N importance-sampled points.
    Returns (bce_loss, dice_loss), both scalars."""
    if logits.dim() == 4 and logits.shape[1] == 1:
        L = logits
    else:
        L = logits[:, 0:1]
    coords = sampler.sample_uncertain_points(L)                     # (B, N, 2)
    pred = sampler.sample_with_grad(L, coords).squeeze(-1)          # (B, N)
    tgt = targets.float().unsqueeze(1)                              # (B, 1, H, W)
    gt_at_pts = sampler.sample(tgt, coords, mode="nearest").squeeze(-1)  # (B, N)
    bce = _bce_at_points(pred, gt_at_pts, sample_weight, pos_weight)
    dice = _dice_at_points(pred, gt_at_pts, sample_weight, dice_smooth)
    return bce, dice


# ══════════════════════════════════════════════════════════════════════
# Combined loss
# ══════════════════════════════════════════════════════════════════════

class CombinedLoss(nn.Module):
    """Sum of weighted loss components. Returns a scalar plus a dict of components.

    When `loss_cfg.use_pointrend == True`, BCE+Dice are computed via the
    PointRend importance-sampling path (Mask2Former's actual training-time
    loss). Boundary / Lovasz / Connectivity losses still use dense pixels
    since they're already cheap or boundary-aware by construction.
    """

    def __init__(self, loss_cfg) -> None:
        super().__init__()
        self.cfg = loss_cfg
        self.bce = (BCELoss(
            pos_weight=loss_cfg.pos_weight,
            focal_gamma=float(getattr(loss_cfg, "focal_gamma", 0.0)),
            ohem_top_k_ratio=float(getattr(loss_cfg, "ohem_top_k_ratio", 0.0)),
        ) if loss_cfg.bce_weight > 0 else None)
        self.dice = DiceLoss(smooth=loss_cfg.dice_smooth) if loss_cfg.dice_weight > 0 else None
        self.tversky = (TverskyLoss(
            alpha=float(getattr(loss_cfg, "tversky_alpha", 0.7)),
            beta=float(getattr(loss_cfg, "tversky_beta", 0.3)),
            smooth=loss_cfg.dice_smooth,
        ) if float(getattr(loss_cfg, "tversky_weight", 0.0)) > 0 else None)
        self.boundary = BoundaryLoss(kernel_size=loss_cfg.boundary_kernel_size) \
            if loss_cfg.boundary_weight > 0 else None
        self.lovasz = LovaszLoss() if loss_cfg.lovasz_weight > 0 else None
        self.conn = ConnectivityLoss() if loss_cfg.connectivity_weight > 0 else None

        # PointRend importance sampler (lazily used iff enabled)
        self.use_pointrend = bool(getattr(loss_cfg, "use_pointrend", False))
        if self.use_pointrend:
            self.point_sampler = PointSampler(
                n_points=int(getattr(loss_cfg, "pointrend_n_points", 12544)),
                oversample_ratio=float(getattr(loss_cfg, "pointrend_oversample_ratio", 3.0)),
                importance_ratio=float(getattr(loss_cfg, "pointrend_importance_ratio", 0.75)),
            )
        else:
            self.point_sampler = None

        # Edge-aware auxiliary loss (refinement head's edge prediction)
        self.edge_loss_weight = float(getattr(loss_cfg, "edge_loss_weight", 0.0))
        if self.edge_loss_weight > 0:
            self.edge_loss = EdgeLoss(
                dilate=int(getattr(loss_cfg, "edge_loss_dilate", 3)),
                pos_weight=float(getattr(loss_cfg, "edge_loss_pos_weight", 8.0)),
            )
        else:
            self.edge_loss = None

    def set_pos_weight(self, pos_weight: float) -> None:
        if self.bce is not None:
            self.bce.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                sample_weight: Optional[torch.Tensor] = None,
                refined_logits: Optional[torch.Tensor] = None,
                aux_logits: Optional[list[torch.Tensor]] = None,
                edge_logits: Optional[torch.Tensor] = None,
                refined_iter_logits: Optional[list[torch.Tensor]] = None,
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns (total_loss, components).

        `aux_logits`: list of per-decoder-layer mask logits for deep
            supervision (Mask2Former-style). The FINAL element should equal
            `logits`; the loss applies to ALL of them with weight
            `cfg.deep_supervision_weight` per aux entry. Skip via empty list
            or None.

        components values are DETACHED tensors (no `.item()` here) so the
        caller can batch GPU->CPU syncs once per logging interval.
        """
        # Accumulate the total in fp32 even when logits are bf16/fp16 (under
        # autocast). bf16's 7-bit mantissa loses precision on small loss values.
        components: dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=logits.device, dtype=torch.float32)

        # Helper to compute BCE+Dice on `lg` — uses PointRend sampling when
        # enabled (faster + boundary-aware), else dense per-pixel BCE/Dice.
        def _bce_dice(lg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if (self.use_pointrend and self.point_sampler is not None
                    and self.training):
                pos_w = self.bce.pos_weight if self.bce is not None else None
                return pointrend_bce_dice(
                    lg, targets, self.point_sampler,
                    sample_weight=sample_weight, pos_weight=pos_w,
                    dice_smooth=self.cfg.dice_smooth,
                )
            bce_v = (self.bce(lg, targets, sample_weight=sample_weight)
                     if self.bce is not None else None)
            dice_v = (self.dice(lg, targets, sample_weight=sample_weight)
                      if self.dice is not None else None)
            return bce_v, dice_v

        def add(name: str, weight: float, fn, lg):
            nonlocal total
            if fn is None or weight <= 0:
                return
            v = fn(lg, targets, sample_weight=sample_weight)
            components[name] = v.detach()
            total = total + weight * v.float()

        # Coarse-mask losses (FINAL decoder layer)
        bce_coarse, dice_coarse = _bce_dice(logits)
        if bce_coarse is not None and self.cfg.bce_weight > 0:
            components["bce_coarse"] = bce_coarse.detach()
            total = total + self.cfg.bce_weight * bce_coarse.float()
        if dice_coarse is not None and self.cfg.dice_weight > 0:
            components["dice_coarse"] = dice_coarse.detach()
            total = total + self.cfg.dice_weight * dice_coarse.float()
        # Tversky (recall-friendly Dice variant) — dense, applied to coarse
        add("tversky_coarse",
            float(getattr(self.cfg, "tversky_weight", 0.0)),
            self.tversky, logits)
        # Boundary / Lovasz / Connectivity stay dense (they ARE boundary-aware
        # already, and aren't typically point-sampled in M2F).
        add("boundary_coarse", self.cfg.boundary_weight, self.boundary, logits)
        add("lovasz_coarse", self.cfg.lovasz_weight, self.lovasz, logits)
        add("conn_coarse", self.cfg.connectivity_weight, self.conn, logits)

        # Refined-mask losses (if refinement head active) — half weight
        if refined_logits is not None:
            bce_r, dice_r = _bce_dice(refined_logits)
            if bce_r is not None and self.cfg.bce_weight > 0:
                components["bce_refined"] = bce_r.detach()
                total = total + (self.cfg.bce_weight * 0.5) * bce_r.float()
            if dice_r is not None and self.cfg.dice_weight > 0:
                components["dice_refined"] = dice_r.detach()
                total = total + (self.cfg.dice_weight * 0.5) * dice_r.float()
            add("boundary_refined", self.cfg.boundary_weight * 0.5, self.boundary,
                refined_logits)
            add("tversky_refined",
                float(getattr(self.cfg, "tversky_weight", 0.0)) * 0.5,
                self.tversky, refined_logits)

        # Deep supervision: apply BCE+Dice to EACH decoder-layer mask
        # (excluding the last, already counted above).
        ds_w = float(getattr(self.cfg, "deep_supervision_weight", 0.0))
        if ds_w > 0 and aux_logits is not None and len(aux_logits) > 1:
            aux_to_supervise = aux_logits[:-1]    # drop final (== `logits`)
            n = len(aux_to_supervise)
            per_layer_w = ds_w / max(1, n)
            aux_bce_total = torch.zeros((), device=logits.device, dtype=torch.float32)
            aux_dice_total = torch.zeros((), device=logits.device, dtype=torch.float32)
            for al in aux_to_supervise:
                bce_a, dice_a = _bce_dice(al)
                if bce_a is not None and self.cfg.bce_weight > 0:
                    aux_bce_total = (aux_bce_total
                                     + per_layer_w * self.cfg.bce_weight * bce_a.float())
                if dice_a is not None and self.cfg.dice_weight > 0:
                    aux_dice_total = (aux_dice_total
                                      + per_layer_w * self.cfg.dice_weight * dice_a.float())
            total = total + aux_bce_total + aux_dice_total
            components["bce_aux_avg"] = (aux_bce_total / max(per_layer_w, 1e-9)).detach()
            components["dice_aux_avg"] = (aux_dice_total / max(per_layer_w, 1e-9)).detach()

        # Edge-aware auxiliary loss on the refinement head's edge head (if any)
        if (self.edge_loss is not None and edge_logits is not None
                and self.edge_loss_weight > 0):
            ev = self.edge_loss(edge_logits, targets, sample_weight=sample_weight)
            components["edge"] = ev.detach()
            total = total + self.edge_loss_weight * ev.float()

        # Iterative refinement: deep-supervision over the refinement loop's
        # intermediate iterations (the FINAL iteration's logits are already
        # supervised as `refined_logits` above).
        if (refined_iter_logits is not None and len(refined_iter_logits) > 1
                and refined_logits is not None):
            iter_aux = refined_iter_logits[:-1]            # drop final
            n = len(iter_aux)
            iter_w_total = float(getattr(self.cfg, "refinement_iter_aux_weight", 0.5))
            per_iter_w = iter_w_total / max(1, n)
            iter_bce_total = torch.zeros((), device=logits.device, dtype=torch.float32)
            iter_dice_total = torch.zeros((), device=logits.device, dtype=torch.float32)
            for il in iter_aux:
                bce_i, dice_i = _bce_dice(il)
                if bce_i is not None and self.cfg.bce_weight > 0:
                    iter_bce_total = (iter_bce_total
                                      + per_iter_w * self.cfg.bce_weight * bce_i.float())
                if dice_i is not None and self.cfg.dice_weight > 0:
                    iter_dice_total = (iter_dice_total
                                       + per_iter_w * self.cfg.dice_weight * dice_i.float())
            total = total + iter_bce_total + iter_dice_total
            components["bce_refine_iter_avg"] = (iter_bce_total / max(per_iter_w, 1e-9)).detach()
            components["dice_refine_iter_avg"] = (iter_dice_total / max(per_iter_w, 1e-9)).detach()

        components["total"] = total.detach()
        return total, components
