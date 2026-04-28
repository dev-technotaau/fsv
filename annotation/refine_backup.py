"""Enterprise-grade manual mask refinement using SAM 3 interactive predictor.

Workflow:
  • Loads each image + its auto-generated mask from dataset/annotations_v1/
  • Displays in matplotlib with the mask overlaid in red
  • Click LEFT to ADD a region to the mask (SAM 3 segments around your click)
  • Click RIGHT to REMOVE a region from the mask
  • Edits accumulate — never lose previous additions on subsequent clicks
  • Edge refinement cascade (DenseCRF → guided filter → morphology) on save
  • Detects class flips (positive ↔ negative) and tracks them
  • Robust resume — auto-jumps to next un-reviewed image on startup
  • Atomic writes — Ctrl+C never corrupts a file
  • SAM mask candidate cycling — press C to try alternative mask shapes
  • Toggle V to see auto-pipeline's original mask vs your edits

Note on PointRend:
  PointRend is a TRAINED neural-network module that refines mask boundaries
  using the source image's high-frequency features. It requires a model
  trained for the target domain (fence segmentation). We don't have one,
  and training one would be a multi-day project.

  Functional equivalent: SAM 3's mask decoder already uses high-resolution
  attention-based refinement (the same idea as PointRend in spirit).
  Combined with the DenseCRF+guided-filter+morphology cascade, boundary
  quality is at the practical ceiling for static-image segmentation.

  If you later train a PointRend head on your final 33K dataset, you can
  drop it into the EdgeRefiner.refine() method as Stage 0 (before CRF).

Class handling:
  Auto pipeline produces 3 classes (0=bg, 1=fence_wood, 2=not_target).
  Manual refinement is binary: 0=background, 1=fence_wood. The "not_target"
  absorber class was an auto-pipeline routing trick — once you've manually
  reviewed an image, you directly know fence vs not, so class 2 is gone.
  When loading an existing mask, class 2 pixels are treated as background
  initially (you can click to add them to fence if needed).

Cumulative click semantics (fixes the SAM 2 issue from golden_set tool):
  • Each click runs SAM 3 with ONLY that single point as prompt
  • Result is UNION'd (positive click) or SUBTRACTED (negative click)
    from the current mask
  • So clicking on a missed fence area NEVER loses your previous additions
  • Different from SAM 2's all-points-together behavior which can shift
    earlier regions

Controls:
  Mouse (SAM mode — default):
    Left-click   →  SAM segments + adds region to fence            green dot
    Right-click  →  SAM segments + removes region from fence       red X
    Middle-drag  →  pan (works in any mode, anytime)
    Scroll wheel →  zoom in/out at cursor (any mode, anytime)
  Mouse (BRUSH mode — press B to toggle):
    Left-click+drag   →  paint fence pixels directly (no SAM)
    Right-click+drag  →  erase fence pixels directly (no SAM)
  Keyboard:
    Space         →  SAVE refined mask + advance to next image
    Right-arrow   →  next image (auto-saves if dirty)
    Left-arrow    →  previous image (auto-saves if dirty)
    R             →  reset mask to empty (no fence at all)
    Shift+R       →  restore mask to original auto-generated state
    U             →  undo last click / brush stroke
    B             →  toggle SAM ↔ BRUSH mode
    [ / ]         →  decrease / increase brush radius
    Z             →  reset zoom to fit-image
    Q             →  quit (saves nothing extra; in-flight changes lost)
    H             →  toggle help overlay

Output written to dataset/annotations_v1/ (same as auto-pipeline):
  • masks/<id>.png             ← overwritten with binary 0/1 mask
  • masks_preview/<id>.png     ← overwritten with B/W
  • viz/<id>.png               ← overwritten with red overlay
  • manual_review.jsonl        ← appended per save (review history; authoritative)
  • qa_queue.jsonl             ← entry pruned on save (no longer needs review)
  • heatmaps/<id>.png          ← deleted on save (auto-pipeline heatmap is stale)

The auto-pipeline's results.jsonl and dataset/manifest.jsonl are TREATED AS
IMMUTABLE — manual overrides live in manual_review.jsonl. To produce the final
merged view (auto + manual overrides applied), run:

    python -m annotation.export_final
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Reduce CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")


# ─────────────────────────────────────────────────────────────────────────────
# Per-image state container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ImageState:
    image_id: str
    image_path: Path
    mask_path: Path
    image: Image.Image                   # source PIL image
    image_np: np.ndarray                  # source RGB numpy (H, W, 3)
    H: int
    W: int

    original_mask: np.ndarray             # bool (H, W) — auto-generated, frozen reference
    current_mask: np.ndarray              # bool (H, W) — working state, edited by clicks
    original_class: str | None            # from manifest "class" field (pos/neg)

    points: list = field(default_factory=list)   # list of (x, y, label) for undo
    edit_history: list = field(default_factory=list)   # list of (mask_before, point) for undo

    @property
    def dirty(self) -> bool:
        """True if current_mask differs from original_mask."""
        return not np.array_equal(self.current_mask, self.original_mask)

    @property
    def fence_pixel_count(self) -> int:
        return int(self.current_mask.sum())

    @property
    def fence_coverage(self) -> float:
        return self.fence_pixel_count / max(self.H * self.W, 1)

    @property
    def manual_class(self) -> str:
        """Derived from current_mask: 'pos' if any fence pixels, else 'neg'."""
        return "pos" if self.fence_pixel_count > 0 else "neg"


# ─────────────────────────────────────────────────────────────────────────────
# SAM 3 click-based refiner (uses the SAM 3 interactive image predictor)
# ─────────────────────────────────────────────────────────────────────────────

class SAM3ClickRefiner:
    """Wraps SAM 3's SAM-1-style interactive predictor for click-based prompts."""

    def __init__(self, device: str = "cuda") -> None:
        print(f"[sam3] loading SAM 3 image model with interactive predictor on {device}...")
        import torch
        from sam3 import build_sam3_image_model
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        import sam3
        sam3_root = Path(sam3.__file__).parent
        bpe_path = str(sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz")
        # enable_inst_interactivity=True gives us the SAM3InteractiveImagePredictor
        # which has the SAM 2-style predict(point_coords, point_labels, ...) API.
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=device,
            enable_inst_interactivity=True,
        )
        self.predictor = self.model.inst_interactive_predictor
        if self.predictor is None:
            raise RuntimeError(
                "SAM 3 interactive predictor is None — build_sam3_image_model "
                "didn't honor enable_inst_interactivity=True."
            )
        self.device = device
        self._torch = torch
        n = sum(p.numel() for p in self.model.parameters())
        print(f"  total params: {n / 1e6:.1f}M  (incl. interactive predictor)")
        print(f"  device: {next(self.model.parameters()).device}")

    @property
    def torch(self):
        return self._torch

    def encode_image(self, image_np: np.ndarray) -> None:
        """Encode the image once. All subsequent click predictions reuse cache."""
        torch = self._torch
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
            self.predictor.set_image(image_np)

    def predict_at_point(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Single-point SAM 3 prediction. Returns (masks_3xHxW, ious_3) so the
        caller can pick the best initially AND let user cycle alternatives.

        Always uses POSITIVE label internally — the caller decides whether to
        UNION (add) or SUBTRACT (remove) the result from the current mask.
        That gives 100% predictable cumulative behavior (the SAM 2 ambiguity
        bug from the golden_set tool can't happen).
        """
        torch = self._torch
        pts = np.array([[float(x), float(y)]], dtype=np.float32)
        lbls = np.array([1], dtype=np.int32)   # always positive
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
            masks, ious, _ = self.predictor.predict(
                point_coords=pts,
                point_labels=lbls,
                multimask_output=True,
            )
        if masks is None or len(masks) == 0:
            return None
        # SAM 3 returns 3 candidate masks for ambiguous single-point prompts.
        # Sort by IoU descending so caller can take [0] for best, or cycle.
        order = np.argsort(-ious)
        return (masks[order].astype(bool), ious[order].astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Edge refinement cascade (CRF → guided filter → morphology)
# ─────────────────────────────────────────────────────────────────────────────

class EdgeRefiner:
    def __init__(self, prefer_crf: bool = True):
        self._crf = self._try_load_crf() if prefer_crf else None
        self._guided = self._try_load_guided()
        avail = []
        if self._crf is not None:
            avail.append("DenseCRF")
        if self._guided is not None:
            avail.append("guided_filter")
        avail.append("morphology")
        print(f"[edge] cascade available: {' -> '.join(avail)}")

    def _try_load_crf(self):
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax
            return (dcrf, unary_from_softmax)
        except ImportError:
            return None

    def _try_load_guided(self):
        try:
            import cv2
            from cv2 import ximgproc
            return ximgproc.guidedFilter
        except (ImportError, AttributeError):
            return None

    def refine(self, mask: np.ndarray, image_np: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Apply available stages. Conservative — no stage shrinks mask drastically."""
        stages_used: list[str] = []
        if not mask.any():   # empty mask, nothing to refine
            return mask, stages_used

        # Stage 1: DenseCRF
        if self._crf is not None:
            try:
                dcrf, unary_from_softmax = self._crf
                H, W = mask.shape
                soft = np.stack([1.0 - mask.astype(np.float32),
                                 mask.astype(np.float32)], axis=0)
                # Slight smoothing to avoid degenerate softmax
                soft = soft * 0.9 + 0.05
                d = dcrf.DenseCRF2D(W, H, 2)
                d.setUnaryEnergy(unary_from_softmax(soft))
                d.addPairwiseGaussian(sxy=3, compat=3)
                d.addPairwiseBilateral(sxy=80, srgb=13,
                                       rgbim=image_np.astype(np.uint8), compat=10)
                Q = d.inference(5)
                crf_mask = np.argmax(np.array(Q).reshape(2, H, W), axis=0).astype(bool)
                # Safety: only accept CRF result if it doesn't drop too much area
                if crf_mask.sum() >= 0.5 * mask.sum():
                    mask = crf_mask
                    stages_used.append("DenseCRF")
            except Exception as e:
                print(f"  [edge] CRF failed: {type(e).__name__}: {str(e)[:80]}")

        # Stage 2: Guided filter
        if self._guided is not None:
            try:
                import cv2
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                m_f = mask.astype(np.float32)
                refined = self._guided(guide=gray, src=m_f, radius=4, eps=1e-4)
                gf_mask = refined > 0.5
                if gf_mask.sum() >= 0.7 * mask.sum():
                    mask = gf_mask
                    stages_used.append("guided_filter")
            except Exception as e:
                print(f"  [edge] guided filter failed: {type(e).__name__}: {str(e)[:80]}")

        # Stage 3: Morphology
        try:
            import cv2
            kernel = np.ones((3, 3), np.uint8)
            mu = mask.astype(np.uint8)
            mu = cv2.morphologyEx(mu, cv2.MORPH_CLOSE, kernel)
            mu = cv2.morphologyEx(mu, cv2.MORPH_OPEN, kernel)
            mask = mu.astype(bool)
            stages_used.append("morphology")
        except ImportError:
            pass
        return mask, stages_used


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _atomic_save_png(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    img.save(tmp, format="PNG", optimize=True)
    os.replace(tmp, path)


def load_existing_mask(mask_path: Path, H: int, W: int) -> np.ndarray:
    """Load auto-generated mask. Convert any class > 0 OTHER than fence_wood (1)
    to background. Returns bool (H, W) where True = fence_wood pixel.

    Why: auto-pipeline produced 3 classes (0=bg, 1=fence_wood, 2=not_target).
    For manual review, we treat class 1 as the editable fence region.
    Class 2 (not_target) was the absorber class — those pixels were detected
    as wood-like but routed away from staining. In manual review, the user
    can decide to include them by clicking.
    """
    if not mask_path.exists():
        return np.zeros((H, W), dtype=bool)
    arr = np.array(Image.open(mask_path))
    return (arr == 1)


# ─────────────────────────────────────────────────────────────────────────────
# Manual refinement application
# ─────────────────────────────────────────────────────────────────────────────

class ManualRefinementApp:
    def __init__(self, args):
        self.args = args
        self.annotations_root = Path(args.annotations_root)
        self.masks_dir = self.annotations_root / "masks"
        self.preview_dir = self.annotations_root / "masks_preview"
        self.viz_dir = self.annotations_root / "viz"
        self.heatmaps_dir = self.annotations_root / "heatmaps"
        self.review_log_path = self.annotations_root / "manual_review.jsonl"
        self.qa_queue_path = self.annotations_root / "qa_queue.jsonl"
        for d in (self.masks_dir, self.preview_dir, self.viz_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Load manifest
        manifest_path = Path(args.manifest)
        with manifest_path.open("r", encoding="utf-8") as f:
            self.rows = [json.loads(l) for l in f if l.strip()]
        self.id_to_row = {r["id"]: r for r in self.rows}
        print(f"Loaded {len(self.rows):,} manifest rows from {manifest_path}")

        # Load review history
        self.reviewed_ids: set[str] = set()
        self.last_reviewed = None    # most recently reviewed image_id
        if self.review_log_path.exists():
            with self.review_log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                        self.reviewed_ids.add(e["image_id"])
                        self.last_reviewed = e["image_id"]
                    except Exception:
                        pass
            print(f"[resume] {len(self.reviewed_ids):,} images already reviewed")

        # Load QA queue (auto-pipeline flagged these for review). On manual save
        # we prune the corresponding entry so the QA queue stays accurate.
        self.qa_queue_rows: list[dict] = []
        self.qa_queue_ids: set[str] = set()
        if self.qa_queue_path.exists():
            with self.qa_queue_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                        self.qa_queue_rows.append(e)
                        if "image_id" in e:
                            self.qa_queue_ids.add(e["image_id"])
                    except Exception:
                        pass
            n_already_reviewed = len(self.qa_queue_ids & self.reviewed_ids)
            print(f"[qa-queue] {len(self.qa_queue_ids):,} images flagged for QA "
                  f"({n_already_reviewed:,} already reviewed)")

        # Build navigation order
        if args.only_unreviewed:
            self.nav_order = [r for r in self.rows if r["id"] not in self.reviewed_ids]
            print(f"[only-unreviewed] navigating only {len(self.nav_order):,} unreviewed")
        elif args.only_class:
            self.nav_order = [r for r in self.rows if r.get("class") == args.only_class]
            print(f"[only-class={args.only_class}] navigating {len(self.nav_order):,} rows")
        else:
            self.nav_order = list(self.rows)

        if not self.nav_order:
            print("Nothing to review.")
            return

        # Find starting position. Priority:
        #   1. --start-at <id_prefix> if user specified it
        #   2. After last reviewed image (resume where you left off)
        #   3. First un-reviewed image
        #   4. Index 0 as final fallback
        self.idx = 0
        if args.start_at:
            try:
                self.idx = next(i for i, r in enumerate(self.nav_order)
                                if r["id"].startswith(args.start_at))
                print(f"[start-at] starting at index {self.idx} ({self.nav_order[self.idx]['id']})")
            except StopIteration:
                print(f"[start-at] '{args.start_at}' not found — starting at 0")
        elif self.last_reviewed:
            # Find position of last reviewed in nav_order, advance one
            try:
                last_idx = next(i for i, r in enumerate(self.nav_order)
                                if r["id"] == self.last_reviewed)
                self.idx = min(last_idx + 1, len(self.nav_order) - 1)
                print(f"[resume] continuing from after {self.last_reviewed[:8]} → index {self.idx}")
            except StopIteration:
                # Last reviewed not in nav_order (e.g. --only-class filter excluded it)
                # Fall through to first-unreviewed
                pass
        if self.idx == 0 and self.reviewed_ids:
            # Try to find first unreviewed
            try:
                self.idx = next(i for i, r in enumerate(self.nav_order)
                                if r["id"] not in self.reviewed_ids)
                if self.idx > 0:
                    print(f"[resume] first unreviewed is at index {self.idx}")
            except StopIteration:
                print(f"[resume] all images in nav_order already reviewed; starting at 0 for re-review")
                self.idx = 0

        # Build SAM 3 + edge refiner
        self.refiner = SAM3ClickRefiner(device=args.device)
        self.edge_refiner = EdgeRefiner(prefer_crf=args.crf)

        # Per-image state
        self.state: ImageState | None = None
        self._busy = False    # prevents click-spam during SAM inference
        self._show_help = False
        self._show_original_overlay = False  # toggled by V key
        # Cycling SAM mask candidates: stores last click info so C can re-pick
        self._last_click_candidates = None   # (masks_3xhw, ious_3, x, y, was_positive)
        self._last_candidate_idx = 0
        self._save_errors: list[str] = []    # surface in title if save fails

        # ── Brush mode (manual paint/erase, no SAM inference) ──
        self._mode = "sam"                   # "sam" or "brush"
        self._brush_radius = 12              # pixels in image coords
        self._brush_min_radius = 1
        self._brush_max_radius = 400
        self._brush_active = False           # True while LMB/RMB held in brush mode
        self._brush_button = None            # 1=add, 3=erase
        self._brush_prev_pt = None           # (cx, cy) for line-interpolated strokes
        self._brush_cursor = None            # matplotlib Circle patch (re-created each redraw)
        self._last_motion_xy = None          # last cursor pos in data coords (for cursor circle)
        self._last_paint_time = 0.0          # throttle paint redraws
        # ── Pan + zoom ──
        self._pan_active = False
        self._pan_start_pix = None           # (event.x, event.y) display coords at pan start
        self._pan_start_xlim = None
        self._pan_start_ylim = None
        self._pan_data_per_pix_x = 0.0
        self._pan_data_per_pix_y = 0.0
        self._zoom_active = False            # True if user has zoomed/panned (so _redraw preserves view)

        # GUI state — built in run()
        self.fig = None
        self.ax = None
        self._plt = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def run(self) -> None:
        if not self.nav_order:
            return
        import matplotlib.pyplot as plt
        self._plt = plt
        plt.rcParams["toolbar"] = "None"
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._load_at_index(self.idx)
        plt.show()
        # On window close — print session summary
        n_reviewed_in_nav = sum(1 for r in self.nav_order
                                 if r["id"] in self.reviewed_ids)
        print()
        print("=" * 60)
        print("Session summary")
        print("=" * 60)
        print(f"  reviewed (total):     {len(self.reviewed_ids):,}")
        print(f"  reviewed (this nav):  {n_reviewed_in_nav}/{len(self.nav_order)}")
        print(f"  remaining in nav:     {len(self.nav_order) - n_reviewed_in_nav}")
        if self._save_errors:
            print(f"  SAVE ERRORS:          {len(self._save_errors)}")
            for e in self._save_errors[:5]:
                print(f"    - {e}")
        print(f"  review log:           {self.review_log_path}")
        print("Done.")

    def _load_at_index(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.nav_order):
            print(f"Index {idx} out of range. Closing.")
            self._plt.close(self.fig)
            return
        self.idx = idx
        row = self.nav_order[idx]
        image_id = row["id"]
        image_path = Path(row["path"])
        mask_path = self.masks_dir / f"{image_id}.png"

        if not image_path.exists():
            print(f"  [skip missing image] {image_path}")
            self._advance(+1)
            return

        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        H, W = image_np.shape[:2]

        # Load existing auto mask (binary view)
        original_mask = load_existing_mask(mask_path, H, W)
        current_mask = original_mask.copy()

        # Encode image into SAM 3 (~1-2s)
        print(f"\n[{idx+1}/{len(self.nav_order)}] {image_id} <- {image_path.name}  "
              f"({W}x{H}, fence={100*current_mask.mean():.1f}%)")
        t = time.time()
        self.refiner.encode_image(image_np)
        print(f"  encoded in {time.time()-t:.1f}s")

        self.state = ImageState(
            image_id=image_id,
            image_path=image_path,
            mask_path=mask_path,
            image=image,
            image_np=image_np,
            H=H, W=W,
            original_mask=original_mask,
            current_mask=current_mask,
            original_class=row.get("class"),
        )
        # Reset per-image transient view + interaction state
        self._zoom_active = False
        self._brush_active = False
        self._brush_button = None
        self._brush_prev_pt = None
        self._brush_cursor = None
        self._last_motion_xy = None
        self._pan_active = False
        self._last_click_candidates = None
        self._last_candidate_idx = 0
        self._redraw()

    def _advance(self, delta: int) -> None:
        new_idx = self.idx + delta
        if new_idx < 0:
            print("  Already at first image.")
            return
        if new_idx >= len(self.nav_order):
            print("  Already at last image.")
            return
        self._load_at_index(new_idx)

    # ── click handling ────────────────────────────────────────────────────

    def _on_click(self, event) -> None:
        if self.state is None:
            return
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Middle button → start pan (any mode, anytime)
        if event.button == 2:
            self._start_pan(event)
            return

        # Brush mode: paint/erase directly (no SAM)
        if self._mode == "brush" and event.button in (1, 3):
            if x < 0 or y < 0 or x >= self.state.W or y >= self.state.H:
                return
            self._start_brush_stroke(event)
            return

        # SAM mode: original behavior
        if self._busy:
            return
        if x < 0 or y < 0 or x >= self.state.W or y >= self.state.H:
            return

        is_positive = (event.button == 1)   # left = add, right = remove
        self._busy = True
        try:
            result = self.refiner.predict_at_point(x, y)
            if result is None:
                print("  [click] SAM returned no usable mask")
                return
            masks_3, ious_3 = result
            # Validate shape
            if masks_3[0].shape != (self.state.H, self.state.W):
                print(f"  [click] SAM mask shape mismatch: {masks_3[0].shape} vs ({self.state.H}, {self.state.W})")
                return

            # Cache candidates for cycling via C key
            self._last_click_candidates = (masks_3, ious_3, x, y, is_positive)
            self._last_candidate_idx = 0
            sam_mask = masks_3[0]   # best IoU

            # Save snapshot for undo BEFORE modifying
            self.state.edit_history.append(self.state.current_mask.copy())
            self.state.points.append((float(x), float(y), 1 if is_positive else 0))

            # Cumulative semantics — never lose previous additions
            if is_positive:
                self.state.current_mask = self.state.current_mask | sam_mask
            else:
                self.state.current_mask = self.state.current_mask & ~sam_mask

            self._redraw()
        finally:
            self._busy = False

    # ── brush + pan + zoom ────────────────────────────────────────────────

    def _start_brush_stroke(self, event) -> None:
        """LMB/RMB pressed in brush mode: snapshot for undo, paint initial dab."""
        is_add = (event.button == 1)
        self._brush_active = True
        self._brush_button = event.button
        self._brush_prev_pt = None    # first paint segment is a circle at (x, y)
        self._last_paint_time = 0.0
        # Snapshot for undo BEFORE modifying
        self.state.edit_history.append(self.state.current_mask.copy())
        # Track the press point as a "click" marker (so it shows up like SAM clicks)
        self.state.points.append((float(event.xdata), float(event.ydata),
                                   1 if is_add else 0))
        # SAM candidate cycling no longer applies (last edit was a brush stroke)
        self._last_click_candidates = None
        # Apply the initial dab
        self._paint_segment(event.xdata, event.ydata, is_add)
        self._redraw()

    def _paint_segment(self, x: float, y: float, add: bool) -> None:
        """Paint or erase a brush stroke segment ending at (x, y).
        Connects from self._brush_prev_pt with a thick line so fast drags
        leave no gaps. Operates on a sub-region for speed.
        """
        if self.state is None:
            return
        cx, cy = int(round(x)), int(round(y))
        H, W = self.state.H, self.state.W
        # Clamp center into image bounds (paint can still extend outward via radius)
        cx_c = max(0, min(W - 1, cx))
        cy_c = max(0, min(H - 1, cy))
        radius = int(self._brush_radius)
        if self._brush_prev_pt is not None:
            px, py = self._brush_prev_pt
        else:
            px, py = cx_c, cy_c
        # Sub-region bounding box around the line+circle
        x_min = max(0, min(px, cx_c) - radius - 1)
        x_max = min(W, max(px, cx_c) + radius + 2)
        y_min = max(0, min(py, cy_c) - radius - 1)
        y_max = min(H, max(py, cy_c) + radius + 2)
        if x_min >= x_max or y_min >= y_max:
            self._brush_prev_pt = (cx_c, cy_c)
            return
        sub_h = y_max - y_min
        sub_w = x_max - x_min
        sub = np.zeros((sub_h, sub_w), dtype=np.uint8)
        try:
            import cv2
            if (px, py) != (cx_c, cy_c):
                cv2.line(sub,
                         (px - x_min, py - y_min),
                         (cx_c - x_min, cy_c - y_min),
                         1, thickness=max(1, 2 * radius))
            cv2.circle(sub, (cx_c - x_min, cy_c - y_min), radius, 1, -1)
        except ImportError:
            # Numpy fallback: just a disk at the current point
            yy, xx = np.ogrid[:sub_h, :sub_w]
            sub[((xx - (cx_c - x_min)) ** 2 + (yy - (cy_c - y_min)) ** 2) <= radius * radius] = 1
        sub_b = sub.astype(bool)
        region = self.state.current_mask[y_min:y_max, x_min:x_max]
        if add:
            region = region | sub_b
        else:
            region = region & ~sub_b
        self.state.current_mask[y_min:y_max, x_min:x_max] = region
        self._brush_prev_pt = (cx_c, cy_c)

    def _start_pan(self, event) -> None:
        if event.x is None or event.y is None:
            return
        self._pan_active = True
        self._pan_start_pix = (event.x, event.y)
        self._pan_start_xlim = self.ax.get_xlim()
        self._pan_start_ylim = self.ax.get_ylim()
        bbox = self.ax.bbox
        # Width can be 0 right after window resize — guard against div-by-zero
        if bbox.width <= 0 or bbox.height <= 0:
            self._pan_active = False
            return
        self._pan_data_per_pix_x = (self._pan_start_xlim[1] - self._pan_start_xlim[0]) / bbox.width
        self._pan_data_per_pix_y = (self._pan_start_ylim[1] - self._pan_start_ylim[0]) / bbox.height

    def _do_pan(self, event) -> None:
        if event.x is None or event.y is None or self._pan_start_pix is None:
            return
        dx_pix = event.x - self._pan_start_pix[0]
        dy_pix = event.y - self._pan_start_pix[1]
        dx_data = dx_pix * self._pan_data_per_pix_x
        dy_data = dy_pix * self._pan_data_per_pix_y
        new_xlim = (self._pan_start_xlim[0] - dx_data,
                    self._pan_start_xlim[1] - dx_data)
        new_ylim = (self._pan_start_ylim[0] - dy_data,
                    self._pan_start_ylim[1] - dy_data)
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self._zoom_active = True
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        """Zoom in/out at cursor position."""
        if self.state is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        cx, cy = event.xdata, event.ydata
        if event.button == "up":
            scale = 1.0 / 1.25
        elif event.button == "down":
            scale = 1.25
        else:
            return
        new_xlim = (cx + (cur_xlim[0] - cx) * scale,
                    cx + (cur_xlim[1] - cx) * scale)
        new_ylim = (cy + (cur_ylim[0] - cy) * scale,
                    cy + (cur_ylim[1] - cy) * scale)
        self._apply_zoom(new_xlim, new_ylim)

    def _apply_zoom(self, new_xlim, new_ylim) -> None:
        if self.state is None:
            return
        W, H = self.state.W, self.state.H
        width = abs(new_xlim[1] - new_xlim[0])
        height = abs(new_ylim[1] - new_ylim[0])
        # Don't zoom in past ~3 image pixels visible (silly)
        if width < 3 or height < 3:
            return
        # Don't zoom out past 2x image (keep image identifiable, snap back to fit)
        if width > W * 2.0 or height > H * 2.0:
            self._reset_zoom()
            return
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self._zoom_active = True
        self.fig.canvas.draw_idle()

    def _reset_zoom(self) -> None:
        """Restore default fit-to-image view."""
        if self.state is None:
            return
        self.ax.set_xlim(-0.5, self.state.W - 0.5)
        self.ax.set_ylim(self.state.H - 0.5, -0.5)   # inverted for image
        self._zoom_active = False
        self.fig.canvas.draw_idle()

    def _on_release(self, event) -> None:
        """End any active brush stroke or pan operation."""
        if self._brush_active and event.button == self._brush_button:
            self._brush_active = False
            self._brush_button = None
            self._brush_prev_pt = None
        if event.button == 2 and self._pan_active:
            self._pan_active = False
            self._pan_start_pix = None

    def _on_motion(self, event) -> None:
        if event.inaxes != self.ax:
            # Hide brush cursor when leaving axes
            if self._brush_cursor is not None and self._brush_cursor.get_visible():
                self._brush_cursor.set_visible(False)
                self.fig.canvas.draw_idle()
            # If mid-stroke, break the line so re-entering doesn't span the gap
            self._brush_prev_pt = None
            return
        if event.xdata is None or event.ydata is None:
            return
        self._last_motion_xy = (event.xdata, event.ydata)

        # Pan takes precedence
        if self._pan_active:
            self._do_pan(event)
            return

        # Brush stroke (held button + drag)
        if self._brush_active:
            now = time.time()
            # Throttle to ~30fps so big-image redraw doesn't lag
            if now - self._last_paint_time < 0.033:
                return
            self._last_paint_time = now
            is_add = (self._brush_button == 1)
            self._paint_segment(event.xdata, event.ydata, is_add)
            self._redraw()
            return

        # In brush mode but no drag: cheap cursor-circle update (no full redraw)
        if self._mode == "brush" and self._brush_cursor is not None:
            self._brush_cursor.set_center((event.xdata, event.ydata))
            self._brush_cursor.set_radius(self._brush_radius)
            if not self._brush_cursor.get_visible():
                self._brush_cursor.set_visible(True)
            self.fig.canvas.draw_idle()

    def _cycle_last_candidate(self) -> None:
        """C key handler: try next SAM candidate mask for last click."""
        if self._last_click_candidates is None:
            print("  [cycle] no recent click to cycle")
            return
        masks_3, ious_3, x, y, was_positive = self._last_click_candidates
        # Undo the last click's effect first
        if not self.state.edit_history:
            print("  [cycle] no edit history; can't cycle")
            return
        self.state.current_mask = self.state.edit_history[-1].copy()
        # Advance to next candidate
        self._last_candidate_idx = (self._last_candidate_idx + 1) % len(masks_3)
        sam_mask = masks_3[self._last_candidate_idx]
        if was_positive:
            self.state.current_mask = self.state.current_mask | sam_mask
        else:
            self.state.current_mask = self.state.current_mask & ~sam_mask
        print(f"  [cycle] candidate {self._last_candidate_idx + 1}/{len(masks_3)} "
              f"(IoU={ious_3[self._last_candidate_idx]:.2f})")
        self._redraw()

    # ── keyboard handling ─────────────────────────────────────────────────

    def _on_key(self, event) -> None:
        if self.state is None:
            return
        key = (event.key or "").lower()

        if key == " ":
            # Save (always — even if not dirty, replaces existing files,
            # ensuring no duplication and consistency).
            self._save_current()
            self._advance(+1)
        elif key == "right" or key == "n":
            # Auto-save if dirty, else skip without writing
            if self.state.dirty:
                self._save_current()
            self._advance(+1)
        elif key == "left":
            # Auto-save if dirty, then go back. (`b` is reserved for brush toggle.)
            if self.state.dirty:
                self._save_current()
            self._advance(-1)
        elif key == "r":
            # Reset to empty (mark as no-fence; will save as background-only)
            self.state.edit_history.append(self.state.current_mask.copy())
            self.state.current_mask = np.zeros_like(self.state.current_mask)
            self.state.points.clear()
            print("  [reset] mask cleared (will save as background-only)")
            self._redraw()
        elif key == "shift+r":
            # Restore to original auto-generated mask
            self.state.current_mask = self.state.original_mask.copy()
            self.state.edit_history.clear()
            self.state.points.clear()
            print("  [restore] mask reverted to auto-generated original")
            self._redraw()
        elif key == "u":
            # Undo last edit (click or reset)
            if self.state.edit_history:
                self.state.current_mask = self.state.edit_history.pop()
                if self.state.points:
                    self.state.points.pop()
                self._last_click_candidates = None   # candidates no longer valid
                self._redraw()
            else:
                print("  [undo] no edits to undo")
        elif key == "c":
            # Cycle through SAM's 3 candidate masks for the last click
            self._cycle_last_candidate()
        elif key == "v":
            # Toggle display of original auto-generated mask (yellow overlay)
            self._show_original_overlay = not self._show_original_overlay
            print(f"  [view] auto-mask overlay: "
                  f"{'ON' if self._show_original_overlay else 'OFF'}")
            self._redraw()
        elif key == "b":
            # Toggle SAM ↔ BRUSH mode
            self._mode = "brush" if self._mode == "sam" else "sam"
            # Cancel any in-progress brush stroke when switching
            self._brush_active = False
            self._brush_button = None
            self._brush_prev_pt = None
            print(f"  [mode] switched to {self._mode.upper()}"
                  f"{f' (radius={self._brush_radius})' if self._mode == 'brush' else ''}")
            self._redraw()
        elif key == "[":
            self._brush_radius = max(self._brush_min_radius, self._brush_radius - 2)
            print(f"  [brush] radius = {self._brush_radius}")
            self._redraw()
        elif key == "]":
            self._brush_radius = min(self._brush_max_radius, self._brush_radius + 2)
            print(f"  [brush] radius = {self._brush_radius}")
            self._redraw()
        elif key == "z":
            # Reset zoom to fit-image
            self._reset_zoom()
            self._redraw()
        elif key == "h":
            self._show_help = not self._show_help
            self._redraw()
        elif key == "q":
            # Warn if dirty — don't silently lose changes
            if self.state.dirty:
                print(f"  [quit] WARNING: unsaved changes on {self.state.image_id[:12]}!")
                print(f"         Press SPACE first to save, then Q to quit.")
                print(f"         Or press Q again to quit anyway (changes lost).")
                if getattr(self, "_quit_warned", False):
                    # Second Q press — go ahead
                    print(f"  [quit] quitting anyway at index {self.idx + 1}")
                    self._plt.close(self.fig)
                self._quit_warned = True
            else:
                print(f"  [quit] stopping at index {self.idx + 1}")
                self._plt.close(self.fig)

    # ── render ────────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        if self.state is None:
            return
        s = self.state
        # Preserve zoom/pan view across the clear() call
        saved_xlim = saved_ylim = None
        if self._zoom_active:
            saved_xlim = self.ax.get_xlim()
            saved_ylim = self.ax.get_ylim()
        self.ax.clear()
        self._brush_cursor = None    # invalidated by clear()
        # Show source image
        self.ax.imshow(s.image_np)
        # Optional: yellow overlay of ORIGINAL auto-generated mask (for comparison)
        if self._show_original_overlay and s.original_mask.any():
            orig_overlay = np.zeros((*s.original_mask.shape, 4))
            orig_overlay[s.original_mask] = [1.0, 1.0, 0.0, 0.30]   # yellow
            self.ax.imshow(orig_overlay)
        # Overlay CURRENT mask in semi-transparent red (drawn on top)
        if s.current_mask.any():
            overlay = np.zeros((*s.current_mask.shape, 4))
            overlay[s.current_mask] = [1.0, 0.0, 0.0, 0.42]
            self.ax.imshow(overlay)
        # Draw click markers
        for x, y, lbl in s.points:
            if lbl == 1:
                self.ax.plot(x, y, "o", color="#00ff66", markersize=10,
                             markeredgecolor="black", markeredgewidth=1.5)
            else:
                self.ax.plot(x, y, "x", color="#ff3030", markersize=11,
                             markeredgecolor="white", markeredgewidth=2)
        # Title — multi-line for richer info
        dirty_tag = "  *DIRTY*" if s.dirty else ""
        n_clicks = f"+{sum(1 for _,_,l in s.points if l==1)} -{sum(1 for _,_,l in s.points if l==0)}"
        cls_tag = f"class: {s.original_class or '?'} → {s.manual_class}"
        if s.original_class != s.manual_class and s.dirty:
            cls_tag += " (CHANGED)"
        # Progress: reviewed count out of nav size
        n_reviewed_in_nav = sum(1 for r in self.nav_order if r["id"] in self.reviewed_ids)
        progress = f"[{self.idx+1}/{len(self.nav_order)}]  reviewed {n_reviewed_in_nav}/{len(self.nav_order)}"
        view_tag = " [V:auto-overlay ON]" if self._show_original_overlay else ""
        cyc_tag = ""
        if self._last_click_candidates is not None:
            cyc_tag = f"  [C: candidate {self._last_candidate_idx + 1}/3]"
        err_tag = ""
        if self._save_errors:
            err_tag = f"  [SAVE ERRORS: {len(self._save_errors)}]"
        mode_tag = f"  MODE:{self._mode.upper()}"
        if self._mode == "brush":
            mode_tag += f"(r={self._brush_radius})"
        zoom_tag = "  [ZOOMED]" if self._zoom_active else ""
        self.ax.set_title(
            f"{progress}  {s.image_id[:12]}{dirty_tag}{mode_tag}{zoom_tag}  |  "
            f"fence={s.fence_coverage:.1%}  |  clicks: {n_clicks}  |  {cls_tag}{view_tag}{cyc_tag}{err_tag}\n"
            f"L=add  R=remove  SPACE=save  →=next  ←=back  R=reset  Shift+R=restore  "
            f"C=cycle  V=auto-overlay  U=undo  B=brush  [/]=radius  Z=fit  scroll=zoom  mid-drag=pan  H=help  Q=quit",
            fontsize=9,
        )
        self.ax.axis("off")

        # In brush mode, draw a cursor circle at the last known mouse position
        if self._mode == "brush":
            from matplotlib.patches import Circle
            if self._last_motion_xy is not None:
                cx, cy = self._last_motion_xy
                visible = True
            else:
                cx, cy = s.W / 2.0, s.H / 2.0
                visible = False
            edge_color = "#ff5050" if self._brush_button == 3 else "#00ddff"
            self._brush_cursor = Circle(
                (cx, cy), radius=self._brush_radius,
                fill=False, edgecolor=edge_color, linewidth=1.6,
            )
            self._brush_cursor.set_visible(visible)
            self.ax.add_patch(self._brush_cursor)

        # Restore zoom/pan if user had zoomed (otherwise imshow's defaults apply)
        if saved_xlim is not None:
            self.ax.set_xlim(saved_xlim)
            self.ax.set_ylim(saved_ylim)

        if self._show_help:
            help_text = (
                "MOUSE (SAM mode — default):\n"
                "  Left-click   SAM segments + ADDS region to fence\n"
                "  Right-click  SAM segments + REMOVES region from fence\n\n"
                "MOUSE (BRUSH mode — press B to toggle):\n"
                "  L-click+drag PAINT fence pixels (no SAM, pixel-exact)\n"
                "  R-click+drag ERASE fence pixels\n\n"
                "MOUSE (always available, any mode):\n"
                "  Middle-drag  pan the view\n"
                "  Scroll wheel zoom in/out at cursor\n\n"
                "KEYBOARD:\n"
                "  SPACE        save mask + advance to next\n"
                "  →            next image (auto-save if dirty)\n"
                "  ←            previous image (auto-save if dirty)\n"
                "  R            clear mask completely (saved as 'no fence')\n"
                "  Shift+R      restore to auto-generated original\n"
                "  C            cycle SAM's 3 candidate masks for last click\n"
                "  V            toggle yellow overlay of original auto-mask\n"
                "  U            undo last edit (SAM click OR brush stroke)\n"
                "  B            toggle SAM ↔ BRUSH mode\n"
                "  [ / ]        decrease / increase brush radius\n"
                "  Z            reset zoom to fit-image\n"
                "  H            toggle this help\n"
                "  Q            quit (warning if dirty — press Q twice)\n\n"
                "SAVE BEHAVIOR:\n"
                "  - Brush strokes mark image dirty, save the same way as SAM clicks\n"
                "  - Saves overwrite mask / preview / viz files\n"
                "  - One row appended to manual_review.jsonl per save\n"
                "  - Class flips (pos↔neg) tracked automatically\n"
                "  - Failed saves shown in title; image stays unreviewed\n\n"
                "TIPS:\n"
                "  - Use SAM for big regions, BRUSH for fine touch-up\n"
                "  - Zoom in (scroll) before brushing for pixel-accurate edits\n"
                "  - Each SAM click runs independently — earlier additions stay\n"
                "  - C cycles SAM candidates when it picks the wrong shape\n"
                "  - V shows what auto-pipeline thought (compare red vs yellow)"
            )
            self.ax.text(
                0.02, 0.98, help_text,
                transform=self.ax.transAxes, fontsize=10, family="monospace",
                color="white", verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="black", alpha=0.85),
            )
        self.fig.canvas.draw_idle()

    # ── save ──────────────────────────────────────────────────────────────

    def _save_current(self) -> None:
        if self.state is None:
            return
        s = self.state

        try:
            # Apply edge refinement only if mask non-empty AND dirty
            if s.dirty and s.current_mask.any():
                refined, stages = self.edge_refiner.refine(s.current_mask, s.image_np)
                print(f"  [save] edge refinement: {' -> '.join(stages) if stages else '(none)'}")
            else:
                refined = s.current_mask
                stages = []

            # Write class-ID mask (binary 0/1 — manual is binary, no class 2)
            class_map = refined.astype(np.uint8)
            _atomic_save_png(Image.fromarray(class_map, mode="L"), s.mask_path)
            # Preview (B/W)
            preview = (class_map * 255).astype(np.uint8)
            _atomic_save_png(Image.fromarray(preview, mode="L"),
                             self.preview_dir / f"{s.image_id}.png")
            # Viz (red overlay on source)
            viz = s.image_np.copy()
            if refined.any():
                red = np.array([255, 0, 0], dtype=np.float32)
                viz[refined] = (viz[refined] * 0.45 + red * 0.55).astype(np.uint8)
            _atomic_save_png(Image.fromarray(viz),
                             self.viz_dir / f"{s.image_id}.png")

            # Append review entry
            manual_class = "pos" if refined.any() else "neg"
            class_changed = (s.original_class is not None and
                              s.original_class != manual_class)
            entry = {
                "image_id": s.image_id,
                "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "n_clicks_positive": sum(1 for _,_,l in s.points if l == 1),
                "n_clicks_negative": sum(1 for _,_,l in s.points if l == 0),
                "fence_pixel_count_before": int(s.original_mask.sum()),
                "fence_pixel_count_after": int(refined.sum()),
                "fence_coverage_after": float(refined.mean()),
                "edit_distance": int((s.original_mask != refined).sum()),
                "edge_refinement_stages": stages,
                "original_class": s.original_class,
                "manual_class": manual_class,
                "class_changed": class_changed,
                "dirty_when_saved": s.dirty,
            }
            with self.review_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # Update tracking
            self.reviewed_ids.add(s.image_id)
            self.last_reviewed = s.image_id
            # Mark current state as the new "original" for this session — if user
            # comes back and clicks, dirty starts fresh from the saved state.
            s.original_mask = refined.copy()
            s.current_mask = refined.copy()
            s.edit_history.clear()
            s.points.clear()
            self._last_click_candidates = None
            self._quit_warned = False    # reset quit-confirmation state

            # Prune QA queue (best-effort — failure here is non-fatal)
            self._prune_qa_queue(s.image_id)
            # Stale heatmap from auto-pipeline (best-effort)
            self._delete_stale_heatmap(s.image_id)

            flip_msg = ""
            if class_changed:
                flip_msg = f"  CLASS FLIP: {s.original_class} -> {manual_class}"
            print(f"  [saved] {s.image_id}  fence={refined.mean():.1%}{flip_msg}")

        except (OSError, IOError, PermissionError) as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"
            print(f"  [SAVE FAILED] {err}")
            self._save_errors.append(f"{s.image_id}: {err}")
            # Don't update tracking — image NOT marked as reviewed
            self._redraw()  # show error tag in title

    # ── auxiliary cleanup on save ─────────────────────────────────────────

    def _prune_qa_queue(self, image_id: str) -> None:
        """Remove image_id from qa_queue.jsonl if present. Atomic rewrite.
        Best-effort — failure is logged but doesn't block the save flow."""
        if image_id not in self.qa_queue_ids:
            return
        try:
            self.qa_queue_ids.discard(image_id)
            self.qa_queue_rows = [r for r in self.qa_queue_rows
                                  if r.get("image_id") != image_id]
            tmp = self.qa_queue_path.with_suffix(self.qa_queue_path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for r in self.qa_queue_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(tmp, self.qa_queue_path)
            print(f"  [qa-queue] pruned {image_id[:12]} ({len(self.qa_queue_rows):,} remain)")
        except (OSError, IOError, PermissionError) as e:
            # Re-add to in-memory set so we'll retry next time
            self.qa_queue_ids.add(image_id)
            print(f"  [qa-queue] WARN: prune failed: {type(e).__name__}: {str(e)[:80]}")

    def _delete_stale_heatmap(self, image_id: str) -> None:
        """Delete heatmaps/<id>.png if present (auto-pipeline confidence map
        is stale after manual edit). Best-effort."""
        hp = self.heatmaps_dir / f"{image_id}.png"
        try:
            if hp.exists():
                hp.unlink()
                print(f"  [heatmap] deleted stale {hp.name}")
        except (OSError, IOError, PermissionError) as e:
            print(f"  [heatmap] WARN: delete failed: {type(e).__name__}: {str(e)[:80]}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--annotations-root", type=Path,
                    default=Path("dataset/annotations_v1"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("dataset/manifest.jsonl"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-crf", dest="crf", action="store_false", default=True,
                    help="Disable DenseCRF stage of edge refinement (faster, "
                         "minor quality loss).")
    ap.add_argument("--only-unreviewed", action="store_true",
                    help="Skip already-reviewed images (default navigates all).")
    ap.add_argument("--only-class", choices=["pos", "neg"],
                    help="Only navigate manifest rows of this class.")
    ap.add_argument("--start-at", type=str, default=None,
                    help="Image ID prefix to jump to as starting point.")
    args = ap.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")
    if not args.annotations_root.exists():
        raise SystemExit(f"Annotations root not found: {args.annotations_root}")

    app = ManualRefinementApp(args)
    app.run()


if __name__ == "__main__":
    main()
