"""Enterprise-grade manual mask refinement using SAM 3 interactive predictor.

Features (all wired in, all keyboard-accessible):

  Click-based segmentation (SAM 3):
    - Left/right click adds/removes a region (cumulative semantics -- the
      SAM 2 bug from goldenset is fixed: each click runs SAM independently)
    - C cycles SAM's 3 candidate masks for the last click
    - X + drag draws a bounding box prompt for SAM (often single-shot for
      whole fences vs. many clicks)

  Manual brush (no SAM):
    - B toggles BRUSH mode; L-drag paints, R-drag erases
    - [ / ] adjust brush radius (1-400 px)

  View / inspection:
    - Scroll wheel zooms at cursor; middle-drag pans; Z resets to fit
    - V overlays the original auto-mask in yellow (compare red vs yellow)
    - D shows diff vs auto: green=added, red=removed, yellow=kept
    - E cycles main overlay: filled -> outline -> off
    - , / .  decrease / increase mask opacity
    - Mini-map (top-right) shows viewport when zoomed
    - Info panel (right side) shows manifest+results metadata

  Workflow / ergonomics:
    - --order qa-first | coverage | random | manifest    (triage queue)
    - Background prefetch encodes the next image into SAM while you're
      reviewing the current one -- advance is near-instant
    - A: accept-as-is (save unchanged auto-mask, no edge refinement, advance)
    - SPACE: save with edge refinement, advance
    - -> / N: next image (auto-saves if dirty, skips if clean)
    - <- : previous image (auto-saves if dirty)
    - F: fill small holes; K: kill small disconnected specks
    - U: undo last edit;  R: clear mask;  Shift+R: restore auto-mask
    - H: toggle help overlay;  Q: quit (Q twice if dirty)

  Reliability:
    - Atomic writes (Ctrl+C never corrupts a file)
    - Periodic crash-safe backup of manual_review.jsonl
    - QA queue auto-pruned on save; stale heatmap auto-deleted
    - Robust resume: jumps to last reviewed + 1 (or first unreviewed)

Note on PointRend:
  Functionally equivalent: SAM 3's mask decoder already does PointRend-spirit
  attention-based refinement. Combined with DenseCRF + guided filter +
  morphology, boundary quality is at the practical ceiling without training
  a domain-specific refinement head.

Class handling:
  Auto pipeline produces 3 classes (0=bg, 1=fence_wood, 2=not_target).
  Manual refinement is binary: 0=background, 1=fence_wood. Class 2 (absorber)
  is treated as background initially -- click to include if the user judges
  that absorbed region IS actually fence.

Output written to dataset/annotations_v1/ (same as auto-pipeline):
  * masks/<id>.png             <- overwritten with binary 0/1 mask
  * masks_preview/<id>.png     <- overwritten with B/W
  * viz/<id>.png               <- overwritten with red overlay
  * manual_review.jsonl        <- appended per save (authoritative override log)
  * manual_review.jsonl.bak    <- periodic crash-safe backup
  * qa_queue.jsonl             <- entry pruned on save
  * heatmaps/<id>.png          <- deleted on save (stale post-edit)

The auto-pipeline's results.jsonl and dataset/manifest.jsonl are TREATED AS
IMMUTABLE. Manual overrides live in manual_review.jsonl. To produce the
final merged view (auto + manual overrides applied), run:

    python -m annotation.export_final
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
import shutil
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Reduce CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

# Prefer Qt5Agg over TkAgg on Windows: Qt uses native widgets with hardware
# acceleration, TkAgg goes through Tkinter PhotoImage (CPU-bound, ~3-5x slower
# for blit on large images). User can override with MPLBACKEND env var.
if "MPLBACKEND" not in os.environ:
    try:
        import PyQt5  # noqa: F401  -- presence check
        import matplotlib
        matplotlib.use("Qt5Agg", force=False)
    except ImportError:
        try:
            import PyQt6  # noqa: F401
            import matplotlib
            matplotlib.use("QtAgg", force=False)
        except ImportError:
            pass    # fall back to matplotlib's default (likely TkAgg on Windows)


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

    original_mask: np.ndarray             # bool (H, W) -- auto-generated, frozen reference
    current_mask: np.ndarray              # bool (H, W) -- working state, edited by clicks
    original_class: str | None            # from manifest "class" field (pos/neg)

    # Per-image metadata for info panel (filled in by app)
    auto_meta: dict | None = None         # row from results.jsonl
    manifest_meta: dict | None = None     # row from manifest.jsonl

    points: list = field(default_factory=list)   # list of (x, y, label) for undo
    edit_history: list = field(default_factory=list)   # list of mask snapshots for undo

    load_started_at: float = 0.0          # wall-clock time view became active

    @property
    def dirty(self) -> bool:
        return not np.array_equal(self.current_mask, self.original_mask)

    @property
    def fence_pixel_count(self) -> int:
        return int(self.current_mask.sum())

    @property
    def fence_coverage(self) -> float:
        return self.fence_pixel_count / max(self.H * self.W, 1)

    @property
    def manual_class(self) -> str:
        return "pos" if self.fence_pixel_count > 0 else "neg"


# ─────────────────────────────────────────────────────────────────────────────
# SAM 3 click-based refiner with snapshot/restore for prefetch
# ─────────────────────────────────────────────────────────────────────────────

class SAM3ClickRefiner:
    """Wraps SAM 3's interactive predictor. Adds state snapshot/restore so
    the prefetcher can encode multiple images and we can swap them in cheaply."""

    def __init__(self, device: str = "cuda") -> None:
        print(f"[sam3] loading SAM 3 image model with interactive predictor on {device}...")
        import torch
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        import sam3
        sam3_root = Path(sam3.__file__).parent
        bpe_path = str(sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz")
        self.model = build_sam3_image_model(
            bpe_path=bpe_path, device=device, enable_inst_interactivity=True,
        )
        if self.model.inst_interactive_predictor is None:
            raise RuntimeError(
                "SAM 3 interactive predictor is None -- build_sam3_image_model "
                "didn't honor enable_inst_interactivity=True."
            )
        # Sam3Processor is the ONLY supported entry point for image encoding.
        # The interactive predictor's own model.backbone is intentionally None
        # (sam3.model_builder.build_tracker is called with with_backbone=False);
        # the OUTER model's backbone is what runs forward_image, and predict_inst
        # threads features into the inst_interactive_predictor on each predict call.
        self.processor = Sam3Processor(self.model, device=device)
        self.device = device
        self._torch = torch
        self.gpu_lock = threading.Lock()    # serialize GPU ops across threads
        # The "active" inference state — set by activate_state(); used by
        # predict_at_point / predict_with_box. Contains backbone_out + sizes.
        self._active_state: dict | None = None
        n = sum(p.numel() for p in self.model.parameters())
        print(f"  total params: {n / 1e6:.1f}M  (incl. interactive predictor)")
        print(f"  device: {next(self.model.parameters()).device}")

    @property
    def torch(self):
        return self._torch

    def encode_image(self, image_np: np.ndarray) -> dict:
        """Run backbone + adapter on the image. Returns an inference state
        dict (contains backbone_out, original_height, original_width).
        This dict IS the cacheable per-image snapshot.

        Note: we pass PIL.Image not numpy because Sam3Processor.set_image has
        a bug where np.ndarray's shape[-2:] gives (W, 3) instead of (H, W)
        for HWC layout. PIL Image's .size returns (W, H) reliably.
        """
        torch = self._torch
        image_pil = Image.fromarray(image_np) if isinstance(image_np, np.ndarray) else image_np
        with self.gpu_lock:
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
                state = self.processor.set_image(image_pil)
        return state

    def activate_state(self, state: dict) -> None:
        """Make `state` the active inference state for subsequent predict() calls."""
        self._active_state = state

    def predict_at_point(self, x: float, y: float
                         ) -> tuple[np.ndarray, np.ndarray] | None:
        """Single positive point -> 3 candidate masks sorted by IoU."""
        if self._active_state is None:
            return None
        torch = self._torch
        pts = np.array([[float(x), float(y)]], dtype=np.float32)
        lbls = np.array([1], dtype=np.int32)
        with self.gpu_lock:
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
                masks, ious, _ = self.model.predict_inst(
                    self._active_state,
                    point_coords=pts, point_labels=lbls, multimask_output=True,
                )
        if masks is None or len(masks) == 0:
            return None
        order = np.argsort(-ious)
        return (masks[order].astype(bool), ious[order].astype(np.float32))

    def predict_with_box(self, x0: float, y0: float, x1: float, y1: float
                         ) -> tuple[np.ndarray, np.ndarray] | None:
        """Bounding box prompt -> 3 candidate masks sorted by IoU.
        Box is XYXY in image coords."""
        if self._active_state is None:
            return None
        torch = self._torch
        bx0, bx1 = sorted([float(x0), float(x1)])
        by0, by1 = sorted([float(y0), float(y1)])
        if (bx1 - bx0) < 2 or (by1 - by0) < 2:
            return None
        box = np.array([bx0, by0, bx1, by1], dtype=np.float32)
        with self.gpu_lock:
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
                masks, ious, _ = self.model.predict_inst(
                    self._active_state, box=box, multimask_output=True,
                )
        if masks is None or len(masks) == 0:
            return None
        order = np.argsort(-ious)
        return (masks[order].astype(bool), ious[order].astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Edge refinement cascade (DenseCRF -> guided filter -> morphology)
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

    def refine(self, mask: np.ndarray, image_np: np.ndarray
               ) -> tuple[np.ndarray, list[str]]:
        stages_used: list[str] = []
        if not mask.any():
            return mask, stages_used

        if self._crf is not None:
            try:
                dcrf, unary_from_softmax = self._crf
                H, W = mask.shape
                soft = np.stack([1.0 - mask.astype(np.float32),
                                 mask.astype(np.float32)], axis=0)
                soft = soft * 0.9 + 0.05
                d = dcrf.DenseCRF2D(W, H, 2)
                d.setUnaryEnergy(unary_from_softmax(soft))
                d.addPairwiseGaussian(sxy=3, compat=3)
                d.addPairwiseBilateral(sxy=80, srgb=13,
                                       rgbim=image_np.astype(np.uint8), compat=10)
                Q = d.inference(5)
                crf_mask = np.argmax(np.array(Q).reshape(2, H, W), axis=0).astype(bool)
                if crf_mask.sum() >= 0.5 * mask.sum():
                    mask = crf_mask
                    stages_used.append("DenseCRF")
            except Exception as e:
                print(f"  [edge] CRF failed: {type(e).__name__}: {str(e)[:80]}")

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
# Background prefetcher -- pre-loads images and pre-encodes them into SAM 3
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PrefetchEntry:
    idx: int
    image_id: str
    image: Image.Image
    image_np: np.ndarray
    original_mask: np.ndarray
    H: int
    W: int
    sam_state: dict | None = None    # populated when GPU prefetch completes


class Prefetcher:
    """Background thread that pre-loads images from disk and (when SAM is
    idle) pre-encodes them into the SAM 3 predictor. Hides ~1-3 s of dead
    time per image switch."""

    def __init__(self, refiner: SAM3ClickRefiner, masks_dir: Path,
                 max_ahead: int = 2):
        self.refiner = refiner
        self.masks_dir = masks_dir
        self.max_ahead = max(0, int(max_ahead))
        self._cache: dict[int, PrefetchEntry] = {}
        self._cache_lock = threading.Lock()
        self._task_q: "queue.Queue[tuple[int, dict] | None]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._enabled = self.max_ahead > 0

    def start(self) -> None:
        if not self._enabled:
            return
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                         name="prefetcher")
        self._thread.start()
        print(f"[prefetch] worker started (max_ahead={self.max_ahead})")

    def stop(self) -> None:
        self._stop.set()
        self._task_q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def schedule(self, idx_row_pairs: list[tuple[int, dict]]) -> None:
        """Queue image indices for prefetch. Pairs of (idx, manifest_row)."""
        if not self._enabled:
            return
        # Drop stale tasks: re-prioritize on each call
        with self._cache_lock:
            wanted = {idx for idx, _ in idx_row_pairs}
            stale = [k for k in self._cache if k not in wanted]
            # We keep a small grace cache (one back) for <--key navigation
            for k in stale[:-1]:
                self._cache.pop(k, None)
        for idx, row in idx_row_pairs:
            with self._cache_lock:
                if idx in self._cache:
                    continue
            self._task_q.put((idx, row))

    def take(self, idx: int) -> PrefetchEntry | None:
        with self._cache_lock:
            return self._cache.pop(idx, None)

    def peek(self, idx: int) -> bool:
        with self._cache_lock:
            return idx in self._cache

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                task = self._task_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if task is None:
                break
            idx, row = task
            try:
                entry = self._load_entry(idx, row)
                if entry is None:
                    continue
                # GPU encode (serialized via gpu_lock -- main thread
                # gets priority by virtue of usually being first to acquire)
                try:
                    snap = self.refiner.encode_image(entry.image_np)
                    entry.sam_state = snap
                except Exception as e:
                    # Non-fatal; main thread will encode on demand
                    print(f"  [prefetch] encode failed for {entry.image_id[:8]}: "
                          f"{type(e).__name__}: {str(e)[:60]}")
                with self._cache_lock:
                    self._cache[idx] = entry
            except Exception as e:
                print(f"  [prefetch] worker error: {type(e).__name__}: {str(e)[:80]}")

    def _load_entry(self, idx: int, row: dict) -> PrefetchEntry | None:
        image_path = Path(row["path"])
        if not image_path.exists():
            return None
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        H, W = image_np.shape[:2]
        mask_path = self.masks_dir / f"{row['id']}.png"
        original_mask = load_existing_mask(mask_path, H, W)
        return PrefetchEntry(
            idx=idx, image_id=row["id"], image=image, image_np=image_np,
            original_mask=original_mask, H=H, W=W,
        )


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _atomic_save_png(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    img.save(tmp, format="PNG", optimize=True)
    os.replace(tmp, path)


def load_existing_mask(mask_path: Path, H: int, W: int) -> np.ndarray:
    if not mask_path.exists():
        return np.zeros((H, W), dtype=bool)
    arr = np.array(Image.open(mask_path))
    return (arr == 1)


def load_jsonl_safe(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


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
        self.review_backup_path = self.annotations_root / "manual_review.jsonl.bak"
        self.qa_queue_path = self.annotations_root / "qa_queue.jsonl"
        self.results_path = self.annotations_root / "results.jsonl"
        for d in (self.masks_dir, self.preview_dir, self.viz_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ── Load manifest ───────────────────────────────────────────────
        manifest_path = Path(args.manifest)
        with manifest_path.open("r", encoding="utf-8") as f:
            self.rows = [json.loads(l) for l in f if l.strip()]
        self.id_to_row = {r["id"]: r for r in self.rows}
        print(f"Loaded {len(self.rows):,} manifest rows from {manifest_path}")

        # ── Load review history (auto-deduped: keep latest entry per image_id) ─
        self.reviewed_ids: set[str] = set()
        self.last_reviewed = None
        if self.review_log_path.exists():
            all_entries = load_jsonl_safe(self.review_log_path)
            deduped: dict[str, dict] = {}
            for e in all_entries:
                iid = e.get("image_id")
                if iid:
                    deduped[iid] = e   # last write wins
                    self.reviewed_ids.add(iid)
                    self.last_reviewed = iid
            # If the log has duplicate rows (e.g., re-saving an unchanged image
            # by going back+forward), consolidate them so the file matches the
            # in-memory state. One backup is kept the first time we dedupe.
            if len(deduped) < len(all_entries):
                n_dups = len(all_entries) - len(deduped)
                print(f"[dedupe] manual_review.jsonl has {n_dups:,} duplicate "
                      f"rows; consolidating to {len(deduped):,} unique entries")
                bak = self.review_log_path.with_suffix(".jsonl.predupe.bak")
                if not bak.exists():
                    try:
                        shutil.copy2(self.review_log_path, bak)
                        print(f"[dedupe] original backed up to {bak.name}")
                    except (OSError, IOError) as e:
                        print(f"[dedupe] WARN: backup failed: {e}")
                tmp = self.review_log_path.with_suffix(
                    self.review_log_path.suffix + ".tmp")
                try:
                    with tmp.open("w", encoding="utf-8") as f:
                        for entry in deduped.values():
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    os.replace(tmp, self.review_log_path)
                except (OSError, IOError) as e:
                    print(f"[dedupe] WARN: rewrite failed: {e}")
            print(f"[resume] {len(self.reviewed_ids):,} images already reviewed")

        # ── Load auto results (for ordering + info panel) ───────────────
        self.auto_by_id: dict[str, dict] = {}
        if self.results_path.exists():
            print(f"[results] loading {self.results_path} for triage + info panel...")
            for r in load_jsonl_safe(self.results_path):
                if "image_id" in r:
                    self.auto_by_id[r["image_id"]] = r
            print(f"[results] {len(self.auto_by_id):,} auto rows loaded")

        # ── Load QA queue ───────────────────────────────────────────────
        # qa_queue.jsonl gets pruned on every save, so it only holds the
        # currently-unreviewed flagged images. To compute "how many QA images
        # have I completed?" we cross-reference against results.jsonl's
        # immutable needs_review flag, then UNION with the current qa_queue
        # (in case some were flagged outside the needs_review mechanism).
        self.qa_queue_rows: list[dict] = load_jsonl_safe(self.qa_queue_path)
        self.qa_queue_ids: set[str] = {
            r["image_id"] for r in self.qa_queue_rows if "image_id" in r
        }
        # Original total QA set (= still-flagged + already-pruned). We can
        # reconstruct because results.jsonl is immutable, so the
        # needs_review flag still reflects the original auto-pipeline state.
        qa_original_ids = {
            iid for iid, am in self.auto_by_id.items()
            if am.get("needs_review", False)
        } | self.qa_queue_ids
        qa_reviewed = qa_original_ids & self.reviewed_ids
        qa_remaining = len(qa_original_ids) - len(qa_reviewed)
        if qa_original_ids:
            print(f"[qa-queue] {len(qa_original_ids):,} originally flagged for "
                  f"QA -- {len(qa_reviewed):,} reviewed, {qa_remaining:,} remaining")

        # ── Build navigation order (apply --order, --only-class etc.) ───
        self.nav_order = self._build_nav_order(args)
        if not self.nav_order:
            print("Nothing to review.")
            return

        # ── Resolve start index ─────────────────────────────────────────
        self.idx = self._resolve_start_index(args)

        # ── Build SAM 3 + edge refiner + prefetcher ─────────────────────
        self.refiner = SAM3ClickRefiner(device=args.device)
        self.edge_refiner = EdgeRefiner(prefer_crf=args.crf)
        n_prefetch = 0 if args.no_prefetch else max(0, int(args.prefetch))
        self.prefetcher = Prefetcher(
            self.refiner, self.masks_dir, max_ahead=n_prefetch,
        )

        # ── Per-image state ─────────────────────────────────────────────
        self.state: ImageState | None = None
        self._busy = False
        self._show_help = False
        self._show_original_overlay = False  # V key
        self._show_diff = False               # D key
        self._overlay_mode = "filled"         # "filled" | "outline" | "off"  (E key)
        self._mask_alpha = 0.42               # , / .  keys
        self._last_click_candidates = None
        self._last_candidate_idx = 0
        self._save_errors: list[str] = []

        # Brush
        self._mode = "sam"                   # "sam" or "brush"
        self._brush_radius = 12
        self._brush_min_radius = 1
        self._brush_max_radius = 400
        self._brush_active = False
        self._brush_button = None
        self._brush_prev_pt = None
        self._brush_cursor = None
        self._last_motion_xy = None
        self._last_paint_time = 0.0
        self._last_status_update_time = 0.0  # throttle motion-driven status updates

        # Persistent text artists (created once in run(), updated via set_text
        # on each refresh — much cheaper than ax.clear()+ax.text() per event).
        self._info_text_artist = None
        self._status_text_artist = None

        # Cached mask overlay artist. When this exists AND we're in filled
        # mode without diff/auto-overlay extras, brush strokes can update
        # mask via blitting (very cheap) instead of full _redraw.
        self._mask_overlay_artist = None
        # Pre-allocated uint8 RGBA buffer for the mask overlay. Sized per image
        # in _load_at_index. Reusing this avoids a 28 MB float64 allocation
        # on every brush stroke segment.
        self._overlay_buf: np.ndarray | None = None
        # Pre-composited uint8 RGB display buffer (source image with mask
        # alpha-blended in numpy). Used in filled mode to bypass matplotlib's
        # per-frame RGBA alpha-compositing — that compositing was producing
        # the "lighter brush" effect because the cached background ended up
        # with the previously-drawn mask baked in, then the new mask was
        # alpha-composited on top, doubling the alpha for the existing mask
        # and leaving brush-painted pixels at single-coat alpha.
        # By pre-composing in numpy and showing one RGB image, matplotlib
        # never does alpha compositing — every pixel ends up exactly the
        # color we computed.
        self._composed_buf: np.ndarray | None = None
        # Other animated artists that must be drawn ON TOP of the composed
        # image during blit (they'd otherwise be hidden by it). Tracked so
        # _blit_animated can render them in correct z-order.
        self._click_marker_artists: list = []      # Line2D from ax.plot
        self._help_text_artist = None              # ax.text help overlay
        # Blitting: cached background bitmap (everything except animated artists).
        # Captured automatically on every full draw via the draw_event handler.
        # Brush motion bypasses matplotlib's full-figure rasterization by:
        #   1. restore_region(bg)  -- restores cached pixels (cheap, OS-level memcpy)
        #   2. draw_artist(overlay), draw_artist(cursor)  -- only animated artists
        #   3. blit(dirty_bbox)  -- pushes only the small changed region to OS
        # Compared to draw_idle(): no axes-level rasterization, no source-image
        # texture re-upload, no marker re-render. Typically 10-50x faster.
        self._blit_bg = None
        # For partial-blit dirty-region tracking. The OS blit step on Windows
        # TkAgg scales with bbox area, so blitting only the brush stroke +
        # cursor region (typically a few hundred pixels) instead of the full
        # main axes (millions of pixels) is the largest remaining win.
        self._brush_dirty_image_bbox: tuple[int, int, int, int] | None = None
        self._prev_cursor_xy: tuple[float, float] | None = None

        # Pan + zoom
        self._pan_active = False
        self._pan_start_pix = None
        self._pan_start_xlim = None
        self._pan_start_ylim = None
        self._pan_data_per_pix_x = 0.0
        self._pan_data_per_pix_y = 0.0
        self._zoom_active = False

        # Box prompt (X + drag)
        self._box_drag_active = False
        self._box_start_xy: tuple[float, float] | None = None
        self._box_current_xy: tuple[float, float] | None = None
        self._box_armed = False              # True when X is held -> next click starts box
        self._box_rect_artist = None

        # Backup
        self._saves_since_backup = 0
        self._backup_every_n = max(1, int(args.backup_every))

        # Session timing
        self._session_start = time.time()
        self._image_durations: list[float] = []  # seconds per saved image

        # Speck threshold (K key)
        self._speck_threshold = max(1, int(args.speck_threshold))

        # GUI state
        self.fig = None
        self.ax = None       # main image axes
        self.ax_info = None  # right side panel
        self.ax_status = None  # bottom status bar
        self.ax_minimap = None  # inset in main axes
        self._plt = None

    # ── ordering / navigation construction ───────────────────────────────

    def _build_nav_order(self, args) -> list[dict]:
        rows = list(self.rows)
        if args.only_unreviewed:
            rows = [r for r in rows if r["id"] not in self.reviewed_ids]
            print(f"[only-unreviewed] {len(rows):,} rows after filter")
        if args.only_class:
            rows = [r for r in rows if r.get("class") == args.only_class]
            print(f"[only-class={args.only_class}] {len(rows):,} rows after filter")

        order = (args.order or "manifest").lower()
        if order == "manifest":
            return rows
        if order == "random":
            seed = args.seed if args.seed is not None else 42
            rng = random.Random(seed)
            rng.shuffle(rows)
            print(f"[order=random] shuffled with seed={seed}")
            return rows
        if order == "qa-first":
            qa_ids = self.qa_queue_ids
            def key(r):
                rid = r["id"]
                in_qa = rid in qa_ids
                auto = self.auto_by_id.get(rid, {})
                needs_review = bool(auto.get("needs_review", False))
                conf = float(auto.get("overall_confidence", 1.0))
                # priority: QA-flagged first, then needs_review, then low confidence
                return (not in_qa, not needs_review, conf)
            rows = sorted(rows, key=key)
            n_qa = sum(1 for r in rows if r["id"] in qa_ids)
            print(f"[order=qa-first] {n_qa:,} QA-flagged rows surfaced first")
            return rows
        if order == "coverage":
            def key(r):
                auto = self.auto_by_id.get(r["id"], {})
                return float(auto.get("fence_wood_coverage", 0.0))
            rows = sorted(rows, key=key)
            print(f"[order=coverage] sorted ascending -- likely-wrong (low coverage) first")
            return rows
        print(f"WARN: unknown --order {order!r}; falling back to manifest order")
        return rows

    def _resolve_start_index(self, args) -> int:
        if not self.nav_order:
            return 0
        if args.start_at:
            try:
                return next(i for i, r in enumerate(self.nav_order)
                             if r["id"].startswith(args.start_at))
            except StopIteration:
                print(f"[start-at] '{args.start_at}' not found -- starting at 0")
                return 0
        if self.last_reviewed:
            try:
                last_idx = next(i for i, r in enumerate(self.nav_order)
                                 if r["id"] == self.last_reviewed)
                idx = min(last_idx + 1, len(self.nav_order) - 1)
                print(f"[resume] continuing from after {self.last_reviewed[:8]} "
                      f"-> index {idx}")
                return idx
            except StopIteration:
                pass
        if self.reviewed_ids:
            try:
                idx = next(i for i, r in enumerate(self.nav_order)
                            if r["id"] not in self.reviewed_ids)
                print(f"[resume] first unreviewed at index {idx}")
                return idx
            except StopIteration:
                print(f"[resume] all images in nav_order already reviewed; index 0")
                return 0
        return 0

    # ── lifecycle ────────────────────────────────────────────────────────

    def run(self) -> None:
        if not self.nav_order:
            return
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        self._plt = plt
        plt.rcParams["toolbar"] = "None"
        # Disable matplotlib's built-in keymaps. They steal keys we use:
        #   f -> fullscreen, h/r -> home view, k/l -> log scale toggle,
        #   c/v -> back/forward, q -> quit, p/o -> pan/zoom mode,
        #   s -> save figure, g -> grid toggle.
        # Clearing every keymap.* rcParam ensures only our _on_key handler fires.
        for _km in [k for k in plt.rcParams if k.startswith("keymap.")]:
            plt.rcParams[_km] = []
        # Layout: main image (large), info panel (right ~220 px), status bar (bottom)
        self.fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(2, 2, figure=self.fig,
                      width_ratios=[5, 1], height_ratios=[40, 1],
                      wspace=0.02, hspace=0.04,
                      left=0.01, right=0.99, top=0.96, bottom=0.02)
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax_info = self.fig.add_subplot(gs[0, 1])
        self.ax_status = self.fig.add_subplot(gs[1, :])
        for ax in (self.ax_info, self.ax_status):
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        # Persistent text artists — created ONCE here, then updated via
        # .set_text() on each refresh. Avoids expensive ax.clear()+ax.text()
        # on every motion event (which is what made the GUI laggy).
        self.ax_info.set_facecolor("#202020")
        self._info_text_artist = self.ax_info.text(
            0.04, 0.98, "", transform=self.ax_info.transAxes,
            fontsize=8.5, family="monospace", color="white",
            verticalalignment="top",
        )
        self.ax_status.set_facecolor("#101010")
        self._status_text_artist = self.ax_status.text(
            0.005, 0.5, "", transform=self.ax_status.transAxes,
            fontsize=8.5, family="monospace", color="#9adfff",
            verticalalignment="center",
        )
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("key_release_event", self._on_key_release)
        self.fig.canvas.mpl_connect("close_event", lambda *_a: self.prefetcher.stop())
        # Blitting: capture canvas background after every full draw, redraw
        # animated artists on top. Invalidate cached bg on resize.
        self.fig.canvas.mpl_connect("draw_event", self._on_canvas_draw)
        self.fig.canvas.mpl_connect("resize_event", lambda *_a: setattr(self, "_blit_bg", None))

        self.prefetcher.start()
        self._load_at_index(self.idx)
        plt.show()
        self.prefetcher.stop()
        self._show_session_summary()

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

        # Try prefetch cache first
        cached = self.prefetcher.take(idx)
        if cached is not None and cached.image_id == image_id:
            image, image_np = cached.image, cached.image_np
            H, W = cached.H, cached.W
            original_mask = cached.original_mask
            sam_state = cached.sam_state
            cache_tag = " [prefetched]" if sam_state is not None else " [io-prefetched]"
        else:
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            H, W = image_np.shape[:2]
            original_mask = load_existing_mask(mask_path, H, W)
            sam_state = None
            cache_tag = ""

        current_mask = original_mask.copy()

        print(f"\n[{idx+1}/{len(self.nav_order)}] {image_id} <- {image_path.name}  "
              f"({W}x{H}, fence={100*current_mask.mean():.1f}%){cache_tag}")
        t = time.time()
        if sam_state is not None:
            self.refiner.activate_state(sam_state)
            print(f"  SAM state activated in {1000*(time.time()-t):.0f} ms")
        else:
            state = self.refiner.encode_image(image_np)
            self.refiner.activate_state(state)
            print(f"  encoded in {time.time()-t:.1f} s")

        self.state = ImageState(
            image_id=image_id, image_path=image_path, mask_path=mask_path,
            image=image, image_np=image_np, H=H, W=W,
            original_mask=original_mask, current_mask=current_mask,
            original_class=row.get("class"),
            auto_meta=self.auto_by_id.get(image_id),
            manifest_meta=row,
            load_started_at=time.time(),
        )
        # Reset per-image transient state
        self._zoom_active = False
        self._brush_active = False
        self._brush_button = None
        self._brush_prev_pt = None
        self._brush_cursor = None
        self._last_motion_xy = None
        self._pan_active = False
        self._box_drag_active = False
        self._box_start_xy = None
        self._box_current_xy = None
        self._box_rect_artist = None
        self._last_click_candidates = None
        self._last_candidate_idx = 0
        self._mask_overlay_artist = None   # invalidated by upcoming ax.clear()
        self._blit_bg = None                # will be re-captured after _redraw
        self._brush_dirty_image_bbox = None
        self._prev_cursor_xy = None
        # Pre-allocate uint8 RGBA overlay buffer (used by V/D/diff modes)
        self._overlay_buf = np.zeros((H, W, 4), dtype=np.uint8)
        # Pre-composited RGB display buffer (source + mask blended). Used in
        # the fast brush path to avoid matplotlib alpha-compositing entirely.
        self._composed_buf = image_np.copy()
        self._redraw()

        # Kick off prefetch for next images
        self._kick_prefetch()

    def _kick_prefetch(self) -> None:
        if not self.prefetcher._enabled:
            return
        ahead = self.prefetcher.max_ahead
        pairs: list[tuple[int, dict]] = []
        for d in range(1, ahead + 1):
            j = self.idx + d
            if 0 <= j < len(self.nav_order):
                pairs.append((j, self.nav_order[j]))
        # Also keep <--key snappy: prefetch idx-1 if not already saved
        j = self.idx - 1
        if 0 <= j < len(self.nav_order):
            pairs.append((j, self.nav_order[j]))
        self.prefetcher.schedule(pairs)

    def _advance(self, delta: int) -> None:
        new_idx = self.idx + delta
        if new_idx < 0:
            print("  Already at first image.")
            return
        if new_idx >= len(self.nav_order):
            print("  Already at last image.")
            return
        self._load_at_index(new_idx)

    # ── click / mouse handling ────────────────────────────────────────────

    def _on_click(self, event) -> None:
        if self.state is None:
            return
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Middle button -> pan (any mode)
        if event.button == 2:
            self._start_pan(event)
            return

        # X armed -> start box drag (left button only)
        if self._box_armed and event.button == 1:
            if x < 0 or y < 0 or x >= self.state.W or y >= self.state.H:
                return
            self._start_box_drag(event)
            return

        # Brush mode -> paint/erase
        if self._mode == "brush" and event.button in (1, 3):
            if x < 0 or y < 0 or x >= self.state.W or y >= self.state.H:
                return
            self._start_brush_stroke(event)
            return

        # Default: SAM single-click
        if self._busy:
            return
        if x < 0 or y < 0 or x >= self.state.W or y >= self.state.H:
            return
        is_positive = (event.button == 1)
        self._busy = True
        try:
            result = self.refiner.predict_at_point(x, y)
            if result is None:
                print("  [click] SAM returned no usable mask")
                return
            masks_3, ious_3 = result
            if masks_3[0].shape != (self.state.H, self.state.W):
                print(f"  [click] SAM mask shape mismatch: {masks_3[0].shape} "
                      f"vs ({self.state.H}, {self.state.W})")
                return
            self._last_click_candidates = (masks_3, ious_3, x, y, is_positive)
            self._last_candidate_idx = 0
            sam_mask = masks_3[0]
            self.state.edit_history.append(self.state.current_mask.copy())
            self.state.points.append((float(x), float(y), 1 if is_positive else 0))
            if is_positive:
                self.state.current_mask = self.state.current_mask | sam_mask
            else:
                self.state.current_mask = self.state.current_mask & ~sam_mask
            self._redraw()
        finally:
            self._busy = False

    def _on_motion(self, event) -> None:
        if event.inaxes != self.ax:
            if self._brush_cursor is not None and self._brush_cursor.get_visible():
                self._brush_cursor.set_visible(False)
                # Use blit so the cursor disappears immediately without
                # rasterizing the whole figure
                if self._blit_bg is not None:
                    self.fig.canvas.restore_region(self._blit_bg)
                    self._blit_animated()
                    self.fig.canvas.blit(self.ax.bbox)
                else:
                    self.fig.canvas.draw_idle()
            self._brush_prev_pt = None
            return
        if event.xdata is None or event.ydata is None:
            return
        self._last_motion_xy = (event.xdata, event.ydata)

        if self._pan_active:
            self._do_pan(event)
            return

        if self._box_drag_active:
            self._do_box_drag(event)
            return

        if self._brush_active:
            now = time.time()
            # ~165 fps cap to match a 165 Hz display. If a frame's blit takes
            # longer than 6 ms (the natural frame budget at 165 Hz), the next
            # motion event will skip until the budget elapses — preventing a
            # backlog of work but never throttling below display refresh rate.
            if now - self._last_paint_time < 0.006:
                return
            self._last_paint_time = now
            is_add = (self._brush_button == 1)
            self._paint_segment(event.xdata, event.ydata, is_add)
            self._fast_update_mask()
            return

        if self._mode == "brush" and self._brush_cursor is not None:
            if not self._brush_cursor.get_visible():
                self._brush_cursor.set_visible(True)
            # Use the blit fast-path for cursor-only hover updates too
            self._blit_cursor_only()
        # In SAM mode (no brush, no drag): do nothing on bare cursor motion.
        # Updating the status bar with cursor coords on every move triggers
        # a full figure redraw — even at 15 Hz that produces visible lag on
        # large images. The status bar still refreshes on click/key events
        # via _redraw, which is enough to keep the readouts current.

    def _on_release(self, event) -> None:
        if self._brush_active and event.button == self._brush_button:
            self._brush_active = False
            self._brush_button = None
            self._brush_prev_pt = None
        if event.button == 2 and self._pan_active:
            self._pan_active = False
            self._pan_start_pix = None
        if self._box_drag_active and event.button == 1:
            self._finish_box_drag(event)

    # ── pan / zoom ────────────────────────────────────────────────────────

    def _start_pan(self, event) -> None:
        if event.x is None or event.y is None:
            return
        self._pan_active = True
        self._pan_start_pix = (event.x, event.y)
        self._pan_start_xlim = self.ax.get_xlim()
        self._pan_start_ylim = self.ax.get_ylim()
        bbox = self.ax.bbox
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
        self.ax.set_xlim((self._pan_start_xlim[0] - dx_data,
                          self._pan_start_xlim[1] - dx_data))
        self.ax.set_ylim((self._pan_start_ylim[0] - dy_data,
                          self._pan_start_ylim[1] - dy_data))
        self._zoom_active = True
        self._draw_minimap_viewport()
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event) -> None:
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
        if width < 3 or height < 3:
            return
        if width > W * 2.0 or height > H * 2.0:
            self._reset_zoom()
            return
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self._zoom_active = True
        self._draw_minimap_viewport()
        self._update_status_bar()
        self.fig.canvas.draw_idle()

    def _reset_zoom(self) -> None:
        if self.state is None:
            return
        self.ax.set_xlim(-0.5, self.state.W - 0.5)
        self.ax.set_ylim(self.state.H - 0.5, -0.5)
        self._zoom_active = False
        self._draw_minimap_viewport()
        self._update_status_bar()
        self.fig.canvas.draw_idle()

    # ── brush ─────────────────────────────────────────────────────────────

    def _blend_mask_into_composed(self, source: np.ndarray, mask: np.ndarray,
                                    out: np.ndarray, bbox=None) -> None:
        """Compute source*(1-alpha) + red*alpha for masked pixels and write
        into `out`. If bbox=(y0, y1, x0, x1) is provided, only that region is
        recomputed; otherwise the whole image. `out` is modified in place
        (uint8 RGB). `source` and `out` must be the same shape."""
        if bbox is None:
            sub_src = source
            sub_mask = mask
            sub_out = out
        else:
            y0, y1, x0, x1 = bbox
            sub_src = source[y0:y1, x0:x1]
            sub_mask = mask[y0:y1, x0:x1]
            sub_out = out[y0:y1, x0:x1]
        # Start from source (also clears any previous mask blend in this region)
        np.copyto(sub_out, sub_src)
        if not sub_mask.any():
            return
        alpha = float(self._mask_alpha)
        inv = 1.0 - alpha
        # Per-channel blend on masked pixels: out = src*inv + red*alpha
        # Using int math for speed: result = (src*inv + 255*alpha) clipped
        masked_src = sub_src[sub_mask].astype(np.float32)
        blended = masked_src * inv
        blended[:, 0] += 255.0 * alpha   # R channel
        # G, B channels: red*alpha = 0
        sub_out[sub_mask] = blended.clip(0, 255).astype(np.uint8)

    def _start_brush_stroke(self, event) -> None:
        is_add = (event.button == 1)
        self._brush_active = True
        self._brush_button = event.button
        self._brush_prev_pt = None
        self._last_paint_time = 0.0
        self.state.edit_history.append(self.state.current_mask.copy())
        self.state.points.append((float(event.xdata), float(event.ydata),
                                   1 if is_add else 0))
        self._last_click_candidates = None
        self._paint_segment(event.xdata, event.ydata, is_add)
        self._redraw()

    def _paint_segment(self, x: float, y: float, add: bool) -> None:
        if self.state is None:
            return
        cx, cy = int(round(x)), int(round(y))
        H, W = self.state.H, self.state.W
        cx_c = max(0, min(W - 1, cx))
        cy_c = max(0, min(H - 1, cy))
        radius = int(self._brush_radius)
        if self._brush_prev_pt is not None:
            px, py = self._brush_prev_pt
        else:
            px, py = cx_c, cy_c
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
            yy, xx = np.ogrid[:sub_h, :sub_w]
            sub[((xx - (cx_c - x_min)) ** 2 + (yy - (cy_c - y_min)) ** 2) <= radius * radius] = 1
        sub_b = sub.astype(bool)
        region = self.state.current_mask[y_min:y_max, x_min:x_max]
        if add:
            region = region | sub_b
        else:
            region = region & ~sub_b
        self.state.current_mask[y_min:y_max, x_min:x_max] = region
        # Also update the COMPOSED buffer's bbox in-place. This is what the
        # fast brush path actually displays; pre-composing in numpy means
        # matplotlib never alpha-blends on its end (which was the source of
        # the "lighter brush" bug).
        if self._composed_buf is not None and self._overlay_mode == "filled" \
                and not self._show_diff:
            self._blend_mask_into_composed(
                self.state.image_np, self.state.current_mask,
                self._composed_buf, bbox=(y_min, y_max, x_min, x_max),
            )
        # Record dirty image bbox so _fast_update_mask can blit only this
        # region (instead of the full main axes — huge OS-blit cost win)
        if self._brush_dirty_image_bbox is None:
            self._brush_dirty_image_bbox = (x_min, y_min, x_max, y_max)
        else:
            ox0, oy0, ox1, oy1 = self._brush_dirty_image_bbox
            self._brush_dirty_image_bbox = (
                min(ox0, x_min), min(oy0, y_min),
                max(ox1, x_max), max(oy1, y_max),
            )
        self._brush_prev_pt = (cx_c, cy_c)

    # ── blitting fast-path ───────────────────────────────────────────────

    def _on_canvas_draw(self, event) -> None:
        """Hook into matplotlib's draw cycle: AFTER every full draw, capture
        the canvas as a background bitmap (excludes animated artists, since
        matplotlib skips them in regular draws). Then re-render the animated
        artists on top so the user sees them.

        From this point on, _fast_update_mask() and _blit_cursor_only()
        can do near-instant updates by restoring the bitmap and re-drawing
        only the animated artists.
        """
        if self.state is None:
            return
        try:
            # Only capture the main image axes (not status bar / info panel),
            # so updates to those don't waste a copy.
            self._blit_bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self._blit_animated()
        except Exception:
            # On some backends copy_from_bbox can fail (e.g., headless);
            # fall back to no-blit mode.
            self._blit_bg = None

    def _blit_animated(self) -> None:
        """Draw all animated artists on top of whatever's on the canvas now,
        in z-order: composed image (bottom) -> click markers -> box rect ->
        help overlay -> brush cursor (top)."""
        # 1. Composed source+mask image (the big bottom layer)
        if self._mask_overlay_artist is not None:
            self.ax.draw_artist(self._mask_overlay_artist)
        # 2. SAM/brush click markers (green dots / red X)
        for art in self._click_marker_artists:
            self.ax.draw_artist(art)
        # 3. Box-prompt rectangle (only when armed + dragging)
        if self._box_rect_artist is not None:
            self.ax.draw_artist(self._box_rect_artist)
        # 4. Help overlay text (when H toggled on)
        if self._help_text_artist is not None and self._help_text_artist.get_visible():
            self.ax.draw_artist(self._help_text_artist)
        # 5. Brush cursor (always on top so user sees it everywhere)
        if self._brush_cursor is not None and self._brush_cursor.get_visible():
            self.ax.draw_artist(self._brush_cursor)

    def _compute_dirty_blit_bbox(self):
        """Union of brush-stroke dirty region + cursor positions (current and
        previous), in display pixel coords. Returned as a matplotlib Bbox so
        the blit step can push only that region to the OS canvas."""
        from matplotlib.transforms import Bbox
        if self.ax is None:
            return None
        boxes: list[tuple[float, float, float, float]] = []
        # Brush stroke region (image -> display coords)
        if self._brush_dirty_image_bbox is not None:
            x0, y0, x1, y1 = self._brush_dirty_image_bbox
            d00 = self.ax.transData.transform((x0, y0))
            d11 = self.ax.transData.transform((x1, y1))
            boxes.append((min(d00[0], d11[0]), min(d00[1], d11[1]),
                          max(d00[0], d11[0]), max(d00[1], d11[1])))
            self._brush_dirty_image_bbox = None    # consume
        # Cursor positions (current + previous), padded by brush radius
        if self._brush_cursor is not None:
            origin_disp = self.ax.transData.transform((0, 0))
            r_disp = abs(
                self.ax.transData.transform((self._brush_radius, 0))[0]
                - origin_disp[0]
            )
            pad = r_disp + 4
            for xy in (self._prev_cursor_xy, self._last_motion_xy):
                if xy is None:
                    continue
                cx, cy = self.ax.transData.transform(xy)
                boxes.append((cx - pad, cy - pad, cx + pad, cy + pad))
        if not boxes:
            return None
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        y1 = max(b[3] for b in boxes)
        # Clip to ax.bbox (display coords)
        ax_bb = self.ax.bbox
        x0 = max(x0, ax_bb.x0)
        y0 = max(y0, ax_bb.y0)
        x1 = min(x1, ax_bb.x1)
        y1 = min(y1, ax_bb.y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return Bbox.from_extents(x0, y0, x1, y1)

    def _fast_update_mask(self) -> None:
        """Mask refresh during a brush stroke via blitting. Restore the
        cached background bitmap, redraw the (updated) overlay + cursor,
        blit. Falls back to draw_idle() when blit unavailable."""
        if self.state is None:
            return
        fast_ok = (
            self._mask_overlay_artist is not None
            and self._composed_buf is not None
            and self._overlay_mode == "filled"
            and not self._show_diff
        )
        if not fast_ok:
            self._redraw()
            return
        # Update animated artist state (set_data is cheap; the composed buffer
        # was already updated in-place by _paint_segment within just the
        # brush bbox — most pixels untouched).
        self._mask_overlay_artist.set_data(self._composed_buf)
        if self._brush_cursor is not None and self._last_motion_xy is not None:
            self._brush_cursor.set_center(self._last_motion_xy)
            self._brush_cursor.set_radius(self._brush_radius)
        if self._blit_bg is not None:
            self.fig.canvas.restore_region(self._blit_bg)
            self._blit_animated()
            self.fig.canvas.blit(self.ax.bbox)
            # flush_events() forces the OS to display the new frame
            # synchronously. Without it, Qt may batch redraws and present
            # them out of step with the cursor — looks like jitter.
            self.fig.canvas.flush_events()
        else:
            # Background not yet captured (first frame after _redraw) — request
            # a full draw so the draw_event handler captures it.
            self.fig.canvas.draw_idle()

    def _blit_cursor_only(self) -> None:
        """Cheap cursor-circle update for hover (no brush stroke active).
        Keeps cursor smooth without a full redraw."""
        if self._brush_cursor is None or self._last_motion_xy is None:
            return
        self._brush_cursor.set_center(self._last_motion_xy)
        self._brush_cursor.set_radius(self._brush_radius)
        if self._blit_bg is not None:
            self.fig.canvas.restore_region(self._blit_bg)
            self._blit_animated()
            self.fig.canvas.blit(self.ax.bbox)
            self.fig.canvas.flush_events()
        else:
            self.fig.canvas.draw_idle()

    # ── box prompt (X + drag) ─────────────────────────────────────────────

    def _start_box_drag(self, event) -> None:
        from matplotlib.patches import Rectangle
        self._box_drag_active = True
        self._box_start_xy = (float(event.xdata), float(event.ydata))
        self._box_current_xy = self._box_start_xy
        # Draw a visible rectangle that follows the cursor
        rect = Rectangle(self._box_start_xy, 0, 0, fill=False,
                         edgecolor="#00ff66", linewidth=2.0, linestyle="--",
                         animated=True)
        self._box_rect_artist = rect
        self.ax.add_patch(rect)
        # Use blit so the rect appears on top of the composed image
        if self._blit_bg is not None:
            self.fig.canvas.restore_region(self._blit_bg)
            self._blit_animated()
            self.fig.canvas.blit(self.ax.bbox)
            self.fig.canvas.flush_events()
        else:
            self.fig.canvas.draw_idle()

    def _do_box_drag(self, event) -> None:
        if self._box_start_xy is None or self._box_rect_artist is None:
            return
        x0, y0 = self._box_start_xy
        x1, y1 = float(event.xdata), float(event.ydata)
        self._box_current_xy = (x1, y1)
        self._box_rect_artist.set_xy((min(x0, x1), min(y0, y1)))
        self._box_rect_artist.set_width(abs(x1 - x0))
        self._box_rect_artist.set_height(abs(y1 - y0))
        # Blit so the rect updates smoothly without full-figure redraw
        if self._blit_bg is not None:
            self.fig.canvas.restore_region(self._blit_bg)
            self._blit_animated()
            self.fig.canvas.blit(self.ax.bbox)
            self.fig.canvas.flush_events()
        else:
            self.fig.canvas.draw_idle()

    def _finish_box_drag(self, event) -> None:
        if not self._box_drag_active:
            return
        x0, y0 = self._box_start_xy or (0, 0)
        x1 = float(event.xdata) if event.xdata is not None else self._box_current_xy[0]
        y1 = float(event.ydata) if event.ydata is not None else self._box_current_xy[1]
        # Remove the temporary rectangle artist
        if self._box_rect_artist is not None:
            try:
                self._box_rect_artist.remove()
            except Exception:
                pass
            self._box_rect_artist = None
        self._box_drag_active = False
        # ONE box drag per X-arm. Reset _box_armed so the user's NEXT click
        # is a normal SAM click, not another box. (Without this, the next
        # click was being interpreted as a fresh box drag start, which is
        # what made box mode feel "broken" after the first use.)
        self._box_armed = False
        self._update_status_bar()
        # Run SAM with the box prompt
        if self.state is None:
            return
        try:
            self._busy = True
            result = self.refiner.predict_with_box(x0, y0, x1, y1)
            if result is None:
                print("  [box] too small or no mask returned")
                return
            masks_3, ious_3 = result
            if masks_3[0].shape != (self.state.H, self.state.W):
                print(f"  [box] mask shape mismatch")
                return
            sam_mask = masks_3[0]
            self.state.edit_history.append(self.state.current_mask.copy())
            # Box prompts default to ADD (positive). Use negative box via Shift+X if needed.
            # Track the box centroid as a positive marker.
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            self.state.points.append((cx, cy, 1))
            self._last_click_candidates = (masks_3, ious_3, cx, cy, True)
            self._last_candidate_idx = 0
            self.state.current_mask = self.state.current_mask | sam_mask
            print(f"  [box] added region (best IoU={ious_3[0]:.2f})")
            self._redraw()
        finally:
            self._busy = False

    # ── candidate cycling ────────────────────────────────────────────────

    def _cycle_last_candidate(self) -> None:
        if self._last_click_candidates is None:
            print("  [cycle] no recent click to cycle")
            return
        masks_3, ious_3, x, y, was_positive = self._last_click_candidates
        if not self.state.edit_history:
            print("  [cycle] no edit history; can't cycle")
            return
        self.state.current_mask = self.state.edit_history[-1].copy()
        self._last_candidate_idx = (self._last_candidate_idx + 1) % len(masks_3)
        sam_mask = masks_3[self._last_candidate_idx]
        if was_positive:
            self.state.current_mask = self.state.current_mask | sam_mask
        else:
            self.state.current_mask = self.state.current_mask & ~sam_mask
        print(f"  [cycle] candidate {self._last_candidate_idx + 1}/{len(masks_3)} "
              f"(IoU={ious_3[self._last_candidate_idx]:.2f})")
        self._redraw()

    # ── auto-cleanup macros ──────────────────────────────────────────────

    def _do_fill_holes(self) -> None:
        """Morphological close to fill small interior holes in the fence mask."""
        if self.state is None or not self.state.current_mask.any():
            print("  [fill] empty mask, nothing to fill")
            return
        try:
            import cv2
        except ImportError:
            print("  [fill] cv2 unavailable")
            return
        before = self.state.current_mask.copy()
        mu = self.state.current_mask.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mu = cv2.morphologyEx(mu, cv2.MORPH_CLOSE, kernel, iterations=2)
        after = mu.astype(bool)
        if np.array_equal(before, after):
            print("  [fill] no change")
            return
        self.state.edit_history.append(before)
        self.state.current_mask = after
        added = int(after.sum() - before.sum())
        print(f"  [fill] holes filled (+{added:,} pixels)")
        self._redraw()

    def _do_kill_specks(self) -> None:
        """Remove disconnected components smaller than --speck-threshold."""
        if self.state is None or not self.state.current_mask.any():
            print("  [kill] empty mask, nothing to clean")
            return
        try:
            import cv2
        except ImportError:
            print("  [kill] cv2 unavailable")
            return
        before = self.state.current_mask.copy()
        mu = before.astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mu, connectivity=8)
        if n <= 1:
            print("  [kill] no components")
            return
        keep = np.zeros_like(mu, dtype=bool)
        kept = 0
        killed = 0
        killed_pixels = 0
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= self._speck_threshold:
                keep |= (labels == i)
                kept += 1
            else:
                killed += 1
                killed_pixels += area
        if np.array_equal(before, keep):
            print(f"  [kill] all {kept} components above threshold ({self._speck_threshold} px)")
            return
        self.state.edit_history.append(before)
        self.state.current_mask = keep
        print(f"  [kill] removed {killed} specks ({killed_pixels:,} pixels), kept {kept}")
        self._redraw()

    def _do_accept(self) -> None:
        """Save without edge refinement and advance. Equivalent to confirming
        the auto-mask is correct as-is."""
        if self.state is None:
            return
        # If user hasn't edited, this preserves the auto-mask byte-for-byte.
        # If they HAVE edited, accept saves the current edits as-is (no CRF).
        self._save_current(skip_edge_refinement=True, accept_mode=True)
        self._advance(+1)

    # ── keyboard ──────────────────────────────────────────────────────────

    def _on_key(self, event) -> None:
        if self.state is None:
            return
        key = (event.key or "").lower()

        # X: arm box-prompt mode (next left-click+drag becomes a box)
        if key == "x":
            if not self._box_armed:
                self._box_armed = True
                print("  [box] armed -- next L-click+drag will draw a box for SAM")
                self._update_status_bar()
                self.fig.canvas.draw_idle()
            return

        if key == " ":
            self._save_current()
            self._advance(+1)
        elif key == "right" or key == "n":
            if self.state.dirty:
                self._save_current()
            self._advance(+1)
        elif key == "left":
            if self.state.dirty:
                self._save_current()
            self._advance(-1)
        elif key == "a":
            self._do_accept()
        elif key == "f":
            self._do_fill_holes()
        elif key == "k":
            self._do_kill_specks()
        elif key == "r":
            self.state.edit_history.append(self.state.current_mask.copy())
            self.state.current_mask = np.zeros_like(self.state.current_mask)
            self.state.points.clear()
            print("  [reset] mask cleared (will save as background-only)")
            self._redraw()
        elif key == "shift+r":
            self.state.current_mask = self.state.original_mask.copy()
            self.state.edit_history.clear()
            self.state.points.clear()
            print("  [restore] mask reverted to auto-generated original")
            self._redraw()
        elif key == "u":
            if self.state.edit_history:
                self.state.current_mask = self.state.edit_history.pop()
                if self.state.points:
                    self.state.points.pop()
                self._last_click_candidates = None
                self._redraw()
            else:
                print("  [undo] no edits to undo")
        elif key == "c":
            self._cycle_last_candidate()
        elif key == "v":
            self._show_original_overlay = not self._show_original_overlay
            print(f"  [view] auto-mask overlay: "
                  f"{'ON' if self._show_original_overlay else 'OFF'}")
            self._redraw()
        elif key == "d":
            self._show_diff = not self._show_diff
            print(f"  [diff] vs auto: {'ON' if self._show_diff else 'OFF'}")
            self._redraw()
        elif key == "e":
            cycle = ["filled", "outline", "off"]
            self._overlay_mode = cycle[(cycle.index(self._overlay_mode) + 1) % 3]
            print(f"  [overlay] mode = {self._overlay_mode}")
            self._redraw()
        elif key == "b":
            self._mode = "brush" if self._mode == "sam" else "sam"
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
        elif key == ",":
            self._mask_alpha = max(0.05, self._mask_alpha - 0.07)
            print(f"  [opacity] mask alpha = {self._mask_alpha:.2f}")
            self._redraw()
        elif key == ".":
            self._mask_alpha = min(0.95, self._mask_alpha + 0.07)
            print(f"  [opacity] mask alpha = {self._mask_alpha:.2f}")
            self._redraw()
        elif key == "z":
            self._reset_zoom()
            self._redraw()
        elif key == "h":
            self._show_help = not self._show_help
            self._redraw()
        elif key == "q":
            if self.state.dirty:
                print(f"  [quit] WARNING: unsaved changes on {self.state.image_id[:12]}!")
                print(f"         Press SPACE first to save, then Q to quit.")
                print(f"         Or press Q again to quit anyway (changes lost).")
                if getattr(self, "_quit_warned", False):
                    print(f"  [quit] quitting anyway at index {self.idx + 1}")
                    self._plt.close(self.fig)
                self._quit_warned = True
            else:
                print(f"  [quit] stopping at index {self.idx + 1}")
                self._plt.close(self.fig)

    def _on_key_release(self, event) -> None:
        key = (event.key or "").lower()
        if key == "x" and self._box_armed and not self._box_drag_active:
            # Releasing X without starting a drag disarms it
            self._box_armed = False
            self._update_status_bar()
            self.fig.canvas.draw_idle()

    # ── render: main image, overlays, mini-map, info, status ─────────────

    def _redraw(self) -> None:
        if self.state is None:
            return
        s = self.state

        # Save zoom/pan state
        saved_xlim = saved_ylim = None
        if self._zoom_active:
            saved_xlim = self.ax.get_xlim()
            saved_ylim = self.ax.get_ylim()

        self.ax.clear()
        self._brush_cursor = None
        self._box_rect_artist = None
        self._mask_overlay_artist = None    # invalidated by ax.clear()
        self._click_marker_artists = []     # invalidated by ax.clear()
        self._help_text_artist = None       # invalidated by ax.clear()
        if self.ax_minimap is not None:
            try:
                self.ax_minimap.remove()
            except Exception:
                pass
            self.ax_minimap = None

        # 1) Source image. interpolation='nearest' is critical for perf —
        # default 'antialiased' triggers a slow filtered resample of the full
        # image on every draw (~10-30ms for 2160x1620). 'nearest' is a fast
        # integer lookup. Visual difference is negligible at typical zoom
        # levels for this workflow.
        self.ax.imshow(s.image_np, interpolation="nearest")

        # 2) Optional yellow overlay of ORIGINAL auto-mask
        if self._show_original_overlay and s.original_mask.any():
            yel = np.zeros((*s.original_mask.shape, 4))
            yel[s.original_mask] = [1.0, 1.0, 0.0, 0.30]
            self.ax.imshow(yel)

        # 3) Main mask overlay -- diff mode OR overlay mode (filled/outline/off)
        if self._show_diff:
            self._draw_diff_overlay(s)
        else:
            self._draw_main_overlay(s)

        # 4) Click markers
        for x, y, lbl in s.points:
            if lbl == 1:
                lines = self.ax.plot(x, y, "o", color="#00ff66", markersize=8,
                                      markeredgecolor="black", markeredgewidth=1.2)
            else:
                lines = self.ax.plot(x, y, "x", color="#ff3030", markersize=10,
                                      markeredgecolor="white", markeredgewidth=1.8)
            # animated=True so blit treats these as overlay artists drawn on
            # top of the composed image (otherwise the composed RGB layer
            # would hide them).
            for ln in lines:
                ln.set_animated(True)
                self._click_marker_artists.append(ln)

        # Title -- concise; full info goes into status bar + info panel
        dirty_tag = "  *DIRTY*" if s.dirty else ""
        cls_tag = f"{s.original_class or '?'} -> {s.manual_class}"
        if s.original_class != s.manual_class and s.dirty:
            cls_tag += " (CHANGED)"
        n_reviewed_in_nav = sum(1 for r in self.nav_order
                                 if r["id"] in self.reviewed_ids)
        progress = f"[{self.idx+1}/{len(self.nav_order)}]  reviewed {n_reviewed_in_nav}/{len(self.nav_order)}"
        self.ax.set_title(
            f"{progress}   {s.image_id[:16]}{dirty_tag}   class: {cls_tag}   "
            f"fence={s.fence_coverage:.1%}",
            fontsize=10,
        )
        self.ax.axis("off")

        # 5) Brush cursor
        if self._mode == "brush":
            from matplotlib.patches import Circle
            if self._last_motion_xy is not None:
                cx, cy = self._last_motion_xy
                visible = True
            else:
                cx, cy = s.W / 2.0, s.H / 2.0
                visible = False
            edge_color = "#ff5050" if self._brush_button == 3 else "#00ddff"
            self._brush_cursor = Circle((cx, cy), radius=self._brush_radius,
                                         fill=False, edgecolor=edge_color,
                                         linewidth=1.6, animated=True)
            self._brush_cursor.set_visible(visible)
            self.ax.add_patch(self._brush_cursor)

        # 6) Restore zoom/pan
        if saved_xlim is not None:
            self.ax.set_xlim(saved_xlim)
            self.ax.set_ylim(saved_ylim)

        # 7) Mini-map (only if zoomed)
        if self._zoom_active:
            self._draw_minimap()

        # 8) Help overlay
        if self._show_help:
            self._draw_help_overlay()

        # 9) Side panel + status bar
        self._draw_info_panel()
        self._update_status_bar()

        self.fig.canvas.draw_idle()

    def _draw_main_overlay(self, s: ImageState) -> None:
        if self._overlay_mode == "off":
            return
        if self._overlay_mode == "outline":
            if not s.current_mask.any():
                return
            try:
                import cv2
                mu = s.current_mask.astype(np.uint8)
                contours, _ = cv2.findContours(mu, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    pts = cnt.squeeze(1) if cnt.ndim == 3 else cnt
                    if pts.ndim == 2 and pts.shape[0] >= 2:
                        self.ax.plot(pts[:, 0], pts[:, 1], "-",
                                      color="red", linewidth=1.5)
            except Exception:
                # Fallback: render filled
                ov = np.zeros((*s.current_mask.shape, 4))
                ov[s.current_mask] = [1.0, 0.0, 0.0, self._mask_alpha]
                self._mask_overlay_artist = self.ax.imshow(ov, interpolation="nearest")
            return
        # filled mode — pre-composite source + mask in numpy into an RGB
        # buffer. matplotlib then displays a single RGB image with no alpha
        # compositing, which fixes the "lighter brush" alpha-mismatch bug
        # (matplotlib's RGBA blit was double-applying alpha on the cached
        # background's mask region) AND is faster (no per-pixel alpha math
        # in the renderer).
        if self._composed_buf is None or self._composed_buf.shape[:2] != (s.H, s.W):
            self._composed_buf = s.image_np.copy()
        else:
            np.copyto(self._composed_buf, s.image_np)
        if s.current_mask.any():
            self._blend_mask_into_composed(
                s.image_np, s.current_mask, self._composed_buf, bbox=None,
            )
        # animated=True: matplotlib excludes from the cached background;
        # blit fast-path draws via draw_artist explicitly.
        self._mask_overlay_artist = self.ax.imshow(
            self._composed_buf, animated=True, interpolation="nearest",
        )

    def _draw_diff_overlay(self, s: ImageState) -> None:
        added = s.current_mask & ~s.original_mask
        removed = ~s.current_mask & s.original_mask
        kept = s.current_mask & s.original_mask
        ov = np.zeros((s.H, s.W, 4))
        if kept.any():
            ov[kept] = [1.0, 1.0, 0.0, self._mask_alpha * 0.7]   # yellow
        if added.any():
            ov[added] = [0.0, 1.0, 0.0, self._mask_alpha]         # green
        if removed.any():
            ov[removed] = [1.0, 0.0, 0.0, self._mask_alpha]       # red
        self.ax.imshow(ov)

    def _draw_minimap(self) -> None:
        if self.state is None:
            return
        s = self.state
        # Inset axes in top-right of main axes
        self.ax_minimap = self.ax.inset_axes(
            [0.78, 0.74, 0.20, 0.24], transform=self.ax.transAxes,
        )
        self.ax_minimap.imshow(s.image_np)
        if s.current_mask.any():
            ov = np.zeros((*s.current_mask.shape, 4))
            ov[s.current_mask] = [1.0, 0.0, 0.0, 0.45]
            self.ax_minimap.imshow(ov)
        self.ax_minimap.set_xticks([])
        self.ax_minimap.set_yticks([])
        for sp in self.ax_minimap.spines.values():
            sp.set_edgecolor("white")
            sp.set_linewidth(1.2)
        self._draw_minimap_viewport()

    def _draw_minimap_viewport(self) -> None:
        if self.ax_minimap is None or self.state is None:
            return
        from matplotlib.patches import Rectangle
        # Clear previous viewport rectangles
        for art in list(self.ax_minimap.patches):
            try:
                art.remove()
            except Exception:
                pass
        if not self._zoom_active:
            return
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x0 = min(xlim)
        x1 = max(xlim)
        y0 = min(ylim)
        y1 = max(ylim)
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                         fill=False, edgecolor="#00ff66", linewidth=2.0)
        self.ax_minimap.add_patch(rect)

    def _draw_info_panel(self) -> None:
        if self._info_text_artist is None or self.state is None:
            return
        s = self.state

        lines = []
        lines.append(f"IMAGE  {s.image_id[:14]}")
        lines.append(f"file   {s.image_path.name}")
        if s.manifest_meta:
            src = s.manifest_meta.get("subcategory") or s.manifest_meta.get("source") or "--"
            lines.append(f"src    {str(src)[:24]}")
        lines.append(f"size   {s.W} x {s.H}")
        # Three sources of truth, shown side-by-side so it's obvious which
        # disagrees:
        #   label  = scraper/manifest intent ('this image was curated as pos/neg')
        #   auto   = what the auto-pipeline actually detected (fence_wood>0)
        #   manual = your decision after editing
        # A 'pos' label on an obviously fence-less image just means the
        # scraper mislabeled it -- your manual judgement is the corrected truth.
        auto_cls = "?"
        if s.auto_meta:
            auto_cls = "pos" if s.auto_meta.get("fence_wood_coverage", 0) > 0 else "neg"
        lines.append(f"label  {s.original_class or '?'}    "
                      f"auto {auto_cls}    manual {s.manual_class}")
        if s.auto_meta:
            am = s.auto_meta
            lines.append("")
            lines.append("--- AUTO PIPELINE ---")
            lines.append(f"conf   {am.get('overall_confidence', 0):.2f}")
            lines.append(f"fence  {am.get('fence_wood_coverage', 0):.1%}")
            lines.append(f"detect {am.get('n_detections', 0)} boxes")
            flags = am.get("flags", []) or []
            if flags:
                lines.append(f"flags  {','.join(flags)[:24]}")
            if am.get("needs_review"):
                lines.append("       NEEDS REVIEW")
            if s.image_id in self.qa_queue_ids:
                lines.append("       IN QA QUEUE")
        lines.append("")
        lines.append("--- THIS EDIT ---")
        lines.append(f"clicks +{sum(1 for _,_,l in s.points if l==1)} "
                      f"-{sum(1 for _,_,l in s.points if l==0)}")
        lines.append(f"fence  {s.fence_coverage:.1%}")
        lines.append(f"dirty  {'YES' if s.dirty else 'no'}")
        if s.dirty:
            edit_dist = int((s.original_mask != s.current_mask).sum())
            lines.append(f"delta  {edit_dist:,} px")
        lines.append("")
        lines.append("--- SESSION ---")
        elapsed = time.time() - self._session_start
        n = len(self._image_durations)
        avg = (sum(self._image_durations) / n) if n else 0.0
        lines.append(f"saved  {n}")
        if avg > 0:
            lines.append(f"avg    {avg:.1f} s/img")
        lines.append(f"time   {self._fmt_duration(elapsed)}")
        if self._save_errors:
            lines.append("")
            lines.append(f"!! {len(self._save_errors)} SAVE ERR")

        text = "\n".join(lines)
        self._info_text_artist.set_text(text)

    def _update_status_bar(self) -> None:
        if self._status_text_artist is None:
            return

        parts = []
        # Mode tag
        if self._box_armed:
            parts.append("MODE=BOX-ARMED")
        elif self._mode == "brush":
            parts.append(f"MODE=BRUSH r={self._brush_radius}")
        else:
            parts.append("MODE=SAM")
        # Overlay
        parts.append(f"overlay={self._overlay_mode}")
        if self._show_diff:
            parts.append("DIFF")
        if self._show_original_overlay:
            parts.append("AUTO-OVL")
        parts.append(f"alpha={self._mask_alpha:.2f}")
        # Zoom
        if self._zoom_active and self.state is not None:
            xlim = self.ax.get_xlim()
            visible_w = abs(xlim[1] - xlim[0])
            zoom_pct = 100.0 * self.state.W / max(visible_w, 1.0)
            parts.append(f"zoom={zoom_pct:.0f}%")
        else:
            parts.append("zoom=fit")
        # Cursor
        if self._last_motion_xy is not None and self.state is not None:
            cx, cy = self._last_motion_xy
            parts.append(f"xy=({int(cx)},{int(cy)})")
        # Cycle
        if self._last_click_candidates is not None:
            n = len(self._last_click_candidates[0])
            parts.append(f"cand={self._last_candidate_idx+1}/{n}")
        # Errors
        if self._save_errors:
            parts.append(f"!!SAVE-ERRORS={len(self._save_errors)}")
        # Per-image elapsed
        if self.state is not None:
            elapsed = time.time() - self.state.load_started_at
            parts.append(f"this={elapsed:.0f}s")
        # Prefetch hint
        if self.prefetcher._enabled:
            parts.append(f"prefetch={self.prefetcher.max_ahead}")

        self._status_text_artist.set_text("  |  ".join(parts))

    def _draw_help_overlay(self) -> None:
        help_text = (
            "MOUSE:\n"
            "  L-click          SAM add region (cumulative)\n"
            "  R-click          SAM remove region\n"
            "  X + L-drag       Box prompt (fast for full fences)\n"
            "  Middle-drag      Pan view\n"
            "  Scroll           Zoom at cursor\n"
            "  Brush mode (B):  L/R-drag = paint/erase pixels\n\n"
            "KEYBOARD:\n"
            "  SPACE   save (with edge refinement) + advance\n"
            "  A       ACCEPT auto-mask as-is, save + advance\n"
            "  -> / N  next (auto-save if dirty)\n"
            "  <-      previous (auto-save if dirty)\n"
            "  R      clear mask;  Shift+R  restore auto\n"
            "  U      undo last edit\n"
            "  C      cycle SAM's 3 candidate masks\n"
            "  V      yellow overlay of original auto-mask\n"
            "  D      diff vs auto (green=add red=rm yellow=kept)\n"
            "  E      cycle overlay (filled/outline/off)\n"
            "  , .    decrease/increase mask opacity\n"
            "  F      fill small holes\n"
            "  K      kill specks (<= --speck-threshold px)\n"
            "  B      toggle SAM <-> BRUSH mode\n"
            "  [ ]    brush radius\n"
            "  Z      reset zoom to fit\n"
            "  X      arm box-prompt; release without dragging to disarm\n"
            "  H      toggle this help\n"
            "  Q      quit (Q twice if dirty)\n\n"
            "TIPS:\n"
            "  - Use X + drag for full fences (1 prompt vs many clicks)\n"
            "  - A is the fastest path for already-correct auto-masks\n"
            "  - F + K together clean up SAM artifacts in 2 keystrokes\n"
            "  - D mode shows exactly what your edits did vs auto\n"
            "  - Mini-map (top-right when zoomed) keeps you oriented"
        )
        self._help_text_artist = self.ax.text(
            0.02, 0.98, help_text,
            transform=self.ax.transAxes, fontsize=9, family="monospace",
            color="white", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="black", alpha=0.88),
        )
        # animated=True so blit redraws it on top of the composed image
        self._help_text_artist.set_animated(True)

    # ── save ──────────────────────────────────────────────────────────────

    def _save_current(self, skip_edge_refinement: bool = False,
                      accept_mode: bool = False) -> None:
        if self.state is None:
            return
        s = self.state
        # Don't write a duplicate row if the image was already reviewed in a
        # previous session AND nothing has changed since. This is the fix for
        # the "go back to verify -> press SPACE/A again" pattern producing
        # multiple identical rows in manual_review.jsonl.
        if (not s.dirty) and s.image_id in self.reviewed_ids:
            print(f"  [skip-resave] {s.image_id[:12]} already reviewed, "
                  f"no changes since")
            return
        try:
            if (not skip_edge_refinement) and s.dirty and s.current_mask.any():
                refined, stages = self.edge_refiner.refine(s.current_mask, s.image_np)
                print(f"  [save] edge refinement: "
                      f"{' -> '.join(stages) if stages else '(none)'}")
            else:
                refined = s.current_mask
                stages = []

            class_map = refined.astype(np.uint8)
            _atomic_save_png(Image.fromarray(class_map, mode="L"), s.mask_path)
            preview = (class_map * 255).astype(np.uint8)
            _atomic_save_png(Image.fromarray(preview, mode="L"),
                             self.preview_dir / f"{s.image_id}.png")
            viz = s.image_np.copy()
            if refined.any():
                red = np.array([255, 0, 0], dtype=np.float32)
                viz[refined] = (viz[refined] * 0.45 + red * 0.55).astype(np.uint8)
            _atomic_save_png(Image.fromarray(viz),
                             self.viz_dir / f"{s.image_id}.png")

            manual_class = "pos" if refined.any() else "neg"
            class_changed = (s.original_class is not None and
                              s.original_class != manual_class)
            duration_s = time.time() - s.load_started_at
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
                "accept_mode": accept_mode,
                "review_duration_s": float(duration_s),
            }
            with self.review_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            self.reviewed_ids.add(s.image_id)
            self.last_reviewed = s.image_id
            self._image_durations.append(duration_s)
            s.original_mask = refined.copy()
            s.current_mask = refined.copy()
            s.edit_history.clear()
            s.points.clear()
            self._last_click_candidates = None
            self._quit_warned = False

            self._prune_qa_queue(s.image_id)
            self._delete_stale_heatmap(s.image_id)

            self._saves_since_backup += 1
            if self._saves_since_backup >= self._backup_every_n:
                self._backup_review_log()
                self._saves_since_backup = 0

            flip_msg = ""
            if class_changed:
                flip_msg = f"  CLASS FLIP: {s.original_class} -> {manual_class}"
            mode_tag = " (accept)" if accept_mode else ""
            print(f"  [saved{mode_tag}] {s.image_id}  fence={refined.mean():.1%}  "
                  f"took={duration_s:.1f}s{flip_msg}")

        except (OSError, IOError, PermissionError) as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"
            print(f"  [SAVE FAILED] {err}")
            self._save_errors.append(f"{s.image_id}: {err}")
            self._redraw()

    # ── auxiliary cleanup on save ────────────────────────────────────────

    def _prune_qa_queue(self, image_id: str) -> None:
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
            self.qa_queue_ids.add(image_id)
            print(f"  [qa-queue] WARN: prune failed: {type(e).__name__}: {str(e)[:80]}")

    def _delete_stale_heatmap(self, image_id: str) -> None:
        hp = self.heatmaps_dir / f"{image_id}.png"
        try:
            if hp.exists():
                hp.unlink()
                print(f"  [heatmap] deleted stale {hp.name}")
        except (OSError, IOError, PermissionError) as e:
            print(f"  [heatmap] WARN: delete failed: {type(e).__name__}: {str(e)[:80]}")

    def _backup_review_log(self) -> None:
        if not self.review_log_path.exists():
            return
        try:
            tmp = self.review_backup_path.with_suffix(
                self.review_backup_path.suffix + ".tmp")
            shutil.copy2(self.review_log_path, tmp)
            os.replace(tmp, self.review_backup_path)
            sz = self.review_backup_path.stat().st_size
            print(f"  [backup] manual_review.jsonl.bak written ({sz/1024:.1f} KB)")
        except (OSError, IOError, PermissionError) as e:
            print(f"  [backup] WARN: failed: {type(e).__name__}: {str(e)[:80]}")

    # ── session-end summary popup ────────────────────────────────────────

    def _show_session_summary(self) -> None:
        elapsed = time.time() - self._session_start
        n = len(self._image_durations)
        n_reviewed_in_nav = sum(1 for r in self.nav_order
                                 if r["id"] in self.reviewed_ids)
        avg = (sum(self._image_durations) / n) if n else 0.0
        flip_count = 0
        if self.review_log_path.exists():
            recent = load_jsonl_safe(self.review_log_path)[-n:] if n else []
            flip_count = sum(1 for r in recent if r.get("class_changed"))
        # Console summary (always)
        print()
        print("=" * 60)
        print("Session summary")
        print("=" * 60)
        print(f"  reviewed (total):      {len(self.reviewed_ids):,}")
        print(f"  reviewed (this nav):   {n_reviewed_in_nav}/{len(self.nav_order)}")
        print(f"  remaining in nav:      {len(self.nav_order) - n_reviewed_in_nav}")
        print(f"  this session:          {n} saved")
        if avg > 0:
            print(f"  avg time/image:        {avg:.1f} s")
        print(f"  session elapsed:       {self._fmt_duration(elapsed)}")
        print(f"  class flips this sess: {flip_count}")
        if self._save_errors:
            print(f"  SAVE ERRORS:           {len(self._save_errors)}")
            for e in self._save_errors[:5]:
                print(f"    - {e}")
        print(f"  review log:            {self.review_log_path}")
        print("Done.")

        # Modal popup (best-effort; matplotlib already closed by now, so
        # use tkinter for an OS-native message box).
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            msg = (
                f"Session finished\n\n"
                f"Saved this session:    {n}\n"
                f"Reviewed (total):      {len(self.reviewed_ids):,}\n"
                f"Remaining in nav:      {len(self.nav_order) - n_reviewed_in_nav}\n"
                f"Class flips:           {flip_count}\n"
                f"Average per image:     {avg:.1f} s\n"
                f"Total elapsed:         {self._fmt_duration(elapsed)}\n"
            )
            if self._save_errors:
                msg += f"\n!! {len(self._save_errors)} SAVE ERRORS -- see console"
            messagebox.showinfo("Manual refinement -- session summary", msg)
            root.destroy()
        except Exception:
            # If tkinter unavailable, console summary above is sufficient
            pass

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        s = int(seconds)
        h, rem = divmod(s, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h {m:02d}m {s:02d}s"
        if m:
            return f"{m}m {s:02d}s"
        return f"{s}s"


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
                    help="Disable DenseCRF stage of edge refinement.")
    ap.add_argument("--only-unreviewed", action="store_true",
                    help="Skip already-reviewed images (default navigates all).")
    ap.add_argument("--only-class", choices=["pos", "neg"],
                    help="Only navigate manifest rows of this class.")
    ap.add_argument("--start-at", type=str, default=None,
                    help="Image ID prefix to jump to as starting point.")
    ap.add_argument("--order", choices=["manifest", "qa-first", "coverage", "random"],
                    default="manifest",
                    help="Navigation order. qa-first: QA-flagged + needs_review + low confidence "
                         "first (recommended for triage). coverage: lowest fence_wood_coverage "
                         "first (likely-incomplete masks first). random: shuffled (use --seed).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed (used with --order random; default 42).")
    ap.add_argument("--prefetch", type=int, default=2,
                    help="Number of images to prefetch ahead (background image load + "
                         "SAM encode). Default 2. Set 0 to disable.")
    ap.add_argument("--no-prefetch", action="store_true",
                    help="Disable background prefetch entirely.")
    ap.add_argument("--speck-threshold", type=int, default=50,
                    help="K key removes connected components smaller than this "
                         "many pixels. Default 50.")
    ap.add_argument("--backup-every", type=int, default=25,
                    help="Write manual_review.jsonl.bak every N saves. Default 25.")
    args = ap.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")
    if not args.annotations_root.exists():
        raise SystemExit(f"Annotations root not found: {args.annotations_root}")

    app = ManualRefinementApp(args)
    app.run()


if __name__ == "__main__":
    main()
