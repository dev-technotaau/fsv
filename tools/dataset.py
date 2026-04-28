"""tools/dataset.py — Production dataset module for two-phase wood-fence
segmentation training.

Wraps the JSONL-based split files produced by split_dataset.py +
build_mask_splits.py into a PyTorch Dataset with augmentation pipelines
tuned for fence segmentation. Designed for two-phase training:

    Phase 1 (512x512, full dataset)        --> stronger augmentation
    Phase 2 (1024x1024, HQ subset, FT)     --> gentler augmentation

QUICK USAGE
-----------
    from torch.utils.data import DataLoader
    from tools.dataset import (
        make_phase1_train_dataset, make_phase1_val_dataset,
        make_phase2_train_dataset, make_phase2_val_dataset,
        verify_split_integrity, compute_pos_weight,
    )

    # Sanity check ONCE at startup
    verify_split_integrity()

    # Phase 1
    train_ds = make_phase1_train_dataset()
    val_ds   = make_phase1_val_dataset()
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True,
                           num_workers=4, pin_memory=True)

    # In your training loop
    for batch in train_dl:
        x = batch["image"].cuda()           # (B, 3, 512, 512) float
        y = batch["mask"].cuda()            # (B, 512, 512) long, values 0/1
        w = batch["sample_weight"].cuda()   # (B,) float, optional loss weighting
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float(),
                                                    weight=w[:, None, None, None])

DEPENDENCIES
------------
    pip install albumentations torch torchvision pillow numpy

DESIGN NOTES
------------
- One FenceDataset class supports any phase via the `transform` argument.
- Four pre-built augmentation pipelines (phase1/2 x train/val).
- Mask values are uint8 0/1 on disk; returned as Long tensor (H, W).
- Optional `weight_by_review_source` weights samples by audit provenance:
    manual = 1.0, auto_accept = 0.85, auto_clear = 0.95, unreviewed = 0.5.
  Encourages the model to pay more attention to human-validated masks.
- All file paths are read from JSONLs — no path constants in this module.
- Smoke-test by running `python tools/dataset.py` — verifies splits + loads
  one sample from each phase + computes class imbalance.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError as e:
    raise ImportError(
        "albumentations is required for this module. Install with:\n"
        "    pip install albumentations\n"
        "Original error: " + str(e)
    ) from e


# ══════════════════════════════════════════════════════════════════════
# Paths and constants
# ══════════════════════════════════════════════════════════════════════

DEFAULT_SPLITS_DIR = Path("dataset/splits")

# ImageNet stats — standard for transfer learning from ImageNet-pretrained
# backbones (UNet, DeepLabV3, SegFormer, Mask2Former, etc.)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Per-sample weights by review provenance (used when
# weight_by_review_source=True). Tuned so manual reviews carry the most
# weight; auto-validated less; never-reviewed least.
REVIEW_SOURCE_WEIGHTS: dict[str, float] = {
    "manual":               1.00,
    "auto_accept_positive": 0.85,
    "auto_negative_clear":  0.95,   # neg masks are easier (just zero out)
    "unreviewed":           0.50,
}


# ══════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════

def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ══════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════

class FenceDataset(Dataset):
    """PyTorch Dataset over a fence-segmentation split.

    Reads two parallel JSONLs:
      - image split (e.g., dataset/splits/train.jsonl)
        — each row has id, path, class, subcategory, etc.
      - mask split  (e.g., dataset/splits/train_masks.jsonl)
        — each row has id, mask_path, review_source, fence_pixel_count, ...

    Joins on `id`. Applies an albumentations transform that handles both
    image and mask consistently (geometric ops keep them aligned).

    Args:
        img_jsonl:               Path to image split file.
        mask_jsonl:              Path to mask split file (parallel rows).
        transform:               Albumentations Compose pipeline.
        weight_by_review_source: If True, returns sample_weight per item based
                                  on REVIEW_SOURCE_WEIGHTS. Use to down-weight
                                  auto-processed samples in your loss.
        custom_weights:          Override REVIEW_SOURCE_WEIGHTS dict.

    Returns dict per sample:
        image:         (3, H, W) FloatTensor (after Normalize + ToTensorV2)
        mask:          (H, W) LongTensor with values 0 or 1
        sample_weight: scalar FloatTensor (1.0 if weight_by_review_source=False)
        metadata:      dict with id, class, subcategory, review_source, ...
    """

    def __init__(
        self,
        img_jsonl: str | Path,
        mask_jsonl: str | Path,
        transform: Optional[Callable] = None,
        weight_by_review_source: bool = False,
        custom_weights: Optional[dict[str, float]] = None,
    ) -> None:
        self.img_jsonl = Path(img_jsonl)
        self.mask_jsonl = Path(mask_jsonl)
        self.img_rows = load_jsonl(self.img_jsonl)
        self.masks: dict[str, dict] = {r["id"]: r for r in load_jsonl(self.mask_jsonl)}

        # Validate every image has a mask row (fail fast)
        missing = [r["id"] for r in self.img_rows if r["id"] not in self.masks]
        if missing:
            raise ValueError(
                f"{len(missing)} images in {self.img_jsonl.name} have no mask "
                f"row in {self.mask_jsonl.name}; first 5: {missing[:5]}"
            )

        self.transform = transform
        self.weight_by_review_source = weight_by_review_source
        self.weights = (custom_weights or REVIEW_SOURCE_WEIGHTS)

    def __len__(self) -> int:
        return len(self.img_rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_row = self.img_rows[idx]
        m = self.masks[img_row["id"]]

        # PIL -> numpy uint8 RGB (albumentations expects HWC RGB).
        # Use context managers so file handles are released promptly even when
        # workers raise — avoids exhausting the OS handle quota over long runs.
        with Image.open(img_row["path"]) as _im:
            image = np.array(_im.convert("RGB"))
        with Image.open(m["mask_path"]) as _mk:
            mask = np.array(_mk)
        if mask.ndim == 3:
            mask = mask[..., 0]   # safety: any unexpected RGB mask -> grayscale
        # Normalize mask values to {0, 1}. PNG masks are commonly stored as
        # {0, 255} (PIL/OpenCV convention for visual masks); treating those as
        # raw class indices would silently break BCE/Dice (label 255 explodes
        # the loss) and metrics (`(targets == 1)` would be False everywhere).
        # `(mask > 0)` is the correct, source-agnostic normalization.
        mask = (mask > 0).astype(np.uint8)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        # Normalize mask dtype: long for CrossEntropy, float for BCE
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.long()

        # Per-sample loss weight from review provenance
        if self.weight_by_review_source:
            w = float(self.weights.get(m.get("review_source"), 1.0))
        else:
            w = 1.0

        return {
            "image": image,
            "mask": mask,
            "sample_weight": torch.tensor(w, dtype=torch.float32),
            "metadata": {
                "id": img_row["id"],
                "class": img_row.get("class"),
                "subcategory": img_row.get("subcategory"),
                "review_source": m.get("review_source"),
                "class_source": m.get("class_source"),
            },
        }


# ══════════════════════════════════════════════════════════════════════
# Augmentation pipelines
# ══════════════════════════════════════════════════════════════════════

def phase1_train_aug(image_size: int = 512) -> A.Compose:
    """Strong augmentation for Phase 1 training at 512x512.

    Targeted scenarios (per the project's deployment requirements):
      - Occlusion overlays (foliage, people, objects)  -> CoarseDropout (random color fill)
      - Lighting shifts (harsh sun, shadows)           -> RandomShadow, RandomSunFlare,
                                                           CLAHE, RandomToneCurve
      - Blur + noise (camera quality)                  -> Gaussian/Motion/Defocus blur,
                                                           GaussNoise, ImageCompression
      - Distance scaling (far vs close fences)         -> OneOf{ZoomIn (RandomResizedCrop),
                                                           ShrinkAndPad (far-fence)}
      - Color similarity (wood vs soil)                -> aggressive HSV + ColorJitter +
                                                           ChannelShuffle (low p) +
                                                           CLAHE for de-color-reliance
    """
    # Sub-pipelines for the OneOf "distance scaling" choice
    zoom_in = A.RandomResizedCrop(size=(image_size, image_size),
                                    scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1.0)
    # "Far fence": shrink the image to a fraction of the target size, pad the
    # rest with reflect-padding (so the background looks natural, not black).
    # Net effect: fence appears smaller in the frame, like it would from far away.
    shrink_and_pad = A.Compose([
        A.LongestMaxSize(max_size=image_size, p=1.0),
        A.RandomScale(scale_limit=(-0.5, -0.1), p=1.0),     # shrink 10-50%
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                       border_mode=2, p=1.0),               # 2 = REFLECT_101
        A.CenterCrop(image_size, image_size, p=1.0),
    ])
    # NEW: boundary-focused crop — picks a crop centered on a fence-edge
    # pixel (Sobel-derived from the GT mask). Forces the model to see
    # boundary cases (fence vs sky/ground/grass/wood-wall/etc) most of
    # the time. Falls back to uniform if no fence in the image.
    boundary_focused = A.Compose([
        # Make sure the image is at least 1.25x crop_size before boundary crop.
        A.LongestMaxSize(max_size=int(image_size * 1.5), p=1.0),
        A.PadIfNeeded(min_height=int(image_size * 1.1),
                       min_width=int(image_size * 1.1),
                       border_mode=2, p=1.0),
        BoundaryAwareCrop(image_size, image_size, boundary_p=0.85, p=1.0),
    ])

    return A.Compose([
        # ── DISTANCE SCALING (far vs close fence + BOUNDARY FOCUS) ───
        A.OneOf([
            zoom_in,            # close-up: zoom into a region of the image
            shrink_and_pad,     # far-away: shrink fence + reflect-pad around it
            boundary_focused,   # fence-edge-focused crop (NEW)
        ], p=1.0),

        # ── GEOMETRIC (applied to both image and mask) ───────────────
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.6, border_mode=0),                  # was limit=15 p=0.5
        A.Perspective(scale=(0.02, 0.07), p=0.4),                  # was 0.02-0.05 p=0.3
        # Subtle non-rigid warps — both image AND mask get the SAME displacement
        # field from albumentations, so geometry stays consistent. ElasticTransform
        # simulates camera lens distortion / wind-bent fence panels; GridDistortion
        # simulates uneven perspectives. Mild settings — fences don't bend much.
        A.OneOf([
            A.ElasticTransform(alpha=30, sigma=5, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),
        ], p=0.2),

        # ── LIGHTING (harsh sun, shadows, exposure) ──────────────────
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1),
                        num_shadows_limit=(1, 3), p=0.4),           # was (1, 2) p=0.3
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), src_radius=180, p=0.15),  # was r=160 p=0.1
        A.RandomToneCurve(scale=0.25, p=0.4),                       # was 0.2 p=0.3
        A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8), p=0.3),  # was p=0.2

        # ── COLOR (wood vs soil de-reliance) ─────────────────────────
        # Stronger color jitter so the model can't over-rely on a fixed brown.
        A.OneOf([
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5,
                           hue=0.18, p=1.0),                        # was 0.4/0.15
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=50,
                                  val_shift_limit=35, p=1.0),       # was 25/40/30
            A.RandomBrightnessContrast(brightness_limit=0.5,
                                         contrast_limit=0.5, p=1.0),  # was 0.4
        ], p=0.85),                                                  # was 0.8
        # ChannelShuffle: rare, swaps RGB channels — extreme color robustness.
        A.ChannelShuffle(p=0.05),

        # ── BLUR + NOISE ─────────────────────────────────────────────
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=9, p=1.0),                       # was limit=7
            A.Defocus(radius=(1, 4), p=1.0),                          # was (1, 3)
        ], p=0.3),                                                   # was 0.2
        A.ImageCompression(quality_range=(50, 100), p=0.4),          # was (60, 100) p=0.3
        A.GaussNoise(std_range=(0.04, 0.25), p=0.3),                 # was 0.20 p=0.2
        A.Downscale(scale_range=(0.4, 0.9), p=0.3),                  # was 0.5 p=0.2

        # ── OCCLUSION OVERLAYS ───────────────────────────────────────
        # 1) Photo-realistic copy-paste — bumped occluder count + probability.
        CopyPasteOccluder(
            occluder_dir=DEFAULT_OCCLUDER_DIR,
            max_per_image=3,                                         # was 2
            scale_range=(0.10, 0.45),                                # was (0.10, 0.40)
            rotation_range=(-25.0, 25.0),                            # was (-20, 20)
            flip_p=0.5, color_jitter=True, cache_in_memory=True,
            p=0.5,                                                   # was 0.4
        ),
        # 1b) Hard-negative wooden non-fence paste — pastes wood into
        #     BACKGROUND only, mask UNCHANGED. Forces the model to learn
        #     "wood texture alone != fence". Critical for scenes with
        #     wooden garden beds, planters, decks. Build the pool with:
        #     `python -m tools.build_occluder_pool --wooden-negatives`
        HardNegativeWoodPaste(
            wood_dir=DEFAULT_HARD_NEG_DIR,
            max_per_image=2, scale_range=(0.10, 0.35),
            flip_p=0.5, max_overlap_with_fence=0.05,
            cache_in_memory=True, p=0.3,
        ),
        # 2) Generic occlusion fallback: random-color holes.
        A.CoarseDropout(
            num_holes_range=(1, 6),                                  # was (1, 5)
            hole_height_range=(image_size // 16, image_size // 5),   # was //6
            hole_width_range=(image_size // 16, image_size // 5),    # was //6
            fill="random",
            fill_mask=0,
            p=0.4,                                                   # was 0.3
        ),

        # ── FINAL: normalize + tensor ────────────────────────────────
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ══════════════════════════════════════════════════════════════════════
# Optional Copy-Paste occluder augmentation (needs occluder image pool)
# ══════════════════════════════════════════════════════════════════════

DEFAULT_OCCLUDER_DIR = Path("dataset/occluders")
DEFAULT_HARD_NEG_DIR = Path("dataset/hard_negatives")


# ══════════════════════════════════════════════════════════════════════
# Hard-negative wood paste (fence-domain enhancement)
# ══════════════════════════════════════════════════════════════════════

class HardNegativeWoodPaste(A.DualTransform):
    """Paste a wooden non-fence cutout (plank / garden bed / deck patch) into
    the BACKGROUND of the training image.

    The mask STAYS unchanged: the pasted wood is supervised as NOT-fence.
    This explicitly teaches the model that wood texture alone is insufficient
    to call something a fence — it needs scene context + shape. Critical for
    real fence images that share scenes with garden beds, planters, decks,
    and similar wooden non-fence structures.

    Pool: built by `python -m tools.build_occluder_pool --wooden-negatives`
    Default location: `dataset/hard_negatives/wood/*.png`

    Args:
        wood_dir:       Recursive *.png glob root for wooden non-fence cutouts.
        max_per_image:  Up to N wood objects pasted per image.
        scale_range:    Object height as fraction of image height.
        flip_p:         Per-object horizontal flip probability.
        max_overlap_with_fence: If a candidate paste region overlaps the
                                fence mask by more than this fraction, that
                                placement is rejected and we try again
                                (max 8 attempts per object).
        cache_in_memory: Preload all wood PNGs into RAM.
        p:              Per-image apply probability.
    """

    def __init__(self,
                 wood_dir: str | Path = DEFAULT_HARD_NEG_DIR,
                 max_per_image: int = 2,
                 scale_range: tuple[float, float] = (0.10, 0.35),
                 flip_p: float = 0.5,
                 max_overlap_with_fence: float = 0.05,
                 cache_in_memory: bool = True,
                 p: float = 0.3) -> None:
        super().__init__(p=p)
        self.wood_dir = Path(wood_dir)
        self.max_per_image = int(max_per_image)
        self.scale_range = scale_range
        self.flip_p = float(flip_p)
        self.max_overlap_with_fence = float(max_overlap_with_fence)
        self.cache_in_memory = cache_in_memory

        self._paths: list[Path] = []
        self._cached: list[np.ndarray] = []
        if self.wood_dir.exists():
            self._paths = sorted(self.wood_dir.rglob("*.png"))
            if cache_in_memory:
                for p_ in self._paths:
                    try:
                        arr = np.array(Image.open(p_).convert("RGBA"))
                        if arr.shape[-1] == 4 and arr[..., 3].sum() > 0:
                            self._cached.append(arr)
                    except Exception:
                        continue
        if not self._paths:
            import warnings
            warnings.warn(
                f"HardNegativeWoodPaste: no PNGs found in '{self.wood_dir}'. "
                f"Build the pool with `python -m tools.build_occluder_pool "
                f"--wooden-negatives`. Transform will be a NO-OP.",
                RuntimeWarning, stacklevel=2,
            )

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("wood_dir", "max_per_image", "scale_range", "flip_p",
                "max_overlap_with_fence", "cache_in_memory")

    def _get_one(self, rng: np.random.Generator) -> np.ndarray | None:
        pool = self._cached if self.cache_in_memory else self._paths
        if not pool:
            return None
        idx = int(rng.integers(0, len(pool)))
        if self.cache_in_memory:
            return pool[idx]
        try:
            return np.array(Image.open(pool[idx]).convert("RGBA"))
        except Exception:
            return None

    def get_params_dependent_on_data(self, params: dict, data: dict) -> dict:
        """Pre-compute paste plan from the (image, mask) pair, so that
        apply() and apply_to_mask() see the same plan."""
        img = data["image"]
        mask = data.get("mask")
        H, W = img.shape[:2]
        if mask is None:
            mask = np.zeros((H, W), dtype=np.uint8)
        seed = int(np.random.randint(0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, self.max_per_image + 1))

        plan: list[dict] = []
        m_bin = (np.asarray(mask) > 0).astype(np.uint8)

        for _ in range(n):
            occ = self._get_one(rng)
            if occ is None or occ.size == 0:
                continue
            if self.flip_p > 0 and rng.random() < self.flip_p:
                occ = occ[:, ::-1, :].copy()
            scale = float(rng.uniform(*self.scale_range))
            target_h = max(8, int(H * scale))
            target_w = max(8, int(occ.shape[1] * (target_h / occ.shape[0])))
            if target_w >= W:
                target_w = W - 1
                target_h = max(8, int(occ.shape[0] * (target_w / occ.shape[1])))
            try:
                occ_resized = np.array(
                    Image.fromarray(occ).resize((target_w, target_h),
                                                  resample=Image.BILINEAR)
                )
            except Exception:
                continue

            # Try up to 8 random placements that don't heavily overlap fence
            placed = False
            for _ in range(8):
                y0 = int(rng.integers(0, max(1, H - target_h + 1)))
                x0 = int(rng.integers(0, max(1, W - target_w + 1)))
                fence_overlap = float(
                    m_bin[y0:y0 + target_h, x0:x0 + target_w].mean()
                )
                if fence_overlap <= self.max_overlap_with_fence:
                    plan.append({
                        "occ": occ_resized,
                        "y0": y0, "x0": x0,
                        "h": target_h, "w": target_w,
                    })
                    placed = True
                    break
            # If no clear background patch found, just skip this occluder
            if not placed:
                continue

        return {"plan": plan}

    def apply(self, image: np.ndarray, plan: Optional[list] = None,
              **params) -> np.ndarray:
        if not plan:
            return image
        out = image.copy()
        for item in plan:
            occ = item["occ"]
            y0, x0, h, w = item["y0"], item["x0"], item["h"], item["w"]
            occ_rgb = occ[..., :3].astype(np.float32)
            alpha = (occ[..., 3:4].astype(np.float32) / 255.0)
            patch = out[y0:y0 + h, x0:x0 + w].astype(np.float32)
            out[y0:y0 + h, x0:x0 + w] = (
                patch * (1.0 - alpha) + occ_rgb * alpha
            ).astype(np.uint8)
        return out

    def apply_to_mask(self, mask: np.ndarray, plan: Optional[list] = None,
                       **params) -> np.ndarray:
        # Mask UNCHANGED — wood pasted into background stays not-fence.
        return mask


# ══════════════════════════════════════════════════════════════════════
# Boundary-aware random crop (fence-edge-focused training crops)
# ══════════════════════════════════════════════════════════════════════

class BoundaryAwareCrop(A.DualTransform):
    """Random crop biased toward fence boundaries.

    With probability `boundary_p`, the crop is centered on a randomly-picked
    fence-boundary pixel (Sobel-derived from the mask). Otherwise the crop
    is uniform random.

    Why this matters for fences specifically:
      - Plain RandomCrop is uniform — it will often crop into pure
        background (sky, road) or pure fence interior. Both are "easy"
        and waste the gradient budget.
      - Boundaries are where the model actually LEARNS — fence vs grass,
        fence vs sky, fence vs ground, fence vs occluder. Biasing crops to
        contain these boundaries dramatically increases per-step signal
        for the hardest part of the segmentation problem.

    Falls back to uniform random if:
      - The mask has no fence pixels (negative image)
      - The mask is fully covered (impossible in practice but handled)
      - The image is smaller than the requested crop (use PadIfNeeded first)
    """

    def __init__(self, height: int, width: int,
                 boundary_p: float = 0.7, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.height = int(height)
        self.width = int(width)
        self.boundary_p = float(boundary_p)

    def _uniform_topleft(self, H: int, W: int) -> tuple[int, int]:
        y0 = int(np.random.randint(0, max(1, H - self.height + 1)))
        x0 = int(np.random.randint(0, max(1, W - self.width + 1)))
        return y0, x0

    def get_params_dependent_on_data(self, params: dict, data: dict) -> dict:
        img = data["image"]
        H, W = img.shape[:2]
        # Image too small — fall back to (0,0) and rely on outer Resize/Pad
        if H < self.height or W < self.width:
            return {"y_min": 0, "x_min": 0}
        mask = data.get("mask")
        # No mask, or coin-flip says use uniform crop
        if (mask is None or mask.size == 0
                or float(np.random.random()) > self.boundary_p):
            y0, x0 = self._uniform_topleft(H, W)
            return {"y_min": y0, "x_min": x0}
        m = (np.asarray(mask) > 0).astype(np.uint8)
        if m.sum() == 0 or m.sum() == m.size:
            y0, x0 = self._uniform_topleft(H, W)
            return {"y_min": y0, "x_min": x0}
        # Cheap 4-neighbor edge detection (no scipy dep needed)
        edge = np.zeros_like(m, dtype=bool)
        edge[:-1] |= (m[:-1] != m[1:])
        edge[1:]  |= (m[1:]  != m[:-1])
        edge[:, :-1] |= (m[:, :-1] != m[:, 1:])
        edge[:, 1:]  |= (m[:, 1:]  != m[:, :-1])
        ys, xs = np.where(edge)
        if len(ys) == 0:
            y0, x0 = self._uniform_topleft(H, W)
            return {"y_min": y0, "x_min": x0}
        i = int(np.random.randint(0, len(ys)))
        cy, cx = int(ys[i]), int(xs[i])
        y0 = int(np.clip(cy - self.height // 2, 0, H - self.height))
        x0 = int(np.clip(cx - self.width // 2, 0, W - self.width))
        return {"y_min": y0, "x_min": x0}

    def apply(self, image: np.ndarray, y_min: int = 0, x_min: int = 0,
              **params) -> np.ndarray:
        return image[y_min:y_min + self.height, x_min:x_min + self.width]

    def apply_to_mask(self, mask: np.ndarray, y_min: int = 0, x_min: int = 0,
                       **params) -> np.ndarray:
        return mask[y_min:y_min + self.height, x_min:x_min + self.width]

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("height", "width", "boundary_p")


class CopyPasteOccluder(A.ImageOnlyTransform):
    """Paste a random RGBA occluder cutout (foliage, tree, person, vehicle,
    etc.) onto the image at a random position and scale. The mask is preserved
    intentionally — we treat the occluder as a "what the camera sees" overlay,
    so the underlying ground-truth fence is still fence-class. This teaches
    the model to predict fence even where it's partially blocked.

    Pool layout (recursive glob):
        dataset/occluders/foliage/leaf_001.png
        dataset/occluders/foliage/branch_002.png
        dataset/occluders/people/person_001.png
        dataset/occluders/...
    Subdirectories are treated as categories for organization but the
    transform samples uniformly across all PNGs.

    Build a pool with: `python -m tools.build_occluder_pool --procedural`
    (or `--from-images <dir>` to extract from real photos via rembg).

    Args:
      occluder_dir:        Directory to glob (recursive) for *.png files.
      max_per_image:       Up to N occluders pasted on each image.
      scale_range:         (min, max) occluder height as fraction of image height.
      rotation_range:      (min, max) degrees of random rotation per occluder.
      flip_p:              Per-occluder horizontal flip probability.
      color_jitter:        If True, randomly shift occluder RGB channels ±20%.
      cache_in_memory:     Preload all occluder PNGs into RAM (recommended:
                            saves ~5-15 ms per sample × N occluders).
      p:                   Per-image apply probability.
    """
    def __init__(self,
                 occluder_dir: str | Path = DEFAULT_OCCLUDER_DIR,
                 max_per_image: int = 2,
                 scale_range: tuple[float, float] = (0.1, 0.4),
                 rotation_range: tuple[float, float] = (-20.0, 20.0),
                 flip_p: float = 0.5,
                 color_jitter: bool = True,
                 cache_in_memory: bool = True,
                 p: float = 0.3) -> None:
        super().__init__(p=p)
        self.occluder_dir = Path(occluder_dir)
        self.max_per_image = max_per_image
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.flip_p = flip_p
        self.color_jitter = color_jitter
        self.cache_in_memory = cache_in_memory

        self._occluder_paths: list[Path] = []
        self._cached: list[np.ndarray] = []
        if self.occluder_dir.exists():
            self._occluder_paths = sorted(self.occluder_dir.rglob("*.png"))
            if cache_in_memory:
                for p_ in self._occluder_paths:
                    try:
                        arr = np.array(Image.open(p_).convert("RGBA"))
                        if arr.shape[-1] == 4 and arr[..., 3].sum() > 0:
                            self._cached.append(arr)
                    except Exception:
                        continue
        if not self._occluder_paths:
            import warnings
            warnings.warn(
                f"CopyPasteOccluder: no occluder PNGs found in "
                f"'{self.occluder_dir}'. This transform will be a NO-OP. "
                f"Build a pool with `python -m tools.build_occluder_pool --procedural`.",
                RuntimeWarning, stacklevel=2,
            )

    def __len__(self) -> int:
        return len(self._cached) if self.cache_in_memory else len(self._occluder_paths)

    # ── Internal helpers ─────────────────────────────────────────────

    def _get_occluder(self, rng: np.random.Generator) -> np.ndarray | None:
        if self.cache_in_memory:
            if not self._cached:
                return None
            return self._cached[int(rng.integers(0, len(self._cached)))]
        if not self._occluder_paths:
            return None
        path = self._occluder_paths[int(rng.integers(0, len(self._occluder_paths)))]
        try:
            return np.array(Image.open(path).convert("RGBA"))
        except Exception:
            return None

    def _augment_occluder(self, occ: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply random flip / rotation / color jitter to the occluder."""
        if self.flip_p > 0 and rng.random() < self.flip_p:
            occ = occ[:, ::-1, :].copy()
        if self.rotation_range != (0.0, 0.0):
            angle = float(rng.uniform(*self.rotation_range))
            if abs(angle) > 0.5:
                pil = Image.fromarray(occ).rotate(
                    angle, resample=Image.BILINEAR, expand=True,
                )
                occ = np.array(pil)
        if self.color_jitter:
            j = rng.uniform(0.8, 1.2, size=3).astype(np.float32)
            rgb = occ[..., :3].astype(np.float32) * j[None, None, :]
            occ = occ.copy()
            occ[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        return occ

    # ── Public API ───────────────────────────────────────────────────

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        # No-op if no occluders available
        if not (self._cached or self._occluder_paths):
            return image
        # Derive a deterministic per-call seed from numpy's global state so the
        # transform is reproducible when the upstream worker_init_fn seeds
        # numpy. (Avoids constructing a fresh non-seeded RNG which leaks
        # entropy from the OS and breaks reproducibility.)
        seed = int(np.random.randint(0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, self.max_per_image + 1))
        H, W = image.shape[:2]
        out = image.copy()
        for _ in range(n):
            occ = self._get_occluder(rng)
            if occ is None or occ.size == 0:
                continue
            occ = self._augment_occluder(occ, rng)
            if occ.shape[0] < 1 or occ.shape[1] < 1:
                continue

            # Scale the occluder relative to image height
            scale = float(rng.uniform(*self.scale_range))
            target_h = max(4, int(H * scale))
            target_w = max(4, int(occ.shape[1] * (target_h / occ.shape[0])))
            if target_w >= W:
                target_w = W - 1
                target_h = max(4, int(occ.shape[0] * (target_w / occ.shape[1])))
            try:
                occ_resized = np.array(
                    Image.fromarray(occ).resize((target_w, target_h),
                                                  resample=Image.BILINEAR)
                )
            except Exception:
                continue

            # Random position; allow partial off-frame pastes (clip to image bounds)
            y0 = int(rng.integers(-target_h // 4, max(1, H - target_h * 3 // 4)))
            x0 = int(rng.integers(-target_w // 4, max(1, W - target_w * 3 // 4)))
            sy0, sx0 = max(0, -y0), max(0, -x0)
            y0_c, x0_c = max(0, y0), max(0, x0)
            y1_c = min(H, y0 + target_h)
            x1_c = min(W, x0 + target_w)
            ph = y1_c - y0_c
            pw = x1_c - x0_c
            if ph <= 0 or pw <= 0:
                continue

            occ_rgb = occ_resized[sy0:sy0 + ph, sx0:sx0 + pw, :3].astype(np.float32)
            alpha = (occ_resized[sy0:sy0 + ph, sx0:sx0 + pw, 3:4].astype(np.float32) / 255.0)
            patch = out[y0_c:y1_c, x0_c:x1_c].astype(np.float32)
            out[y0_c:y1_c, x0_c:x1_c] = (
                patch * (1.0 - alpha) + occ_rgb * alpha
            ).astype(np.uint8)
        return out

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("occluder_dir", "max_per_image", "scale_range",
                "rotation_range", "flip_p", "color_jitter", "cache_in_memory")


def phase1_val_aug(image_size: int = 512) -> A.Compose:
    """Deterministic preprocessing for Phase 1 val/test at 512x512.
    No augmentation — only resize + normalize so val metrics are reproducible."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def phase2_train_aug(image_size: int = 1024) -> A.Compose:
    """Moderately augmented Phase 2 fine-tuning at 1024x1024.
    Still much gentler than phase 1 (we want fine-detail preservation), but
    bumped slightly: stronger crop range, more color, mild lighting/noise/JPEG
    so the FT phase doesn't just memorize the HQ subset."""
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size),
                             scale=(0.75, 1.0), ratio=(0.80, 1.25), p=1.0),  # was (0.85, 1.0)
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.4, border_mode=0),                              # was 8 / 0.3
        A.Perspective(scale=(0.02, 0.04), p=0.2),                              # NEW
        A.OneOf([
            A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25,
                           hue=0.07, p=1.0),                                    # was 0.2 / 0.05
            A.RandomBrightnessContrast(brightness_limit=0.25,
                                         contrast_limit=0.25, p=1.0),          # was 0.2
            A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=18,
                                  val_shift_limit=12, p=1.0),                   # NEW
        ], p=0.6),                                                              # was 0.5
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1),
                        num_shadows_limit=(1, 2), p=0.2),                       # NEW
        A.RandomToneCurve(scale=0.15, p=0.2),                                   # NEW
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),                                  # NEW
        ], p=0.15),                                                             # was 0.1 only Gaussian
        A.GaussNoise(std_range=(0.02, 0.10), p=0.15),                           # NEW
        A.ImageCompression(quality_range=(70, 100), p=0.2),                     # NEW
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def phase2_val_aug(image_size: int = 1024) -> A.Compose:
    """Deterministic preprocessing for Phase 2 val/test at 1024x1024."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ══════════════════════════════════════════════════════════════════════
# Convenience factories
# ══════════════════════════════════════════════════════════════════════

def make_phase1_train_dataset(splits_dir: Path = DEFAULT_SPLITS_DIR,
                               image_size: int = 512,
                               weight_by_review_source: bool = False
                               ) -> FenceDataset:
    return FenceDataset(
        Path(splits_dir) / "train.jsonl",
        Path(splits_dir) / "train_masks.jsonl",
        transform=phase1_train_aug(image_size),
        weight_by_review_source=weight_by_review_source,
    )


def make_phase1_val_dataset(splits_dir: Path = DEFAULT_SPLITS_DIR,
                             image_size: int = 512,
                             split: str = "val") -> FenceDataset:
    """split: 'val' or 'test'."""
    return FenceDataset(
        Path(splits_dir) / f"{split}.jsonl",
        Path(splits_dir) / f"{split}_masks.jsonl",
        transform=phase1_val_aug(image_size),
    )


def make_phase2_train_dataset(splits_dir: Path = DEFAULT_SPLITS_DIR,
                               image_size: int = 1024,
                               weight_by_review_source: bool = False
                               ) -> FenceDataset:
    return FenceDataset(
        Path(splits_dir) / "train_hq.jsonl",
        Path(splits_dir) / "train_hq_masks.jsonl",
        transform=phase2_train_aug(image_size),
        weight_by_review_source=weight_by_review_source,
    )


def make_phase2_val_dataset(splits_dir: Path = DEFAULT_SPLITS_DIR,
                             image_size: int = 1024,
                             split: str = "val_hq") -> FenceDataset:
    """split: 'val_hq' or 'test_hq'."""
    return FenceDataset(
        Path(splits_dir) / f"{split}.jsonl",
        Path(splits_dir) / f"{split}_masks.jsonl",
        transform=phase2_val_aug(image_size),
    )


# ══════════════════════════════════════════════════════════════════════
# Integrity checks
# ══════════════════════════════════════════════════════════════════════

def verify_split_integrity(
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    splits: tuple[str, ...] = ("train", "val", "test",
                                "train_hq", "val_hq", "test_hq"),
    check_mask_files_exist: bool = True,
    sample_check_count: int = 20,
) -> dict[str, dict]:
    """Run sanity checks on all split files. Raises AssertionError on failure.

    Verifies:
      - Each image split has a parallel mask split
      - No ID overlap between train and val/test (within both phases)
      - Every image_id has a corresponding mask metadata row
      - First N mask files actually exist on disk

    Returns a per-split summary dict.
    """
    splits_dir = Path(splits_dir)
    summary: dict[str, dict] = {}
    img_id_sets: dict[str, set[str]] = {}

    for name in splits:
        img_p = splits_dir / f"{name}.jsonl"
        mask_p = splits_dir / f"{name}_masks.jsonl"
        if not img_p.exists():
            print(f"[skip] {img_p} not found")
            continue
        assert mask_p.exists(), f"missing mask split: {mask_p}"

        img_rows = load_jsonl(img_p)
        mask_rows = load_jsonl(mask_p)
        mask_by_id = {r["id"]: r for r in mask_rows}
        img_ids = {r["id"] for r in img_rows}

        # Every image has a mask row
        missing = [r["id"] for r in img_rows if r["id"] not in mask_by_id]
        assert not missing, (
            f"{name}: {len(missing)} images missing mask metadata "
            f"(first 5: {missing[:5]})"
        )

        # Sample-check that mask files exist on disk
        if check_mask_files_exist:
            for r in mask_rows[:sample_check_count]:
                p = Path(r["mask_path"])
                assert p.exists(), f"{name}: mask file missing: {p}"

        img_id_sets[name] = img_ids
        summary[name] = {
            "rows":             len(img_rows),
            "pos":              sum(1 for r in img_rows if r.get("class") == "pos"),
            "neg":              sum(1 for r in img_rows if r.get("class") == "neg"),
            "manual":           sum(1 for r in mask_rows
                                    if r.get("review_source") == "manual"),
            "auto_accept":      sum(1 for r in mask_rows
                                    if r.get("review_source") == "auto_accept_positive"),
            "auto_clear":       sum(1 for r in mask_rows
                                    if r.get("review_source") == "auto_negative_clear"),
        }

    # Cross-split overlap checks (within each phase)
    for phase_splits in [("train", "val", "test"),
                          ("train_hq", "val_hq", "test_hq")]:
        present = [s for s in phase_splits if s in img_id_sets]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = present[i], present[j]
                ov = img_id_sets[a] & img_id_sets[b]
                assert not ov, (
                    f"ID overlap {a} <-> {b}: {len(ov)} ids "
                    f"(first 5: {list(ov)[:5]})"
                )

    return summary


# ══════════════════════════════════════════════════════════════════════
# Class-imbalance helper
# ══════════════════════════════════════════════════════════════════════

def compute_pos_weight(img_jsonl: str | Path,
                        mask_jsonl: str | Path) -> float:
    """Compute pos_weight for BCEWithLogitsLoss to handle class imbalance.

    Returns N_neg_pixels / N_pos_pixels across the entire split, computed
    from cached `fence_pixel_count` in the mask JSONL (no PNG reads).

    Usage:
        pos_w = compute_pos_weight("dataset/splits/train.jsonl",
                                    "dataset/splits/train_masks.jsonl")
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]))
    """
    img_rows = load_jsonl(img_jsonl)
    mask_rows = {r["id"]: r for r in load_jsonl(mask_jsonl)}
    total_pixels = 0
    pos_pixels = 0
    for ir in img_rows:
        m = mask_rows.get(ir["id"])
        if m is None:
            continue
        w = ir.get("width") or 0
        h = ir.get("height") or 0
        if not (w and h):
            continue
        total_pixels += int(w) * int(h)
        pos_pixels += int(m.get("fence_pixel_count", 0))
    if pos_pixels == 0:
        return 1.0
    return (total_pixels - pos_pixels) / pos_pixels


def compute_class_distribution(img_jsonl: str | Path) -> dict[str, int]:
    """Quick {class: count} distribution for a split."""
    rows = load_jsonl(img_jsonl)
    out: dict[str, int] = {}
    for r in rows:
        c = r.get("class", "?")
        out[c] = out.get(c, 0) + 1
    return out


# ══════════════════════════════════════════════════════════════════════
# DataLoader worker initialization (correctness-critical)
# ══════════════════════════════════════════════════════════════════════

def seed_worker(worker_id: int) -> None:
    """worker_init_fn for DataLoader.

    Without this, every worker process inherits the parent's numpy/random RNG
    state at fork — resulting in IDENTICAL augmentation streams across
    workers and effectively shrinking your dataset variety by num_workers×.

    This pulls a unique seed for each worker from torch's global initial seed
    (which IS distinct per worker) and re-seeds Python `random` and numpy.
    Albumentations relies on numpy's global RNG, so this fixes it.
    """
    import random as _random
    base_seed = torch.initial_seed() % (2**32)
    _random.seed(base_seed + worker_id)
    np.random.seed((base_seed + worker_id) % (2**32))


# ══════════════════════════════════════════════════════════════════════
# Standalone smoke test
# ══════════════════════════════════════════════════════════════════════

def _smoke_test() -> int:
    print("\n" + "=" * 70)
    print("Verifying split integrity")
    print("=" * 70)
    summary = verify_split_integrity()
    header = f"  {'split':<12}  {'rows':>6}  {'pos':>5}  {'neg':>5}  " \
             f"{'manual':>6}  {'auto_acc':>8}  {'auto_clr':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, s in summary.items():
        print(f"  {name:<12}  {s['rows']:>6,}  {s['pos']:>5,}  {s['neg']:>5,}  "
              f"{s['manual']:>6,}  {s['auto_accept']:>8,}  {s['auto_clear']:>8,}")

    print("\n" + "=" * 70)
    print("Loading Phase 1 (512px)")
    print("=" * 70)
    train = make_phase1_train_dataset(weight_by_review_source=True)
    val = make_phase1_val_dataset()
    print(f"  train: {len(train):,}  val: {len(val):,}")
    sample = train[0]
    print(f"  sample[0].image:  {tuple(sample['image'].shape)}  "
          f"dtype={sample['image'].dtype}")
    print(f"  sample[0].mask:   {tuple(sample['mask'].shape)}  "
          f"dtype={sample['mask'].dtype}  unique={sorted(sample['mask'].unique().tolist())}")
    print(f"  sample[0].sample_weight: {sample['sample_weight'].item():.2f}")
    print(f"  sample[0].metadata: {sample['metadata']}")

    print("\n" + "=" * 70)
    print("Loading Phase 2 (1024px)")
    print("=" * 70)
    train_hq = make_phase2_train_dataset()
    val_hq = make_phase2_val_dataset()
    print(f"  train_hq: {len(train_hq):,}  val_hq: {len(val_hq):,}")
    sample = train_hq[0]
    print(f"  sample[0].image: {tuple(sample['image'].shape)}")

    print("\n" + "=" * 70)
    print("Class imbalance (BCEWithLogitsLoss pos_weight)")
    print("=" * 70)
    pw_p1 = compute_pos_weight(DEFAULT_SPLITS_DIR / "train.jsonl",
                                 DEFAULT_SPLITS_DIR / "train_masks.jsonl")
    pw_p2 = compute_pos_weight(DEFAULT_SPLITS_DIR / "train_hq.jsonl",
                                 DEFAULT_SPLITS_DIR / "train_hq_masks.jsonl")
    print(f"  Phase 1 train  pos_weight = {pw_p1:.3f}")
    print(f"  Phase 2 train  pos_weight = {pw_p2:.3f}")
    print()
    print("  Use as:")
    print("    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]))")
    print()
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke_test())
