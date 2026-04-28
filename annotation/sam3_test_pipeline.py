"""Standalone SAM 3 wood-fence auto-annotation TEST pipeline.

Uses the official SAM 3 image inference API (Sam3Processor + build_sam3_image_model)
documented in D:/sam3/sam3/examples/sam3_image_predictor_example.ipynb.

NOTE on naming: For STATIC image inference, the SAM 3 image model is used.
SAM 3.1 (multiplex) is a separate checkpoint focused on VIDEO tracking with
multi-object multiplexing. Static-image fence segmentation uses the SAM 3
image model with the same purpose-built architecture.

Maximum-potential features:
  • Multi-prompt ensemble — 10 wood-fence phrasings probed per image,
    union'd into one mask. Image is encoded ONCE; only text prompt changes.
  • Negative-prompt suppression — separate prompts for vinyl/metal/stone/
    wood-siding/wood-furniture detect those regions and remove them from
    the positive mask. Reduces false positives on non-wood fences.
  • Test-time augmentation — original + horizontal flip, masks merged.
  • Multi-stage edge refinement (best available cascade):
        1. DenseCRF (pydensecrf) if installed — gold standard
        2. Guided filter (opencv-contrib) — image-aware fallback
        3. Morphological close+open — universal final pass
  • Per-prompt diagnostic logging — see which phrasings actually fire
  • IoU evaluation vs golden_set masks (if available)
  • Resume support, progress reporting, comprehensive JSON output

Output layout (separate from main pipeline):
  dataset/sam3_test/
    masks/{id}.png            — class-ID mask (0 = bg, 1 = fence_wood)
    masks_preview/{id}.png    — B/W preview (white = fence)
    viz/{id}.png              — bright-red overlay on source image
    heatmaps/{id}.png         — per-pixel max-confidence
    per_image.jsonl           — per-image metadata
    summary.json              — final aggregated metrics

Usage:
    python -m annotation.sam3_test_pipeline
    python -m annotation.sam3_test_pipeline --limit 20
    python -m annotation.sam3_test_pipeline --no-tta --no-crf
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Reduce CUDA fragmentation BEFORE torch loads
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Suppress noisy deprecation warnings from sam3 dependencies
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")


# ─────────────────────────────────────────────────────────────────────────
# Wood-fence prompts (positive + negative for SAM 3's presence-token disambig)
# ─────────────────────────────────────────────────────────────────────────

POSITIVE_PROMPTS = [
    "wooden fence",
    "wood fence",
    "cedar fence",
    "cedar privacy fence",
    "wooden privacy fence",
    "wooden plank fence",
    "wood picket fence",
    "horizontal slat wood fence",
    "wooden shadowbox fence",
    "stockade wood fence",
]

NEGATIVE_PROMPTS = [
    "vinyl fence",
    "metal fence",
    "chain link fence",
    "stone wall",
    "brick wall",
    "concrete fence",
    "wood siding on house",
    "wooden chair",
    "wooden bench",
    "wooden deck",
]


# ─────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class PromptResult:
    prompt: str
    n_detections: int
    max_score: float
    mean_score: float
    fired: bool


@dataclass
class ImageResult:
    image_id: str
    image_path: str
    H: int
    W: int
    fence_pixel_count: int
    fence_coverage: float
    n_total_detections: int
    per_prompt: list[dict]
    iou_vs_golden: float | None
    elapsed_s: float
    edge_refinement_used: list[str]


# ─────────────────────────────────────────────────────────────────────────
# Edge refinement cascade
# ─────────────────────────────────────────────────────────────────────────

class EdgeRefiner:
    """Three-stage cascade. Each stage runs only if available."""

    def __init__(self, prefer_crf: bool = True):
        self.prefer_crf = prefer_crf
        self._crf = self._try_load_crf()
        self._guided = self._try_load_guided()
        avail = []
        if self._crf is not None:
            avail.append("DenseCRF")
        if self._guided is not None:
            avail.append("guided_filter")
        avail.append("morphology")
        print(f"[edge] cascade available: {' -> '.join(avail)}")

    def _try_load_crf(self):
        if not self.prefer_crf:
            return None
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

    def refine(self, mask: np.ndarray, image_np: np.ndarray,
               soft_logits: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
        stages_used: list[str] = []

        # Stage 1: DenseCRF
        if self._crf is not None and soft_logits is not None:
            try:
                mask = self._densecrf_refine(soft_logits, image_np, mask.shape)
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
                mask = refined > 0.5
                stages_used.append("guided_filter")
            except Exception as e:
                print(f"  [edge] guided filter failed: {type(e).__name__}: {str(e)[:80]}")

        # Stage 3: Morphology cleanup — always
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

    def _densecrf_refine(self, soft_logits: np.ndarray, image_np: np.ndarray,
                         shape: tuple) -> np.ndarray:
        dcrf, unary_from_softmax = self._crf
        H, W = shape
        if soft_logits.ndim == 2:
            p = np.stack([1.0 - soft_logits, soft_logits], axis=0).astype(np.float32)
        else:
            p = soft_logits.astype(np.float32)
        d = dcrf.DenseCRF2D(W, H, 2)
        unary = unary_from_softmax(p)
        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image_np.astype(np.uint8), compat=10)
        Q = d.inference(5)
        return np.argmax(np.array(Q).reshape(2, H, W), axis=0).astype(bool)


# ─────────────────────────────────────────────────────────────────────────
# SAM 3 image annotator (uses official Sam3Processor API)
# ─────────────────────────────────────────────────────────────────────────

class SAM3Annotator:
    """Wraps the official SAM 3 image-inference API.

    Key insight: image is encoded ONCE per image (heavy operation). All text
    prompts after that are CHEAP — they just re-run the text encoder + decoder
    over the cached image features. So multi-prompt ensemble is fast.
    """

    def __init__(self, device: str = "cuda", confidence_threshold: float = 0.20,
                 version: str = "sam3") -> None:
        print(f"[sam3] loading SAM {version} image model on {device}...")
        import torch
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # Setup TF32 + bf16 (per official notebook recommendation)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Find bpe_path
        import sam3
        sam3_root = Path(sam3.__file__).parent
        bpe_path = str(sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz")
        print(f"  bpe: {bpe_path}")

        # Build the IMAGE model architecture. By default this also downloads
        # and loads the sam3.pt checkpoint. If user requested sam3.1, build
        # WITHOUT loading any weights and then load sam3.1's detector portion.
        if version == "sam3.1":
            print(f"  [warning] using SAM 3.1 multiplex weights for static-image inference.")
            print(f"            Meta validates 3.1 for video; static-image quality is unverified.")
            # Build model without loading checkpoint
            self.model = build_sam3_image_model(
                bpe_path=bpe_path, device=device, load_from_HF=False,
            )
            self._load_sam31_detector_weights(device)
        else:
            # Standard sam3 path — downloads and loads sam3.pt
            self.model = build_sam3_image_model(bpe_path=bpe_path, device=device)

        self.processor = Sam3Processor(
            self.model, device=device,
            confidence_threshold=confidence_threshold,
        )
        self.device = device
        self._torch = torch

        # Diagnostics
        n = sum(p.numel() for p in self.model.parameters())
        print(f"  total params: {n / 1e6:.1f}M")
        print(f"  on device: {next(self.model.parameters()).device}")

    def _load_sam31_detector_weights(self, device: str) -> None:
        """Manually pull the sam3.1 multiplex checkpoint and load just the
        detector portion into the image model. strict=False so missing
        multiplex-only keys are silently skipped."""
        import torch
        from huggingface_hub import hf_hub_download
        print("  fetching facebook/sam3.1/sam3.1_multiplex.pt ...")
        ckpt_path = hf_hub_download(repo_id="facebook/sam3.1",
                                     filename="sam3.1_multiplex.pt")
        print(f"  loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        # Extract detector.* keys → strip prefix → load into image model
        detector_ckpt = {
            k.replace("detector.", ""): v for k, v in ckpt.items()
            if k.startswith("detector.")
        }
        if not detector_ckpt:
            # Fallback: maybe keys are without prefix
            detector_ckpt = {k: v for k, v in ckpt.items()
                             if not k.startswith(("tracker.", "sam2_predictor."))}
        missing, unexpected = self.model.load_state_dict(detector_ckpt, strict=False)
        print(f"  loaded {len(detector_ckpt)} weights from sam3.1 checkpoint")
        if missing:
            print(f"  missing keys ({len(missing)}, using random init): "
                  f"{missing[:3]}{'...' if len(missing) > 3 else ''}")
        if unexpected:
            print(f"  ignored extra keys ({len(unexpected)}): "
                  f"{unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
        self.model.to(device).eval()

    def encode_image(self, image: Image.Image):
        """One-time image encoding. Returns inference state."""
        torch = self._torch
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
            return self.processor.set_image(image)

    def detect_with_text(self, state, prompt: str) -> dict:
        """Run text prompt against the cached image features. Cheap."""
        torch = self._torch
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
            self.processor.reset_all_prompts(state)
            state = self.processor.set_text_prompt(prompt=prompt, state=state)
        # Extract numpy arrays from state
        masks = state.get("masks")
        scores = state.get("scores")
        boxes = state.get("boxes")
        if masks is None or len(masks) == 0:
            return {
                "masks": np.zeros((0,), dtype=bool),
                "scores": np.zeros((0,), dtype=np.float32),
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "soft_masks": None,
            }
        # Convert to numpy. Force float32 first because numpy doesn't support
        # bfloat16 (which is what tensors are in under torch.autocast(bf16)).
        torch_local = self._torch
        masks_np = masks.detach().to(torch_local.bool).cpu().numpy()
        # Sam3Processor returns masks shape (N, 1, H, W) due to unsqueeze(1)
        # before interpolate. Squeeze the channel dim for downstream code.
        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0]
        scores_np = scores.detach().to(torch_local.float32).cpu().numpy()
        boxes_np = boxes.detach().to(torch_local.float32).cpu().numpy()
        # Get soft masks (for CRF)
        masks_logits = state.get("masks_logits")
        soft_masks = None
        if masks_logits is not None:
            sm = masks_logits.detach().to(torch_local.float32).cpu().numpy()
            # Shape (N, 1, H, W) → (N, H, W)
            if sm.ndim == 4 and sm.shape[1] == 1:
                sm = sm[:, 0]
            soft_masks = sm
        return {
            "masks": masks_np,
            "scores": scores_np,
            "boxes": boxes_np,
            "soft_masks": soft_masks,
        }


# ─────────────────────────────────────────────────────────────────────────
# Mask aggregation helpers
# ─────────────────────────────────────────────────────────────────────────

def merge_masks_with_logits(masks_per_prompt: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
                             H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate masks from multiple prompts. Returns (binary_mask, conf_map)."""
    union = np.zeros((H, W), dtype=bool)
    conf = np.zeros((H, W), dtype=np.float32)
    for masks, scores, soft in masks_per_prompt:
        if len(masks) == 0:
            continue
        for i in range(len(masks)):
            m = masks[i]
            s = float(scores[i])
            if m.shape != (H, W):
                from PIL import Image as PILImage
                m_img = PILImage.fromarray((m * 255).astype(np.uint8))
                m_img = m_img.resize((W, H), PILImage.NEAREST)
                m = np.array(m_img) > 127
            union |= m
            conf = np.maximum(conf, m.astype(np.float32) * s)
    return union, conf


def filter_by_negative_overlap(positive_mask: np.ndarray,
                                negative_results: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
                                min_neg_score: float = 0.30) -> np.ndarray:
    """Drop positive pixels that overlap heavily with confident negative detections."""
    if not negative_results:
        return positive_mask
    H, W = positive_mask.shape
    neg_union = np.zeros((H, W), dtype=bool)
    for masks, scores, _ in negative_results:
        if len(masks) == 0:
            continue
        for i in range(len(masks)):
            if scores[i] >= min_neg_score:
                m = masks[i]
                if m.shape != (H, W):
                    continue
                neg_union |= m
    if not neg_union.any():
        return positive_mask
    return positive_mask & ~neg_union


# ─────────────────────────────────────────────────────────────────────────
# Output writers (atomic)
# ─────────────────────────────────────────────────────────────────────────

def _atomic_save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Pass explicit format='PNG' because PIL can't infer it from .tmp suffix
    img.save(tmp, format="PNG", optimize=True)
    os.replace(tmp, path)


def save_outputs(class_map: np.ndarray, conf_map: np.ndarray,
                 source_image: Image.Image, out_root: Path,
                 image_id: str) -> None:
    H, W = class_map.shape
    _atomic_save(Image.fromarray(class_map.astype(np.uint8), mode="L"),
                 out_root / "masks" / f"{image_id}.png")
    preview = (class_map.astype(np.uint8) * 255).astype(np.uint8)
    _atomic_save(Image.fromarray(preview, mode="L"),
                 out_root / "masks_preview" / f"{image_id}.png")
    src_rgb = np.array(source_image.convert("RGB").resize((W, H)))
    overlay = src_rgb.copy()
    fence = class_map > 0
    red = np.array([255, 0, 0], dtype=np.float32)
    overlay[fence] = (src_rgb[fence] * 0.45 + red * 0.55).astype(np.uint8)
    _atomic_save(Image.fromarray(overlay), out_root / "viz" / f"{image_id}.png")
    heat = (conf_map.clip(0, 1) * 255).astype(np.uint8)
    _atomic_save(Image.fromarray(heat, mode="L"),
                 out_root / "heatmaps" / f"{image_id}.png")


# ─────────────────────────────────────────────────────────────────────────
# IoU eval against golden
# ─────────────────────────────────────────────────────────────────────────

def compute_iou(pred_mask: np.ndarray, golden_mask: np.ndarray) -> float:
    g = (golden_mask == 1)
    p = pred_mask
    inter = (g & p).sum()
    union = (g | p).sum()
    if union == 0:
        return 1.0 if not g.any() and not p.any() else 0.0
    return float(inter / union)


# ─────────────────────────────────────────────────────────────────────────
# Per-image pipeline
# ─────────────────────────────────────────────────────────────────────────

def annotate_image(annotator: SAM3Annotator, refiner: EdgeRefiner,
                   image_path: Path, image_id: str,
                   positive_prompts: list[str], negative_prompts: list[str],
                   use_tta: bool, score_threshold: float
                   ) -> tuple[ImageResult, np.ndarray, np.ndarray, Image.Image]:
    t0 = time.time()
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    img_np = np.array(image)

    # Encode image ONCE (expensive)
    state = annotator.encode_image(image)

    # Stage 1: positive-prompt ensemble (each prompt is CHEAP after image encoding)
    pos_results = []
    per_prompt_log = []
    for p in positive_prompts:
        out = annotator.detect_with_text(state, p)
        masks, scores = out["masks"], out["scores"]
        keep = scores >= score_threshold
        if keep.any():
            pos_results.append((masks[keep], scores[keep], None))
            per_prompt_log.append(PromptResult(
                prompt=p, n_detections=int(keep.sum()),
                max_score=float(scores[keep].max()),
                mean_score=float(scores[keep].mean()),
                fired=True,
            ))
        else:
            per_prompt_log.append(PromptResult(
                prompt=p, n_detections=0, max_score=0.0, mean_score=0.0,
                fired=False,
            ))

    # Stage 2: TTA — flip image and re-run all prompts
    if use_tta:
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        state_flip = annotator.encode_image(flipped)
        for p in positive_prompts:
            out = annotator.detect_with_text(state_flip, p)
            masks, scores = out["masks"], out["scores"]
            keep = scores >= score_threshold
            if keep.any():
                # Flip masks back along width axis
                flipped_back = np.flip(masks[keep], axis=-1)
                pos_results.append((flipped_back, scores[keep], None))

    # Stage 3: aggregate positive masks
    positive_mask, conf_map = merge_masks_with_logits(pos_results, H, W)

    # Stage 4: negative-prompt suppression on the ORIGINAL image
    neg_results = []
    for p in negative_prompts:
        out = annotator.detect_with_text(state, p)
        masks, scores = out["masks"], out["scores"]
        keep = scores >= score_threshold
        if keep.any():
            neg_results.append((masks[keep], scores[keep], None))
    positive_mask_filtered = filter_by_negative_overlap(positive_mask, neg_results)

    # Stage 5: edge refinement cascade
    if positive_mask_filtered.any():
        soft = np.stack([1.0 - conf_map, conf_map], axis=0)
        refined_mask, stages_used = refiner.refine(
            positive_mask_filtered, img_np, soft_logits=soft,
        )
    else:
        refined_mask = positive_mask_filtered
        stages_used = []

    class_map = refined_mask.astype(np.uint8)

    fence_pixel_count = int(class_map.sum())
    fence_coverage = fence_pixel_count / (H * W)
    n_total_detections = sum(len(s) for _, s, _ in pos_results)
    elapsed_s = round(time.time() - t0, 3)

    result = ImageResult(
        image_id=image_id, image_path=str(image_path),
        H=H, W=W,
        fence_pixel_count=fence_pixel_count,
        fence_coverage=fence_coverage,
        n_total_detections=n_total_detections,
        per_prompt=[asdict(p) for p in per_prompt_log],
        iou_vs_golden=None,
        elapsed_s=elapsed_s,
        edge_refinement_used=stages_used,
    )
    return result, class_map, conf_map, image


# ─────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path,
                    default=Path("dataset/golden_set/manifest.jsonl"))
    ap.add_argument("--golden-dir", type=Path,
                    default=Path("dataset/golden_set/masks"))
    ap.add_argument("--out-root", type=Path,
                    default=Path("dataset/sam3_test"))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--score-threshold", type=float, default=0.20)
    ap.add_argument("--version", choices=["sam3", "sam3.1"], default="sam3",
                    help="Which weights to load. 'sam3' = official static-image "
                         "checkpoint (recommended). 'sam3.1' = experimental — uses "
                         "the multiplex video checkpoint's detector portion (Meta "
                         "validates 3.1 only for video; image quality unverified).")
    ap.add_argument("--no-tta", dest="tta", action="store_false", default=True)
    ap.add_argument("--no-crf", dest="crf", action="store_false", default=True)
    ap.add_argument("--no-negative-prompts", dest="use_negatives",
                    action="store_false", default=True)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    print("=" * 70)
    print("SAM 3 image-inference test pipeline")
    print("=" * 70)
    print(f"  manifest:        {args.manifest}")
    print(f"  out_root:        {args.out_root}")
    print(f"  limit:           {args.limit}")
    print(f"  device:          {args.device}")
    print(f"  score_threshold: {args.score_threshold}")
    print(f"  TTA:             {args.tta}")
    print(f"  CRF refinement:  {args.crf}")
    print(f"  negative prompts:{args.use_negatives}")
    print(f"  resume:          {args.resume}")
    print(f"  positive prompts ({len(POSITIVE_PROMPTS)}):")
    for p in POSITIVE_PROMPTS:
        print(f"    - {p}")
    if args.use_negatives:
        print(f"  negative prompts ({len(NEGATIVE_PROMPTS)}):")
        for p in NEGATIVE_PROMPTS:
            print(f"    - {p}")
    print()

    # Load manifest
    rows = [json.loads(l) for l in args.manifest.open(encoding="utf-8") if l.strip()]
    rows = rows[:args.limit]
    print(f"Loaded {len(rows)} images from manifest")

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    per_image_log = out_root / "per_image.jsonl"
    done_ids: set[str] = set()
    if args.resume and per_image_log.exists():
        with per_image_log.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["image_id"])
                except Exception:
                    pass
        print(f"[resume] {len(done_ids)} already done, "
              f"{len(rows) - len(done_ids)} remaining")

    annotator = SAM3Annotator(device=args.device,
                              confidence_threshold=args.score_threshold,
                              version=args.version)
    refiner = EdgeRefiner(prefer_crf=args.crf)
    print()

    negs = NEGATIVE_PROMPTS if args.use_negatives else []
    n_done = 0
    n_skipped = 0
    all_ious: list[float] = []
    t_start = time.time()

    with per_image_log.open("a", encoding="utf-8") as f_log:
        for i, row in enumerate(rows):
            image_id = row["id"]
            if image_id in done_ids:
                n_skipped += 1
                continue
            image_path = Path(row["path"])
            if not image_path.exists():
                print(f"  [{i+1}/{len(rows)}] {image_id[:8]} - file missing, skip")
                continue

            print(f"\n[{i+1}/{len(rows)}] {image_id[:8]}  <-  {image_path.name}")
            try:
                result, class_map, conf_map, image = annotate_image(
                    annotator, refiner, image_path, image_id,
                    POSITIVE_PROMPTS, negs,
                    use_tta=args.tta, score_threshold=args.score_threshold,
                )
            except Exception as e:
                print(f"  [error] {type(e).__name__}: {str(e)[:200]}")
                import traceback
                traceback.print_exc()
                continue

            iou = None
            if args.golden_dir and args.golden_dir.exists():
                gpath = args.golden_dir / f"{image_id}.png"
                if gpath.exists():
                    g_arr = np.array(Image.open(gpath))
                    iou = compute_iou(class_map > 0, g_arr)
                    result.iou_vs_golden = iou
                    all_ious.append(iou)

            save_outputs(class_map, conf_map, image, out_root, image_id)
            f_log.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            f_log.flush()

            n_fired = sum(1 for p in result.per_prompt if p["fired"])
            iou_str = f"  IoU={iou:.3f}" if iou is not None else ""
            refine_str = "/".join(result.edge_refinement_used) if result.edge_refinement_used else "-"
            print(f"  ok fence={result.fence_coverage:.1%}  "
                  f"prompts_fired={n_fired}/{len(POSITIVE_PROMPTS)}  "
                  f"detections={result.n_total_detections}  "
                  f"refine={refine_str}  ({result.elapsed_s:.1f}s){iou_str}")
            n_done += 1

    elapsed = time.time() - t_start
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  processed: {n_done}  (skipped: {n_skipped})")
    print(f"  total time: {elapsed:.0f}s  (avg {elapsed / max(n_done, 1):.1f}s/image)")
    if all_ious:
        ious = np.array(all_ious)
        print(f"  IoU vs golden ({len(ious)} compared):")
        print(f"    mean:   {ious.mean():.3f}")
        print(f"    median: {np.median(ious):.3f}")
        print(f"    P25:    {np.percentile(ious, 25):.3f}")
        print(f"    P75:    {np.percentile(ious, 75):.3f}")
        print(f"    >= 0.5: {(ious >= 0.5).sum()}/{len(ious)}")
        print(f"    >= 0.7: {(ious >= 0.7).sum()}/{len(ious)}")
        print(f"    >= 0.9: {(ious >= 0.9).sum()}/{len(ious)}")

    summary = {
        "n_processed": n_done,
        "n_skipped": n_skipped,
        "elapsed_total_s": elapsed,
        "iou_mean": float(np.mean(all_ious)) if all_ious else None,
        "iou_median": float(np.median(all_ious)) if all_ious else None,
        "iou_count": len(all_ious),
        "settings": {
            "tta": args.tta,
            "crf": args.crf,
            "use_negatives": args.use_negatives,
            "score_threshold": args.score_threshold,
            "positive_prompts": POSITIVE_PROMPTS,
            "negative_prompts": NEGATIVE_PROMPTS if args.use_negatives else [],
        },
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults: {out_root}/")
    print(f"  per_image.jsonl  {per_image_log}")
    print(f"  summary.json     {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
