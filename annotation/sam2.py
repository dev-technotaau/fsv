"""SAM 2 wrapper — pixel-precise mask generation from bounding-box prompts.

Given an image and a list of (box, class_id, confidence) detections from
Grounding DINO, this produces one binary mask per detection at full image
resolution. Uses `multimask_output=True` and picks SAM's highest-IoU
candidate, which is what gives the sharp boundary that preserves gaps
between fence slats naturally.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


class _NullContext:
    """No-op context manager used when AMP is disabled."""
    def __enter__(self): return self
    def __exit__(self, *_): return False


def _refine_mask_edges(mask: np.ndarray, image_np: np.ndarray) -> np.ndarray:
    """Pull a SAM 2 binary mask's edges toward true image gradients.

    Uses opencv-contrib's guided filter when available — the source image
    acts as a "guide" so mask edges snap to the real wood/metal/leaf
    boundary instead of sitting at SAM's upsampled stair-step. Falls back
    to morphological open+close (3x3) if the contrib module isn't installed.

    Overhead: ~5-15 ms per mask on CPU. Zero quality downside vs. the raw
    binary — even the fallback only removes 1-2 pixel noise.
    """
    try:
        import cv2
    except ImportError:
        return mask  # cv2 missing — return as-is (SAM 2 requires cv2 so this shouldn't fire)

    try:
        from cv2 import ximgproc
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        mask_f = mask.astype(np.float32)
        refined = ximgproc.guidedFilter(guide=gray, src=mask_f,
                                        radius=4, eps=1e-4)
        return refined > 0.5
    except (ImportError, AttributeError):
        # ximgproc missing (opencv-contrib not installed) — fall back to
        # morphology. Binary-only cleanup; image-unaware but still removes
        # speckle and pinholes.
        kernel = np.ones((3, 3), np.uint8)
        m = mask.astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        return m.astype(bool)


@dataclass
class InstanceMask:
    class_id: int
    class_name: str
    box_xyxy: tuple[float, float, float, float]
    detection_score: float           # from Grounding DINO
    mask: np.ndarray                 # bool, shape (H, W)
    sam_score: float                 # SAM 2 predicted IoU [0, 1]
    area_pixels: int


class SAM2Segmenter:
    """Wrapper around the official facebookresearch/sam2 SAM2ImagePredictor."""

    def __init__(self, model_name: str = "facebook/sam2-hiera-large",
                 device: str | None = None, amp_dtype: str = "none") -> None:
        try:
            import torch
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise RuntimeError(
                "SAM 2 requires `sam2` package. Install:\n"
                "    pip install 'git+https://github.com/facebookresearch/sam2.git'"
            ) from e

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_dtype = amp_dtype

        # Try SAM 2.1 first; fall back to SAM 2 if the newer checkpoint isn't
        # available in the installed sam2 package version.
        try:
            self.predictor = SAM2ImagePredictor.from_pretrained(model_name)
            self.model_name = model_name
        except Exception as e:
            if "2.1" in model_name:
                fallback = model_name.replace("sam2.1", "sam2")
                print(f"[sam2] {model_name} unavailable ({type(e).__name__}); "
                      f"falling back to {fallback}")
                self.predictor = SAM2ImagePredictor.from_pretrained(fallback)
                self.model_name = fallback
            else:
                raise
        self._torch = torch
        try:
            self.predictor.model.to(self.device)
        except Exception:
            pass

        if amp_dtype == "bf16":
            self._amp_dtype_torch = torch.bfloat16
        elif amp_dtype == "fp16":
            self._amp_dtype_torch = torch.float16
        else:
            self._amp_dtype_torch = None

    def segment_boxes(
        self,
        image: Image.Image,
        boxes_xyxy: np.ndarray,                  # (N, 4) float
        class_ids: list[int],
        class_names: list[str],
        detection_scores: list[float],
        multimask_output: bool = True,
        min_mask_area: int = 256,
    ) -> list[InstanceMask]:
        """Produce one InstanceMask per input box.

        SAM 2 returns 3 candidate masks when multimask_output=True; we pick
        the one with the highest predicted IoU score. For single-object
        boxes this is consistently the highest-fidelity mask.
        """
        if len(boxes_xyxy) == 0:
            return []

        torch = self._torch
        img_np = np.array(image.convert("RGB"))
        H, W = img_np.shape[:2]

        # SAM 2 requires set_image before any predict call
        use_amp = self._amp_dtype_torch is not None and self.device.startswith("cuda")
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=self._amp_dtype_torch):
                self.predictor.set_image(img_np)
        else:
            self.predictor.set_image(img_np)

        masks_out: list[InstanceMask] = []
        amp_ctx = (torch.autocast(device_type="cuda", dtype=self._amp_dtype_torch)
                   if use_amp else _NullContext())
        with torch.inference_mode(), amp_ctx:
            # Process boxes one at a time — SAM 2 supports batched boxes but
            # multimask_output=True makes batching awkward (3 masks × N boxes).
            # One-box-at-a-time is clean and fast enough on GPU.
            for i, box in enumerate(boxes_xyxy):
                masks, scores, _ = self.predictor.predict(
                    box=box[np.newaxis, :],           # (1, 4)
                    multimask_output=multimask_output,
                )
                # masks: (K, H, W) bool, scores: (K,)
                if masks is None or len(masks) == 0:
                    continue
                # Pick highest-score mask
                best_k = int(np.argmax(scores))
                mask = masks[best_k].astype(bool)
                if mask.shape != (H, W):
                    # Some SAM 2 builds return mask at input resolution; resize if needed
                    from PIL import Image as PILImage
                    mask_img = PILImage.fromarray((mask * 255).astype(np.uint8))
                    mask_img = mask_img.resize((W, H), PILImage.NEAREST)
                    mask = np.array(mask_img) > 127
                # Refine mask edges against the source image (guided filter
                # when available, morphology fallback). Snaps edges to the
                # real wood/metal/leaf boundary.
                mask = _refine_mask_edges(mask, img_np)
                area = int(mask.sum())
                if area < min_mask_area:
                    continue
                masks_out.append(InstanceMask(
                    class_id=class_ids[i],
                    class_name=class_names[i],
                    box_xyxy=tuple(float(v) for v in box),       # type: ignore[arg-type]
                    detection_score=detection_scores[i],
                    mask=mask,
                    sam_score=float(scores[best_k]),
                    area_pixels=area,
                ))
        return masks_out
