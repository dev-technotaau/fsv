"""Interactive SAM 2 click-and-segment annotation tool.

Open each image in a matplotlib window, click on fence to segment with SAM 2,
refine with more clicks, save the mask. Outputs land in the same folder layout
the rest of your pipeline uses (masks/ + masks_preview/ + viz/).

Usage:
    python -m annotation.manual_sam
    python -m annotation.manual_sam --manifest dataset/golden_set/manifest.jsonl \
        --out-root dataset/golden_set
    python -m annotation.manual_sam --no-resume   # re-do every image

Controls (mouse on the image window):
    Left-click   →  add POSITIVE point (this pixel IS fence)  ● green
    Right-click  →  add NEGATIVE point (this pixel is NOT fence)  ✕ red

Controls (keyboard):
    Space  →  SAVE current mask + advance to next image
    n      →  SKIP this image (don't save) + advance
    b      →  BACK to previous image
    r      →  RESET all points on current image
    u      →  UNDO last point
    q      →  QUIT (already-saved masks are preserved)

If you advance with NO points clicked, an empty mask is saved (image has no
wood fence). That's the right behavior for "negative" images.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


# Lazy imports kept inside class so help message works without matplotlib.

class ManualSAMAnnotator:
    """Single-image-at-a-time interactive SAM 2 annotator."""

    def __init__(self, manifest_path: Path, out_root: Path,
                 model_name: str = "facebook/sam2.1-hiera-large",
                 resume: bool = True) -> None:
        self.manifest_path = manifest_path
        self.out_root = out_root
        self.masks_dir = out_root / "masks"
        self.preview_dir = out_root / "masks_preview"
        self.viz_dir = out_root / "viz"
        for d in (self.masks_dir, self.preview_dir, self.viz_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Load manifest rows
        with manifest_path.open("r", encoding="utf-8") as f:
            self.rows = [json.loads(l) for l in f if l.strip()]
        print(f"Loaded {len(self.rows)} manifest rows")

        # Skip already-done if resuming
        if resume:
            done = {p.stem for p in self.masks_dir.glob("*.png")}
            self.rows = [r for r in self.rows if r["id"] not in done]
            print(f"[resume] {len(done)} already labeled, {len(self.rows)} remaining")

        # Load SAM 2 (with 2.1 → 2 fallback)
        print(f"Loading SAM 2 ({model_name})...")
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        try:
            self.predictor = SAM2ImagePredictor.from_pretrained(model_name)
        except Exception as e:
            if "2.1" in model_name:
                fb = model_name.replace("sam2.1", "sam2")
                print(f"[sam2] {model_name} unavailable ({type(e).__name__}); fallback to {fb}")
                self.predictor = SAM2ImagePredictor.from_pretrained(fb)
            else:
                raise
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.predictor.model.to(self.device)
        except Exception:
            pass
        self._torch = torch
        print(f"SAM 2 ready on {self.device}")

        # Annotation state (per image)
        self.idx = 0
        self.points: list[tuple[float, float]] = []
        self.labels: list[int] = []     # 1=positive, 0=negative
        self.current_mask: np.ndarray | None = None
        self.img_np: np.ndarray | None = None
        self.image_id: str | None = None

        # matplotlib state — created in run()
        self.fig = None
        self.ax = None

    # ── lifecycle ─────────────────────────────────────────────────────

    def run(self) -> None:
        if not self.rows:
            print("Nothing to do — all manifest rows already have masks.")
            return
        import matplotlib.pyplot as plt
        self._plt = plt
        plt.rcParams["toolbar"] = "None"
        self.fig, self.ax = plt.subplots(figsize=(13, 8))
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._load_current()
        plt.show()
        print("Done.")

    def _load_current(self) -> None:
        if self.idx >= len(self.rows):
            print("=== All images annotated. Closing. ===")
            self._plt.close(self.fig)
            return
        row = self.rows[self.idx]
        self.image_id = row["id"]
        img_path = Path(row["path"])
        if not img_path.exists():
            print(f"  [skip missing] {img_path}")
            self.idx += 1
            self._load_current()
            return
        print(f"\n[{self.idx + 1}/{len(self.rows)}] {self.image_id} ← {img_path.name}")
        self.img_np = np.array(Image.open(img_path).convert("RGB"))
        # SAM 2: encode image once per image (~1-2 sec on GPU)
        with self._torch.inference_mode():
            self.predictor.set_image(self.img_np)
        self.points.clear()
        self.labels.clear()
        self.current_mask = None
        self._redraw()

    # ── interaction ───────────────────────────────────────────────────

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        # mouse button: 1=left=positive, 3=right=negative
        label = 1 if event.button == 1 else 0
        self.points.append((float(x), float(y)))
        self.labels.append(label)
        self._segment()
        self._redraw()

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key == " ":
            self._save()
            self.idx += 1
            self._load_current()
        elif key == "n":
            print("  [skip] not saving")
            self.idx += 1
            self._load_current()
        elif key == "b":
            if self.idx > 0:
                self.idx -= 1
                self._load_current()
            else:
                print("  Already at first image.")
        elif key == "r":
            print("  [reset] cleared all points")
            self.points.clear()
            self.labels.clear()
            self.current_mask = None
            self._redraw()
        elif key == "u":
            if self.points:
                self.points.pop()
                self.labels.pop()
                self.current_mask = None
                if self.points:
                    self._segment()
                self._redraw()
        elif key == "q":
            print(f"  [quit] stopping at image {self.idx + 1}")
            self._plt.close(self.fig)

    # ── segmentation + render ─────────────────────────────────────────

    def _segment(self) -> None:
        if not self.points:
            self.current_mask = None
            return
        pts = np.array(self.points, dtype=np.float32)
        lbls = np.array(self.labels, dtype=np.int32)
        with self._torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                point_coords=pts,
                point_labels=lbls,
                multimask_output=True,
            )
        best = int(np.argmax(scores))
        self.current_mask = masks[best].astype(bool)

    def _redraw(self) -> None:
        self.ax.clear()
        self.ax.imshow(self.img_np)
        if self.current_mask is not None:
            overlay = np.zeros((*self.current_mask.shape, 4))
            overlay[self.current_mask] = [1.0, 0.0, 0.0, 0.45]
            self.ax.imshow(overlay)
        for (x, y), lbl in zip(self.points, self.labels):
            if lbl == 1:
                self.ax.plot(x, y, "o", color="#00ff66", markersize=11,
                             markeredgecolor="black", markeredgewidth=1.5)
            else:
                self.ax.plot(x, y, "x", color="#ff3030", markersize=12,
                             markeredgecolor="white", markeredgewidth=2)
        npoints = len(self.points)
        npos = sum(self.labels)
        nneg = npoints - npos
        self.ax.set_title(
            f"[{self.idx + 1}/{len(self.rows)}] {self.image_id[:8]}  |  "
            f"+{npos} −{nneg}  |  L-click=add  R-click=remove  "
            f"SPACE=save  N=skip  R=reset  U=undo  B=back  Q=quit",
            fontsize=10,
        )
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    # ── save outputs ──────────────────────────────────────────────────

    def _save(self) -> None:
        H, W = self.img_np.shape[:2]
        if self.current_mask is None:
            class_map = np.zeros((H, W), dtype=np.uint8)
            print(f"  saving EMPTY mask (no fence in this image)")
        else:
            class_map = self.current_mask.astype(np.uint8)
            print(f"  saving mask: {int(class_map.sum())} fence pixels "
                  f"({100 * class_map.sum() / (H * W):.1f}% of image)")

        # Class-ID mask: 0=bg, 1=fence_wood
        Image.fromarray(class_map, mode="L").save(
            self.masks_dir / f"{self.image_id}.png", optimize=True
        )
        # Preview: pure B/W (fence=255, bg=0) for easy visual review
        preview = (class_map * 255).astype(np.uint8)
        Image.fromarray(preview, mode="L").save(
            self.preview_dir / f"{self.image_id}.png", optimize=True
        )
        # Viz: red overlay on source image for QA
        viz = self.img_np.copy()
        if self.current_mask is not None:
            red = np.array([255, 0, 0], dtype=np.float32)
            viz[self.current_mask] = (
                viz[self.current_mask] * 0.5 + red * 0.5
            ).astype(np.uint8)
        Image.fromarray(viz).save(
            self.viz_dir / f"{self.image_id}.png", optimize=True
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Interactive SAM 2 click-segment tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--manifest", type=Path,
                    default=Path("dataset/golden_set/manifest.jsonl"))
    ap.add_argument("--out-root", type=Path,
                    default=Path("dataset/golden_set"),
                    help="Output dir (creates masks/, masks_preview/, viz/ inside).")
    ap.add_argument("--model", type=str, default="facebook/sam2.1-hiera-large",
                    help="SAM 2 HF checkpoint.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Don't skip already-labeled images.")
    args = ap.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    annotator = ManualSAMAnnotator(
        manifest_path=args.manifest,
        out_root=args.out_root,
        model_name=args.model,
        resume=not args.no_resume,
    )
    annotator.run()


if __name__ == "__main__":
    main()
