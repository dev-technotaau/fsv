"""Single-image trial: Gemini 3.0 Flash (Vertex AI) auto-mask vs. current
Grounded-SAM 2 mask. Adapted from data_pipeline/auto_label_advanced_v2.py.

Usage (CWD = project root):
    python tools/trial_gemini3_vs_current.py \
        --image-id 0a6002e2-24d1-4607-bfe1-0dfeef36bf37

The image-id must already exist in dataset/manifest.jsonl AND have a current-
method mask in dataset/annotations_v1/masks/. The script:
  1. Looks up the source image from the manifest.
  2. Calls Gemini 3.0 Flash via Vertex AI for fence polygons.
  3. Refines polygons with Shapely (same logic as v2 script).
  4. Rasterizes to a binary mask matching the current-method resolution.
  5. Compares vs. the current-method fence_wood channel.
  6. Writes a side-by-side viz + JSON report to dataset/trials/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from shapely import validation
from shapely.geometry import Polygon
from shapely.ops import unary_union

# ---- Constants from the original v2 script ---------------------------------
MIN_AREA_THRESHOLD = 500
SIMPLIFY_TOLERANCE = 2.0
MIN_POLYGON_POINTS = 3
MAX_POLYGON_POINTS = 500

# Same prompt as data_pipeline/auto_label_advanced_v2.py — keeps the trial honest.
PROMPT = """
Analyze this image and detect all fences with PRECISE BOUNDARIES.

Return a JSON object with these keys:
1. "has_fence": boolean (true if ANY fence exists, false otherwise)
2. "polygons": list of polygons, where each polygon is a list of [y, x] coordinates in NORMALIZED format (0.0 to 1.0)

**CRITICAL INSTRUCTIONS:**
- Trace fence boundaries PRECISELY, following the actual fence edges
- Include ALL visible fence segments (posts, rails, pickets, wire mesh)
- Use sufficient points to capture curves and angles (10-50 points typical)
- Normalize coordinates: divide by image height for y, image width for x
- Order points clockwise or counter-clockwise around the fence perimeter
- If multiple fence segments exist, create separate polygons for each
- Ignore shadows, reflections, or background objects

**Example output:**
{
  "has_fence": true,
  "polygons": [
    [[0.1, 0.2], [0.1, 0.8], [0.3, 0.8], [0.3, 0.2]],
    [[0.5, 0.3], [0.5, 0.7], [0.6, 0.7], [0.6, 0.3]]
  ]
}

If NO fence exists in the image:
{
  "has_fence": false,
  "polygons": []
}
"""


def lookup_manifest(manifest_path: Path, image_id: str) -> dict:
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("id") == image_id:
                return row
    raise SystemExit(f"image_id {image_id} not found in {manifest_path}")


def call_gemini_vertex(image_path: Path, model_name: str, project: str, location: str) -> tuple[dict, float]:
    """Call Gemini via Vertex AI. Returns (parsed_json, latency_seconds)."""
    from google import genai
    from google.genai import types

    client = genai.Client(vertexai=True, project=project, location=location)

    with image_path.open("rb") as f:
        image_bytes = f.read()
    mime = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"

    contents = [
        types.Content(role="user", parts=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            types.Part.from_text(text=PROMPT),
        ])
    ]
    config = types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
    )

    t0 = time.time()
    response = client.models.generate_content(
        model=model_name, contents=contents, config=config,
    )
    elapsed = time.time() - t0
    text = response.text or ""
    return json.loads(text), elapsed


def refine_polygons(polys_raw: list, img_w: int, img_h: int) -> list[list[list[float]]]:
    """Same Shapely refinement as data_pipeline/auto_label_advanced_v2.py."""
    if not polys_raw:
        return []
    shapely_polys = []
    for poly in polys_raw:
        if not isinstance(poly, list) or len(poly) < MIN_POLYGON_POINTS:
            continue
        if len(poly) > MAX_POLYGON_POINTS:
            continue
        try:
            pixel_coords = [(p[1] * img_w, p[0] * img_h) for p in poly]
            shp = Polygon(pixel_coords)
            if not shp.is_valid:
                shp = validation.make_valid(shp)
            if not shp.is_valid or shp.is_empty or shp.area < MIN_AREA_THRESHOLD:
                continue
            simp = shp.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
            if simp.area > MIN_AREA_THRESHOLD:
                shapely_polys.append(simp)
        except Exception:
            continue
    if not shapely_polys:
        return []
    try:
        merged = unary_union(shapely_polys)
        if merged.geom_type == "Polygon":
            final = [merged]
        elif merged.geom_type == "MultiPolygon":
            final = list(merged.geoms)
        else:
            final = shapely_polys
    except Exception:
        final = shapely_polys
    out = []
    for p in final:
        if p.geom_type != "Polygon":
            continue
        coords = list(p.exterior.coords)[:-1]
        out.append([[y / img_h, x / img_w] for x, y in coords])
    return out


def rasterize_polygons(polygons_norm: list, img_w: int, img_h: int) -> np.ndarray:
    """Polygons (normalized [y,x]) → binary uint8 mask {0,255} matching original res.
    Uses the same supersample + Gaussian-smooth approach as convert_labels_v2.py."""
    SUPER = 4
    GAUSS = 5
    th, tw = img_h * SUPER, img_w * SUPER
    big = np.zeros((th, tw), dtype=np.uint8)
    for poly in polygons_norm:
        pts = np.array(
            [[p[1] * tw, p[0] * th] for p in poly], dtype=np.int32,
        )
        if len(pts) >= 3:
            cv2.fillPoly(big, [pts], 255)
    k = GAUSS * SUPER
    if k % 2 == 0:
        k += 1
    big = cv2.GaussianBlur(big, (k, k), 0)
    small = cv2.resize(big, (img_w, img_h), interpolation=cv2.INTER_AREA)
    binary = (small > 127).astype(np.uint8) * 255
    # match v2 dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.dilate(binary, kernel, iterations=1)
    return binary


def binary_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    """pred/ref are uint8 with {0,255}. Treat ref as reference (current method)."""
    p = pred > 127
    r = ref > 127
    tp = int((p & r).sum())
    fp = int((p & ~r).sum())
    fn = int((~p & r).sum())
    tn = int((~p & ~r).sum())
    union = tp + fp + fn
    iou = tp / union if union else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    pixel_agreement = (tp + tn) / (tp + fp + fn + tn)
    return {
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "precision_vs_reference": round(precision, 4),
        "recall_vs_reference": round(recall, 4),
        "pixel_agreement": round(pixel_agreement, 4),
        "pred_coverage_pct": round(100 * p.mean(), 3),
        "ref_coverage_pct": round(100 * r.mean(), 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def make_side_by_side(image_bgr: np.ndarray, gemini_mask: np.ndarray,
                       current_mask_bin: np.ndarray, diff: np.ndarray,
                       out_path: Path) -> None:
    """4-panel composite: original | gemini overlay | current overlay | diff."""
    H, W = image_bgr.shape[:2]
    def overlay(mask: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray:
        col = np.zeros_like(image_bgr)
        col[:] = color_bgr
        m3 = (mask > 127)[:, :, None]
        return np.where(m3, (image_bgr * 0.5 + col * 0.5).astype(np.uint8), image_bgr)

    g_over = overlay(gemini_mask, (0, 0, 255))    # red = Gemini
    c_over = overlay(current_mask_bin, (0, 255, 0))  # green = current

    def label(img: np.ndarray, txt: str) -> np.ndarray:
        out = img.copy()
        cv2.rectangle(out, (0, 0), (W, 28), (0, 0, 0), -1)
        cv2.putText(out, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    top = np.hstack([label(image_bgr, "ORIGINAL"),
                     label(g_over, "GEMINI 3.0 FLASH (red)")])
    bot = np.hstack([label(c_over, "CURRENT: Grounded-SAM 2 (green)"),
                     label(diff, "DIFF: green=ref-only  red=gem-only  yellow=both")])
    composite = np.vstack([top, bot])
    cv2.imwrite(str(out_path), composite)


def diff_visualization(image_bgr: np.ndarray, gemini_mask: np.ndarray,
                        current_mask_bin: np.ndarray) -> np.ndarray:
    """Diff overlay: green=ref-only, red=gem-only, yellow=both."""
    g = gemini_mask > 127
    c = current_mask_bin > 127
    out = image_bgr.copy()
    out[c & ~g] = (out[c & ~g] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
    out[~c & g] = (out[~c & g] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
    out[c & g] = (out[c & g] * 0.4 + np.array([0, 255, 255]) * 0.6).astype(np.uint8)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-id", default="0a6002e2-24d1-4607-bfe1-0dfeef36bf37")
    ap.add_argument("--manifest", type=Path, default=Path("dataset/manifest.jsonl"))
    ap.add_argument("--current-mask-dir", type=Path,
                    default=Path("dataset/annotations_v1/masks"))
    ap.add_argument("--out-root", type=Path,
                    default=Path("dataset/trials/gemini3_flash_vs_current"))
    ap.add_argument("--model", default="gemini-3.0-flash",
                    help="Vertex AI Gemini model id (e.g. gemini-3.0-flash, "
                         "gemini-3-flash-preview, gemini-2.5-flash).")
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--project", default=None,
                    help="GCP project; defaults to project_id from "
                         "GOOGLE_APPLICATION_CREDENTIALS service-account JSON.")
    ap.add_argument("--fence-class-id", type=int, default=1,
                    help="Class id in current mask to treat as fence (default 1=fence_wood).")
    args = ap.parse_args()

    load_dotenv()

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path or not Path(creds_path).exists():
        print(f"ERROR: GOOGLE_APPLICATION_CREDENTIALS not set or file missing: {creds_path}",
              file=sys.stderr)
        return 2

    project = args.project
    if project is None:
        with open(creds_path, "r", encoding="utf-8") as f:
            project = json.load(f).get("project_id")
    if not project:
        print("ERROR: could not determine GCP project_id", file=sys.stderr)
        return 2

    # Lookup source image
    row = lookup_manifest(args.manifest, args.image_id)
    image_path = Path(row["path"])
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        return 2

    # Load current-method mask + image
    current_mask_path = args.current_mask_dir / f"{args.image_id}.png"
    if not current_mask_path.exists():
        print(f"ERROR: current mask missing: {current_mask_path}", file=sys.stderr)
        return 2
    current_full = np.array(Image.open(current_mask_path))
    current_fence_bin = ((current_full == args.fence_class_id).astype(np.uint8)) * 255
    H, W = current_fence_bin.shape

    image_pil = Image.open(image_path).convert("RGB")
    if image_pil.size != (W, H):
        # Resize image to match mask res for visualization parity
        image_pil = image_pil.resize((W, H), Image.LANCZOS)
    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # Output dir
    out_dir = args.out_root / args.image_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[trial] image_id   : {args.image_id}")
    print(f"[trial] source     : {image_path}")
    print(f"[trial] resolution : {W}x{H}")
    print(f"[trial] model      : {args.model}")
    print(f"[trial] project    : {project}")
    print(f"[trial] location   : {args.location}")
    print(f"[trial] out_dir    : {out_dir}")

    # Call Gemini
    print(f"[trial] calling Vertex AI Gemini ...")
    try:
        result, latency_s = call_gemini_vertex(image_path, args.model, project, args.location)
    except Exception as e:
        print(f"ERROR: Gemini call failed: {type(e).__name__}: {e}", file=sys.stderr)
        # Save the error report so the trial dir still has provenance
        (out_dir / "report.json").write_text(json.dumps({
            "image_id": args.image_id, "model": args.model,
            "error": f"{type(e).__name__}: {str(e)[:500]}",
        }, indent=2), encoding="utf-8")
        return 3

    has_fence = bool(result.get("has_fence", False))
    raw_polygons = result.get("polygons", []) if has_fence else []
    print(f"[trial] gemini latency : {latency_s:.2f}s")
    print(f"[trial] has_fence      : {has_fence}")
    print(f"[trial] raw polygons   : {len(raw_polygons)}")

    refined = refine_polygons(raw_polygons, W, H)
    print(f"[trial] refined polygons: {len(refined)}")

    gemini_mask = rasterize_polygons(refined, W, H) if refined else np.zeros((H, W), dtype=np.uint8)
    cv2.imwrite(str(out_dir / "gemini_mask.png"), gemini_mask)
    cv2.imwrite(str(out_dir / "current_fence_binary.png"), current_fence_bin)

    metrics = binary_metrics(gemini_mask, current_fence_bin)
    print(f"[trial] metrics:")
    for k, v in metrics.items():
        print(f"          {k:25s} = {v}")

    diff = diff_visualization(image_bgr, gemini_mask, current_fence_bin)
    cv2.imwrite(str(out_dir / "diff_overlay.png"), diff)
    make_side_by_side(image_bgr, gemini_mask, current_fence_bin, diff,
                       out_dir / "side_by_side.png")

    # Save raw Gemini output + report
    (out_dir / "gemini_raw.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    report = {
        "image_id": args.image_id,
        "source_image": str(image_path),
        "resolution": [W, H],
        "model": args.model,
        "vertex_project": project,
        "vertex_location": args.location,
        "gemini": {
            "latency_s": round(latency_s, 3),
            "has_fence": has_fence,
            "n_raw_polygons": len(raw_polygons),
            "n_refined_polygons": len(refined),
        },
        "current_method": {
            "mask_path": str(current_mask_path),
            "fence_class_id": args.fence_class_id,
            "fence_pixel_count": int((current_fence_bin > 127).sum()),
        },
        "agreement_metrics_treating_current_as_reference": metrics,
        "note": "There is no human-labeled ground truth for this image. "
                "Metrics describe agreement between Gemini's mask and the "
                "current-method (Grounded-SAM 2) fence_wood mask, not absolute accuracy.",
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[trial] wrote: {out_dir}/report.json")
    print(f"[trial] wrote: {out_dir}/side_by_side.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
