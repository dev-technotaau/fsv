"""Export final annotation view by merging auto-pipeline results with manual reviews.

Run after manual refinement to produce the authoritative downstream files:
  - dataset/annotations_v1/results_final.jsonl    (auto + manual overrides applied)
  - dataset/manifest_final.jsonl                   (manifest + class flips applied)

The auto-pipeline's results.jsonl and the original manifest.jsonl are
TREATED AS IMMUTABLE. Manual overrides live in manual_review.jsonl as an
event log. This script materializes the merged view on demand. Run it
whenever you need the latest training-ready state — typically:

    1. After a session of manual refinement
    2. Before kicking off a training run
    3. Before computing dataset statistics

Usage:
    python -m annotation.export_final
    python -m annotation.export_final --dry-run
    python -m annotation.export_final --annotations-root dataset/annotations_v1 \
                                       --manifest dataset/manifest.jsonl

Schema of results_final.jsonl rows (additions over results.jsonl):
  - flags: includes "manual_review" for overridden rows
  - needs_review: forced to False on overridden rows
  - per_class_pixel_counts: recomputed from the actual mask file
  - fence_wood_coverage / fence_wood_confidence: recomputed (confidence=1.0)
  - overall_confidence: 1.0 for manually-reviewed rows
  - instance_detections: cleared (auto detections superseded by human)
  - manual_review: nested dict with reviewed_at, n_clicks, class info, etc.

Schema of manifest_final.jsonl rows (additions over manifest.jsonl):
  - original_class: preserves the pre-flip class for traceability
  - class: updated to manual_class if class_changed=True
  - class_source: "manual_review" if flipped, "auto" otherwise
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [warn] {path.name}:{i} bad JSON: {e}", file=sys.stderr)
    return rows


def atomic_write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def latest_review_per_id(review_rows: list[dict]) -> dict[str, dict]:
    """Last write wins — manual_review.jsonl is append-only, so the most
    recent entry per image_id is the authoritative manual state."""
    out: dict[str, dict] = {}
    for r in review_rows:
        iid = r.get("image_id")
        if iid:
            out[iid] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations-root", type=Path,
                    default=Path("dataset/annotations_v1"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("dataset/manifest.jsonl"))
    ap.add_argument("--out-results", type=Path, default=None,
                    help="Where to write results_final.jsonl. "
                         "Default: <annotations-root>/results_final.jsonl")
    ap.add_argument("--out-manifest", type=Path, default=None,
                    help="Where to write manifest_final.jsonl. "
                         "Default: alongside --manifest")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute counts but write nothing.")
    args = ap.parse_args()

    results_path = args.annotations_root / "results.jsonl"
    review_path = args.annotations_root / "manual_review.jsonl"
    masks_dir = args.annotations_root / "masks"
    out_results = args.out_results or (args.annotations_root / "results_final.jsonl")
    out_manifest = args.out_manifest or args.manifest.with_name("manifest_final.jsonl")

    if not results_path.exists():
        print(f"ERROR: {results_path} not found", file=sys.stderr)
        return 2
    if not args.manifest.exists():
        print(f"ERROR: {args.manifest} not found", file=sys.stderr)
        return 2

    # ── Load all sources
    print(f"Loading auto results: {results_path}")
    auto_rows = load_jsonl(results_path)
    print(f"  {len(auto_rows):,} rows")

    review_by_id: dict[str, dict] = {}
    n_review_entries = 0
    if review_path.exists():
        review_rows = load_jsonl(review_path)
        n_review_entries = len(review_rows)
        review_by_id = latest_review_per_id(review_rows)
        print(f"Loading manual reviews: {review_path}")
        print(f"  {n_review_entries:,} entries -> {len(review_by_id):,} unique images")
    else:
        print(f"  no manual_review.jsonl yet -- output will mirror auto results")

    print(f"Loading manifest: {args.manifest}")
    manifest_rows = load_jsonl(args.manifest)
    print(f"  {len(manifest_rows):,} rows")

    # ── Build merged results
    print(f"\nMerging into: {out_results}")
    print(f"  (re-reads each manually-reviewed mask file to recompute pixel counts;")
    print(f"   ~33K mask reads can take 1-5 minutes depending on disk speed)")
    import time
    out_results_rows: list[dict] = []
    n_overridden = 0
    n_class_flipped = 0
    n_mask_missing = 0
    auto_ids_seen: set[str] = set()
    n_total = len(auto_rows)
    t0 = time.time()
    last_report = t0
    for i, auto in enumerate(auto_rows, 1):
        image_id = auto.get("image_id")
        auto_ids_seen.add(image_id)
        review = review_by_id.get(image_id)

        now = time.time()
        if now - last_report >= 5.0 or i == n_total:
            elapsed = now - t0
            rate = i / max(elapsed, 1e-6)
            eta = (n_total - i) / max(rate, 1e-6)
            print(f"  [{i:>6,}/{n_total:,}]  {rate:6.0f} rows/s  "
                  f"eta={int(eta):>4}s  overridden={n_overridden:,}")
            last_report = now

        if review is None:
            out_results_rows.append(auto)
            continue

        # Recompute fields from the on-disk mask (manual is binary 0/1).
        # Use review log fence_pixel_count_after as a fast-path when available
        # — avoids opening the PNG (10-50x faster).
        mask_path = masks_dir / f"{image_id}.png"
        if not mask_path.exists():
            n_mask_missing += 1
            out_results_rows.append(auto)
            continue

        cached_after = review.get("fence_pixel_count_after")
        if cached_after is not None:
            # Fast path: use the cached pixel count from the review log.
            # Get image dimensions from auto data (no PNG read needed).
            per_class_in_auto = auto.get("per_class_pixel_counts", {}) or {}
            total = sum(int(v) for v in per_class_in_auto.values()) if per_class_in_auto else None
            if total is None or total <= 0:
                # Fall back to PNG read for image size
                try:
                    with Image.open(mask_path) as im:
                        W, H = im.size
                    total = max(W * H, 1)
                except Exception as e:
                    print(f"  [warn] failed to size {mask_path}: {e}", file=sys.stderr)
                    out_results_rows.append(auto)
                    continue
            fence_pixels = int(cached_after)
            bg_pixels = max(total - fence_pixels, 0)
            per_class = {"0": bg_pixels, "1": fence_pixels} if fence_pixels else {"0": total}
        else:
            try:
                arr = np.array(Image.open(mask_path), dtype=np.uint8)
            except Exception as e:
                print(f"  [warn] failed to read {mask_path}: {e}", file=sys.stderr)
                out_results_rows.append(auto)
                continue
            H, W = arr.shape[:2]
            total = max(H * W, 1)
            unique, counts = np.unique(arr, return_counts=True)
            per_class = {str(int(u)): int(c) for u, c in zip(unique, counts)}
            fence_pixels = int(per_class.get("1", 0))

        merged = dict(auto)
        merged["per_class_pixel_counts"] = per_class
        merged["n_classes_present"] = sum(1 for v in per_class.values() if v > 0)
        merged["fence_wood_coverage"] = fence_pixels / total
        merged["fence_wood_confidence"] = 1.0   # human-reviewed = perfect
        merged["overall_confidence"] = 1.0
        merged["flags"] = sorted(set(auto.get("flags", [])) | {"manual_review"})
        merged["needs_review"] = False
        merged["instance_detections"] = []     # auto detections superseded
        merged["n_detections"] = 0
        merged["manual_review"] = {
            "reviewed_at": review.get("reviewed_at"),
            "n_clicks_positive": review.get("n_clicks_positive", 0),
            "n_clicks_negative": review.get("n_clicks_negative", 0),
            "fence_pixel_count_before": review.get("fence_pixel_count_before"),
            "fence_pixel_count_after": review.get("fence_pixel_count_after"),
            "edit_distance": review.get("edit_distance", 0),
            "edge_refinement_stages": review.get("edge_refinement_stages", []),
            "original_class": review.get("original_class"),
            "manual_class": review.get("manual_class"),
            "class_changed": bool(review.get("class_changed", False)),
            "dirty_when_saved": review.get("dirty_when_saved"),
        }
        if review.get("class_changed"):
            n_class_flipped += 1
        n_overridden += 1
        out_results_rows.append(merged)

    # Reviews referencing image_ids not present in results.jsonl are surfaced
    # as a warning (shouldn't normally happen, but flag if so).
    orphan_reviews = set(review_by_id) - auto_ids_seen
    if orphan_reviews:
        print(f"  WARN: {len(orphan_reviews)} reviews reference image_ids "
              f"not in results.jsonl (showing 5):")
        for iid in list(orphan_reviews)[:5]:
            print(f"    - {iid}")

    print(f"  {n_overridden:,} rows overridden by manual review")
    print(f"  {n_class_flipped:,} class flips applied")
    if n_mask_missing:
        print(f"  WARN: {n_mask_missing} reviews reference missing mask files")

    # ── Build merged manifest
    print(f"\nMerging into: {out_manifest}")
    out_manifest_rows: list[dict] = []
    n_manifest_flipped = 0
    n_manifest_reviewed = 0
    for row in manifest_rows:
        image_id = row.get("id")
        review = review_by_id.get(image_id) if image_id else None
        if review is not None:
            n_manifest_reviewed += 1
            if review.get("class_changed"):
                row = dict(row)
                row["original_class"] = row.get("class")
                row["class"] = review.get("manual_class")
                row["class_source"] = "manual_review"
                n_manifest_flipped += 1
            else:
                row = dict(row)
                row["class_source"] = "manual_review"   # confirmed by human
        else:
            row = dict(row)
            row["class_source"] = "auto"
        out_manifest_rows.append(row)
    print(f"  {n_manifest_reviewed:,} manifest rows touched by manual review")
    print(f"  {n_manifest_flipped:,} manifest rows flipped pos<->neg")

    # ── Write
    if args.dry_run:
        print(f"\n[dry-run] would write:")
        print(f"  {out_results}")
        print(f"  {out_manifest}")
        return 0

    atomic_write_jsonl(out_results_rows, out_results)
    atomic_write_jsonl(out_manifest_rows, out_manifest)
    print(f"\nWrote:")
    print(f"  {out_results}    ({len(out_results_rows):,} rows)")
    print(f"  {out_manifest}   ({len(out_manifest_rows):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
