#!/usr/bin/env python3
"""Generate per-split mask JSONLs alongside the image split JSONLs.

For each existing image split (train, val, test, and the _hq variants) emit a
parallel mask JSONL. Each row in a mask JSONL gives the mask file path + a
small bundle of provenance/stats so training code doesn't have to construct
paths or cross-reference manual_review.jsonl at runtime.

Output (mirrors split_dataset.py / build_hq_final.py naming):
  dataset/splits/train_masks.jsonl
  dataset/splits/val_masks.jsonl
  dataset/splits/test_masks.jsonl
  dataset/splits/train_hq_masks.jsonl
  dataset/splits/val_hq_masks.jsonl
  dataset/splits/test_hq_masks.jsonl

Per-row schema:
  {
    "id":                "<image_id>",         # join key with image split JSONL
    "mask_path":         "dataset/annotations_v1/masks/<id>.png",
    "mask_preview_path": "dataset/annotations_v1/masks_preview/<id>.png",
    "viz_path":          "dataset/annotations_v1/viz/<id>.png",   # if --include-viz
    "class":             "pos" | "neg",        # final class (post manual review)
    "class_source":      "manual_review" | "auto",
    "fence_pixel_count": 12345,                # 0 for negatives
    "fence_coverage":    0.4321,
    "review_source":     "manual" | "auto_accept_positive" |
                          "auto_negative_clear" | "unreviewed",
    "reviewed_at":       "2026-04-26T12:34:56",   # may be null for unreviewed
    "class_changed":     true                  # original_class != final_class
  }

Training code then joins on `id`:

    img_rows  = load_jsonl("dataset/splits/train.jsonl")
    mask_rows = {r["id"]: r for r in load_jsonl("dataset/splits/train_masks.jsonl")}
    for img_row in img_rows:
        m = mask_rows[img_row["id"]]
        image = Image.open(img_row["path"])
        mask  = Image.open(m["mask_path"])
        ...

Usage:
    python tools/build_mask_splits.py
    python tools/build_mask_splits.py --include-viz
    python tools/build_mask_splits.py --splits train val test    # only full splits
    python tools/build_mask_splits.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl_atomic(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if path.exists():
        try:
            os.chmod(path, 0o644)
        except OSError:
            pass
    os.replace(tmp, path)


def classify_review_source(review: dict | None) -> str:
    """Map a manual_review.jsonl entry to a single 'review_source' label."""
    if review is None:
        return "unreviewed"
    if review.get("auto_accept_positive"):
        return "auto_accept_positive"
    if review.get("auto_negative_clear"):
        return "auto_negative_clear"
    return "manual"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--annotations-root", type=Path,
                    default=Path("dataset/annotations_v1"))
    ap.add_argument("--splits-dir", type=Path,
                    default=Path("dataset/splits"))
    ap.add_argument("--manifest-final", type=Path,
                    default=Path("dataset/manifest_final.jsonl"),
                    help="For class + class_source. Required.")
    ap.add_argument("--review-log", type=Path,
                    default=Path("dataset/annotations_v1/manual_review.jsonl"),
                    help="For per-image review provenance + stats.")
    ap.add_argument("--splits", nargs="+",
                    default=["train", "val", "test",
                             "train_hq", "val_hq", "test_hq"],
                    help="Split file basenames (without .jsonl) to process.")
    ap.add_argument("--include-viz", action="store_true",
                    help="Include viz_path in each row (large files; usually "
                         "not needed for training).")
    ap.add_argument("--absolute-paths", action="store_true",
                    help="Write absolute paths (default: project-relative).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    masks_dir = args.annotations_root / "masks"
    preview_dir = args.annotations_root / "masks_preview"
    viz_dir = args.annotations_root / "viz"

    # ── Load manifest_final + review log into lookup tables ──────────────
    if not args.manifest_final.exists():
        print(f"ERROR: {args.manifest_final} not found", file=sys.stderr)
        return 2
    print(f"Loading {args.manifest_final}")
    manifest_by_id: dict[str, dict] = {}
    for r in load_jsonl(args.manifest_final):
        if "id" in r:
            manifest_by_id[r["id"]] = r
    print(f"  {len(manifest_by_id):,} manifest rows")

    review_by_id: dict[str, dict] = {}
    if args.review_log.exists():
        print(f"Loading {args.review_log}")
        # Last write wins (manual_review.jsonl is dedupe-d on manual_refine_sam3
        # startup but we re-dedupe here defensively).
        for e in load_jsonl(args.review_log):
            iid = e.get("image_id")
            if iid:
                review_by_id[iid] = e
        print(f"  {len(review_by_id):,} unique reviewed images")
    else:
        print(f"WARN: {args.review_log} not found; review_source will be "
              f"'unreviewed' for everything")

    def to_path_str(p: Path) -> str:
        if args.absolute_paths:
            return str(p.resolve())
        # Project-relative POSIX-style for portability across train scripts
        try:
            return str(p.as_posix())
        except Exception:
            return str(p)

    # ── For each split, build the mask metadata file ─────────────────────
    grand_total = 0
    for split_name in args.splits:
        src = args.splits_dir / f"{split_name}.jsonl"
        if not src.exists():
            print(f"\n[skip] {src} not found")
            continue
        ids = [r["id"] for r in load_jsonl(src)]
        out = args.splits_dir / f"{split_name}_masks.jsonl"

        rows: list[dict] = []
        n_unreviewed = 0
        n_class_changed = 0
        for iid in ids:
            mf = manifest_by_id.get(iid, {})
            rev = review_by_id.get(iid)
            review_source = classify_review_source(rev)
            if rev is None:
                n_unreviewed += 1
            # Final class + change tracking
            cls = mf.get("class")
            class_source = mf.get("class_source", "auto")
            class_changed = bool(rev and rev.get("class_changed", False))
            if class_changed:
                n_class_changed += 1
            # Pixel counts: prefer manual_review's cached fields; fall back to 0
            fence_pixels = (rev or {}).get("fence_pixel_count_after", 0)
            fence_coverage = (rev or {}).get("fence_coverage_after", 0.0)
            row = {
                "id": iid,
                "mask_path": to_path_str(masks_dir / f"{iid}.png"),
                "mask_preview_path": to_path_str(preview_dir / f"{iid}.png"),
                "class": cls,
                "class_source": class_source,
                "fence_pixel_count": int(fence_pixels) if fence_pixels else 0,
                "fence_coverage": float(fence_coverage) if fence_coverage else 0.0,
                "review_source": review_source,
                "reviewed_at": (rev or {}).get("reviewed_at"),
                "class_changed": class_changed,
            }
            if args.include_viz:
                row["viz_path"] = to_path_str(viz_dir / f"{iid}.png")
            rows.append(row)

        # Source-distribution summary for this split
        from collections import Counter
        source_counts = Counter(r["review_source"] for r in rows)
        class_counts = Counter(r["class"] for r in rows)
        print(f"\n=== {split_name}.jsonl  ->  {out.name} ===")
        print(f"  rows:         {len(rows):,}")
        print(f"  classes:      pos={class_counts.get('pos',0):,} "
              f"neg={class_counts.get('neg',0):,}")
        print(f"  review_src:   manual={source_counts.get('manual',0):,}  "
              f"auto_accept={source_counts.get('auto_accept_positive',0):,}  "
              f"auto_clear={source_counts.get('auto_negative_clear',0):,}  "
              f"unreviewed={source_counts.get('unreviewed',0):,}")
        if n_class_changed:
            print(f"  class_flips:  {n_class_changed:,}")

        if not args.dry_run:
            write_jsonl_atomic(rows, out)
            if split_name == "test":
                # Lock test_masks.jsonl read-only just like test.jsonl
                try:
                    os.chmod(out, 0o444)
                except OSError:
                    pass
                print(f"  wrote: {out}  [read-only]")
            else:
                print(f"  wrote: {out}")
        grand_total += len(rows)

    if args.dry_run:
        print(f"\n[dry-run] would write {grand_total:,} mask rows across all splits")
    else:
        print(f"\nDone.  total mask rows written: {grand_total:,}")
        print("\nTraining code pattern:")
        print("    img_rows = load_jsonl('dataset/splits/train.jsonl')")
        print("    masks    = {r['id']: r for r in "
              "load_jsonl('dataset/splits/train_masks.jsonl')}")
        print("    for img_row in img_rows:")
        print("        m = masks[img_row['id']]")
        print("        image = Image.open(img_row['path'])")
        print("        mask  = Image.open(m['mask_path'])")
    return 0


if __name__ == "__main__":
    sys.exit(main())
