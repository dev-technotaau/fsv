"""Auto-review images that are known-negative for the wooden-fence task by
clearing any fence masks and stamping them as reviewed in manual_review.jsonl.

Speeds up manual review by skipping images that don't need a human eye on
each one. Every operation mirrors what manual_refine_sam3._save_current does,
so the resulting state is indistinguishable from "user pressed A on each
image": same mask/preview/viz files, same review log entries, same qa_queue
pruning, same heatmap cleanup. The manual_refine_sam3 GUI's counters
(`reviewed (total)`, `[qa-queue] X reviewed Y remaining`) update automatically
on next launch because they read from manual_review.jsonl.

THREE sources of "should be negative":

  1. MANIFEST-NEG    manifest class == "neg" (curated label at scrape time,
                     e.g. neg_pure_random, neg_nonwood_fence)
  2. AUTO-NEG        auto pipeline found no fence (fence_wood_coverage == 0)
  3. FALSE-POSITIVE  manifest subcategory in --false-positive-subcats list
                     (default: 'style_nonwood'). These were labeled
                     class='pos' at scrape time, but they're NON-wooden
                     fences (chain link, metal, vinyl, bamboo, etc.) which
                     SHOULD be negative for the wooden-fence task. The auto
                     pipeline often falsely detects wood-fence on them.

All three sources are processed by default. Use --no-manifest-neg /
--no-auto-neg / --no-false-positives to restrict, or set
--false-positive-subcats to override the default subcategory list.

Side effects (one entry appended to manual_review.jsonl per processed image):
  - masks/<id>.png           overwritten with all-zero mask (binary 0)
  - masks_preview/<id>.png   overwritten with all-black preview
  - viz/<id>.png             overwritten with the source image (no overlay)
  - manual_review.jsonl      one new row, marked auto_negative_clear=True
  - qa_queue.jsonl           entry pruned if present
  - heatmaps/<id>.png        deleted if present (stale)

Safety:
  - Never re-processes images already in manual_review.jsonl (so manual
    decisions are never overwritten)
  - Atomic writes (Ctrl+C never corrupts a file)
  - Dry-run mode shows the plan without writing

Usage:
    python -m annotation.auto_review_negatives                  # process all 3 sources
    python -m annotation.auto_review_negatives --dry-run        # preview plan
    python -m annotation.auto_review_negatives --limit 100      # smoke test
    python -m annotation.auto_review_negatives --no-auto-neg    # only manifest+FP
    python -m annotation.auto_review_negatives \
        --false-positive-subcats style_nonwood,neg_divider_louver
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Reuse the exact helpers used by manual_refine_sam3 (mask-loading + JSONL
# parsing). For PNG writes we use a LOCAL fast version (see below) — the
# manual GUI's _atomic_save_png uses optimize=True (exhaustive compression
# search) which is fine for one-image-at-a-time but ~10x too slow for bulk.
from annotation.manual_refine_sam3 import (
    load_existing_mask,
    load_jsonl_safe,
)


def _atomic_save_png_fast(img: Image.Image, path: Path) -> None:
    """Atomic PNG write tuned for bulk operations.

    Uses compress_level=1 (zlib's fastest non-trivial setting) instead of
    optimize=True. Trade-off:
      - Image content: BIT-IDENTICAL to optimize=True (PNG is lossless at
        every compression level — only file size differs)
      - Speed: ~10x faster (one compression pass vs exhaustive search)
      - File size: ~1.5-3x larger for natural-image viz files; uniform-data
        files (all-zero masks, all-black previews) barely change because
        zlib already collapses runs of identical bytes very efficiently
    Atomicity: same as the slow version — write to .tmp then os.replace.
    Ctrl+C never leaves a partially-written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    img.save(tmp, format="PNG", optimize=False, compress_level=1)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--annotations-root", type=Path,
                    default=Path("dataset/annotations_v1"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("dataset/manifest.jsonl"))
    ap.add_argument("--include-manifest-neg", dest="include_manifest_neg",
                    action="store_true", default=True,
                    help="Process images with manifest class='neg' (default ON).")
    ap.add_argument("--no-manifest-neg", dest="include_manifest_neg",
                    action="store_false",
                    help="Skip manifest-labeled negatives.")
    ap.add_argument("--include-auto-neg", dest="include_auto_neg",
                    action="store_true", default=True,
                    help="Process images where auto pipeline found no fence "
                         "(fence_wood_coverage == 0). Default ON.")
    ap.add_argument("--no-auto-neg", dest="include_auto_neg",
                    action="store_false",
                    help="Skip auto-detected negatives.")
    ap.add_argument("--include-false-positives", dest="include_false_positives",
                    action="store_true", default=True,
                    help="Process images whose subcategory is in "
                         "--false-positive-subcats (i.e., non-wood fences "
                         "labeled positive at scrape time). Default ON.")
    ap.add_argument("--no-false-positives", dest="include_false_positives",
                    action="store_false",
                    help="Skip subcategory-based false positives.")
    ap.add_argument("--false-positive-subcats", type=str,
                    default="style_nonwood,occlusion,occlusion_mild,scene_context",
                    help="Comma-separated subcategory names to treat as "
                         "false positives (clear mask + mark as neg). "
                         "Default: 'style_nonwood,occlusion,occlusion_mild,"
                         "scene_context'. These are excluded from "
                         "wood-fence training data because: style_nonwood "
                         "= wrong material; occlusion* = fence too hidden "
                         "to be useful; scene_context = fence is incidental "
                         "/ in the background and material/extent unreliable. "
                         "Add more if needed, e.g. ',multi_structure'.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and exit without writing anything.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N images (0 = no limit).")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="Skip the confirmation prompt before processing.")
    args = ap.parse_args()

    annotations_root: Path = args.annotations_root
    masks_dir = annotations_root / "masks"
    preview_dir = annotations_root / "masks_preview"
    viz_dir = annotations_root / "viz"
    heatmaps_dir = annotations_root / "heatmaps"
    review_log_path = annotations_root / "manual_review.jsonl"
    qa_queue_path = annotations_root / "qa_queue.jsonl"
    results_path = annotations_root / "results.jsonl"

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    if not annotations_root.exists():
        print(f"ERROR: annotations root not found: {annotations_root}",
              file=sys.stderr)
        return 2
    for d in (masks_dir, preview_dir, viz_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Load sources ─────────────────────────────────────────────────────
    print(f"Loading manifest: {args.manifest}")
    manifest_rows = load_jsonl_safe(args.manifest)
    manifest_by_id: dict[str, dict] = {
        r["id"]: r for r in manifest_rows if "id" in r
    }
    print(f"  {len(manifest_by_id):,} rows")

    results_by_id: dict[str, dict] = {}
    if results_path.exists():
        print(f"Loading auto results: {results_path}")
        for r in load_jsonl_safe(results_path):
            iid = r.get("image_id")
            if iid:
                results_by_id[iid] = r
        print(f"  {len(results_by_id):,} rows")
    else:
        print(f"  (results.jsonl not found; --include-auto-neg will be ineffective)")

    # ── Load existing review log (don't double-process) ──────────────────
    reviewed_ids: set[str] = set()
    if review_log_path.exists():
        for e in load_jsonl_safe(review_log_path):
            iid = e.get("image_id")
            if iid:
                reviewed_ids.add(iid)
        print(f"Already reviewed: {len(reviewed_ids):,} images "
              f"(these will be skipped)")

    # ── Parse false-positive subcategory list ────────────────────────────
    fp_subcats = {
        s.strip() for s in args.false_positive_subcats.split(",") if s.strip()
    }
    if args.include_false_positives and fp_subcats:
        print(f"False-positive subcategories: {sorted(fp_subcats)}")

    # ── Identify candidates ──────────────────────────────────────────────
    # Each candidate is (image_id, manifest_row, is_manifest_neg, is_auto_neg, is_fp)
    candidates: list[tuple[str, dict, bool, bool, bool]] = []
    for iid, row in manifest_by_id.items():
        if iid in reviewed_ids:
            continue
        is_manifest_neg = row.get("class") == "neg"
        auto = results_by_id.get(iid)
        is_auto_neg = bool(auto and float(auto.get("fence_wood_coverage", 0)) == 0.0)
        # is_fp = subcategory is in false-positive list (non-wood fence
        # labeled as pos that the auto pipeline likely got wrong)
        is_fp = bool(fp_subcats and row.get("subcategory") in fp_subcats)
        process = (
            (args.include_manifest_neg and is_manifest_neg)
            or (args.include_auto_neg and is_auto_neg)
            or (args.include_false_positives and is_fp)
        )
        if process:
            candidates.append((iid, row, is_manifest_neg, is_auto_neg, is_fp))

    n_manifest_only = sum(1 for _, _, mn, an, fp in candidates if mn and not an and not fp)
    n_auto_only = sum(1 for _, _, mn, an, fp in candidates if an and not mn and not fp)
    n_fp_only = sum(1 for _, _, mn, an, fp in candidates if fp and not mn and not an)
    n_neg_both = sum(1 for _, _, mn, an, fp in candidates if mn and an and not fp)
    n_fp_with_other = sum(1 for _, _, mn, an, fp in candidates if fp and (mn or an))
    print(f"\nCandidates for auto-clear: {len(candidates):,}")
    print(f"  manifest-neg only:    {n_manifest_only:,}")
    print(f"  auto-neg only:        {n_auto_only:,}")
    print(f"  false-positive only:  {n_fp_only:,}  (subcat in "
          f"--false-positive-subcats, currently labeled pos)")
    print(f"  manifest+auto agree:  {n_neg_both:,}  (both flag as neg)")
    print(f"  FP + neg overlap:     {n_fp_with_other:,}  (FP subcat AND "
          f"neg-flagged for redundancy)")

    if args.limit and args.limit > 0:
        candidates = candidates[:args.limit]
        print(f"\n[--limit] truncating to first {len(candidates):,}")

    if not candidates:
        print("\nNothing to do.")
        return 0

    if args.dry_run:
        print("\n[dry-run] sample candidates (first 10):")
        for iid, row, mn, an, fp in candidates[:10]:
            tags = []
            if mn:
                tags.append("manifest=neg")
            if an:
                tags.append("auto=neg")
            if fp:
                tags.append("FP-subcat")
            print(f"  {iid}  [{','.join(tags)}]  "
                  f"src={row.get('source','?')[:18]:<18}  "
                  f"sub={row.get('subcategory','?')[:24]}")
        print(f"\n[dry-run] would process {len(candidates):,} images. Nothing written.")
        return 0

    # ── Confirm ──────────────────────────────────────────────────────────
    if not args.yes:
        print(f"\nThis will append {len(candidates):,} rows to manual_review.jsonl,")
        print(f"clear any fence masks in those images, and prune them from qa_queue.")
        print(f"Already-reviewed images are NEVER touched.")
        ans = input("Proceed? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return 0

    # ── Load qa_queue (so we can prune) ──────────────────────────────────
    qa_queue_rows = load_jsonl_safe(qa_queue_path) if qa_queue_path.exists() else []
    qa_queue_ids = {r.get("image_id") for r in qa_queue_rows if "image_id" in r}
    qa_queue_ids.discard(None)

    # ── Process ──────────────────────────────────────────────────────────
    n_cleared = 0          # mask had fence pixels that we wiped
    n_already_empty = 0    # mask was empty; just stamped the audit row
    n_class_flipped = 0    # original_class != "neg"
    n_qa_pruned = 0
    n_heatmap_deleted = 0
    n_image_missing = 0
    n_errors = 0

    t0 = time.time()
    last_report = t0
    print(f"\nProcessing {len(candidates):,} images...")

    for i, (iid, row, is_manifest_neg, is_auto_neg, is_fp) in enumerate(candidates, 1):
        try:
            mask_path = masks_dir / f"{iid}.png"
            image_path = Path(row["path"])
            if not image_path.exists():
                n_image_missing += 1
                continue

            # Image dimensions: prefer manifest (cheap) over loading the file
            W = row.get("width")
            H = row.get("height")
            need_load_image = (W is None or H is None)

            # Load existing mask to detect "had fence"
            if W and H:
                original_mask = load_existing_mask(mask_path, int(H), int(W))
            else:
                with Image.open(image_path) as im:
                    W, H = im.size
                original_mask = load_existing_mask(mask_path, int(H), int(W))

            had_fence = bool(original_mask.any())

            # Write all-zero mask + all-black preview. Cheap, small files.
            class_map = np.zeros((int(H), int(W)), dtype=np.uint8)
            _atomic_save_png_fast(Image.fromarray(class_map, mode="L"), mask_path)
            preview = (class_map * 0).astype(np.uint8)   # all zeros for L mode
            _atomic_save_png_fast(Image.fromarray(preview, mode="L"),
                             preview_dir / f"{iid}.png")

            # Viz: only rewrite if the existing viz might have an overlay
            # (i.e., the mask had fence pixels). For images whose mask was
            # already empty, the existing viz should already be just the
            # source — skip the file write to save I/O at scale.
            if had_fence:
                with Image.open(image_path) as im:
                    image_np = np.array(im.convert("RGB"))
                _atomic_save_png_fast(Image.fromarray(image_np),
                                 viz_dir / f"{iid}.png")

            # Append review entry — same shape as manual_refine_sam3 entries
            original_class = row.get("class")
            class_changed = (original_class is not None
                             and original_class != "neg")
            entry = {
                "image_id": iid,
                "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "n_clicks_positive": 0,
                "n_clicks_negative": 0,
                "fence_pixel_count_before": int(original_mask.sum()),
                "fence_pixel_count_after": 0,
                "fence_coverage_after": 0.0,
                "edit_distance": int(original_mask.sum()),
                "edge_refinement_stages": [],
                "original_class": original_class,
                "manual_class": "neg",
                "class_changed": class_changed,
                "dirty_when_saved": had_fence,
                "accept_mode": False,
                "review_duration_s": 0.0,
                # Extra fields for traceability — manual entries don't have these
                "auto_negative_clear": True,
                "auto_clear_reason": "+".join(
                    r for r, flag in [
                        ("manifest_neg", is_manifest_neg),
                        ("auto_neg", is_auto_neg),
                        ("fp_subcat", is_fp),
                    ] if flag
                ),
                "auto_clear_subcategory": row.get("subcategory") if is_fp else None,
            }
            with review_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            reviewed_ids.add(iid)
            if had_fence:
                n_cleared += 1
            else:
                n_already_empty += 1
            if class_changed:
                n_class_flipped += 1

            # Prune from qa_queue (in-memory; flushed once at the end)
            if iid in qa_queue_ids:
                qa_queue_ids.discard(iid)
                qa_queue_rows = [r for r in qa_queue_rows
                                 if r.get("image_id") != iid]
                n_qa_pruned += 1

            # Stale heatmap (best-effort)
            hp = heatmaps_dir / f"{iid}.png"
            if hp.exists():
                try:
                    hp.unlink()
                    n_heatmap_deleted += 1
                except OSError:
                    pass

            now = time.time()
            if now - last_report >= 5.0 or i == len(candidates):
                elapsed = now - t0
                rate = i / max(elapsed, 1e-6)
                eta = (len(candidates) - i) / max(rate, 1e-6)
                print(f"  [{i:>6,}/{len(candidates):,}]  "
                      f"{rate:5.1f} img/s  eta={int(eta):>4}s  "
                      f"cleared={n_cleared:,} stamped={n_already_empty:,} "
                      f"flipped={n_class_flipped:,}")
                last_report = now

        except Exception as e:
            n_errors += 1
            print(f"  [{i}] {iid[:12]} ERROR: {type(e).__name__}: {str(e)[:80]}")

    # ── Flush qa_queue.jsonl with the prunes applied ─────────────────────
    if n_qa_pruned > 0:
        try:
            tmp = qa_queue_path.with_suffix(qa_queue_path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for r in qa_queue_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(tmp, qa_queue_path)
        except (OSError, IOError) as e:
            print(f"WARN: qa_queue.jsonl rewrite failed: "
                  f"{type(e).__name__}: {e}")

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print("Auto-review summary")
    print("=" * 60)
    print(f"  total processed:   {n_cleared + n_already_empty:,}")
    print(f"    fence cleared:   {n_cleared:,}  (mask had pixels we wiped)")
    print(f"    just stamped:    {n_already_empty:,}  (mask was already empty)")
    print(f"  class flipped:     {n_class_flipped:,}  "
          f"(original manifest class was 'pos', now 'neg')")
    print(f"  qa-queue pruned:   {n_qa_pruned:,}")
    print(f"  heatmaps deleted:  {n_heatmap_deleted:,}")
    print(f"  image file missing: {n_image_missing:,}")
    print(f"  errors:            {n_errors:,}")
    print(f"  elapsed:           {elapsed:.1f}s")
    print()
    print(f"  total reviewed (after this run): {len(reviewed_ids):,}")
    print(f"  remaining unreviewed in manifest: "
          f"{len(manifest_by_id) - len(reviewed_ids):,}")
    print()
    print(f"  review log: {review_log_path}")
    print()
    print("Next: launch manual_refine_sam3 — its counters will reflect the new")
    print("reviewed total automatically. Use --order qa-first to focus on")
    print("the still-unreviewed positives.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
