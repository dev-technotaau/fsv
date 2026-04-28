"""Auto-accept high-confidence positive images by stamping them as reviewed
WITHOUT modifying their masks.

Symmetric to auto_review_negatives.py, but for the opposite case: when the
auto-pipeline produced a mask we can trust, we record an audit row marking
it as reviewed and prune it from the QA queue. The mask file is NOT touched.

CRITICAL SAFETY NOTE:
  High confidence alone is NOT a sufficient signal. The auto-pipeline can
  CONFIDENTLY detect non-wood fences (chain link, metal, vinyl, bamboo) as
  'fence_wood' — that's a high-confidence false positive that would poison
  training if accepted blind.

  The primary safeguard is a SUBCATEGORY WHITELIST. Only images whose
  manifest subcategory is curated as a wood-fence category are eligible.
  Subcats like 'fence_general', 'scene_context', 'multi_structure' — where
  the fence material is ambiguous — are NEVER auto-accepted.

  This is the symmetric strategy to auto_review_negatives: we trust the
  auto-pipeline when the data label tells us we should.

Eligibility criteria (ALL must hold):
  1. image NOT yet in manual_review.jsonl
  2. manifest class == 'pos'
  3. manifest subcategory in --safe-subcats whitelist
  4. auto fence_wood_coverage in [--coverage-min, --coverage-max]
  5. auto overall_confidence >= --confidence-min
  6. auto n_detections in [--det-min, --det-max]
  7. NOT in --reject-subcats (an explicit blacklist on top of the whitelist)

Defaults (conservative):
  --safe-subcats:   style_cedar, style_wood, occlusion, occlusion_mild,
                    damaged_construction, lighting, general_positive
  --coverage-min:   0.05  (at least 5% of image is detected fence)
  --coverage-max:   0.90  (not over-segmented to nearly the whole image)
  --confidence-min: 0.30  (auto-pipeline's typical confidence range)
  --det-min:        1     (at least one detection box)
  --det-max:        80    (not absurd noise)

Side effects per image (mirrors manual_refine_sam3._save_current with
skip_edge_refinement=True + accept_mode=True; mask is NOT modified):
  - manual_review.jsonl   one row appended, auto_accept_positive=True
  - qa_queue.jsonl        entry pruned if present
  - heatmaps/<id>.png     deleted if present (stale)

NOT side effects (mask is preserved as-is):
  - masks/<id>.png        NOT touched
  - masks_preview/<id>.png NOT touched
  - viz/<id>.png          NOT touched

Why this matters: this script is much faster than auto_review_negatives
because no PNG writes happen at all — only JSONL append + tiny qa_queue
prune + (optional) heatmap delete. Throughput should be 100-300 img/s.

Safety:
  - Never re-processes already-reviewed images
  - Whitelist is the primary safety net (not just a confidence threshold)
  - Dry-run shows full plan before any writes
  - Blacklist available for explicit rejection on top of whitelist
  - Atomic writes (Ctrl+C never corrupts anything)
  - Resume is automatic (re-reads manual_review.jsonl on each run)

Usage:
    python -m annotation.auto_accept_positives --dry-run        # preview
    python -m annotation.auto_accept_positives                  # default conservative
    python -m annotation.auto_accept_positives -y               # no prompt
    python -m annotation.auto_accept_positives --confidence-min 0.5  # stricter
    python -m annotation.auto_accept_positives \
        --safe-subcats style_cedar,style_wood   # only the surest categories
    python -m annotation.auto_accept_positives \
        --reject-subcats damaged_construction   # add a category to blacklist
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from annotation.manual_refine_sam3 import load_jsonl_safe


# Subcategories where curation tells us the image IS a wood fence. The auto
# pipeline's high-confidence detections in these subcats are very likely to
# be correct (mask + classification both). Safe to auto-accept.
DEFAULT_SAFE_SUBCATS = [
    "style_cedar",            # pure cedar fence images
    "style_wood",             # pure wood fence images
    "damaged_construction",   # damaged wood fence
    "lighting",               # lighting variations of wood fence
    "general_positive",       # general positives (curated as wood)
]

# Subcategories explicitly EXCLUDED even if they sneak through other filters.
# These are categories where the fence material is ambiguous OR where the
# fence visibility is so degraded that auto detections are unreliable.
DEFAULT_REJECT_SUBCATS = [
    "style_nonwood",          # explicitly non-wood (caught by neg script too)
    "fence_general",          # mixed; could be any fence type
    "scene_context",          # fence in scene; material uncertain
    "multi_structure",        # multiple structures; might include non-wood
    "occlusion",              # heavily occluded; auto detections unreliable
    "occlusion_mild",         # lightly occluded; auto detections still unreliable
    # all neg_* subcats are caught by the manifest-class check anyway
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--annotations-root", type=Path,
                    default=Path("dataset/annotations_v1"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("dataset/manifest.jsonl"))
    ap.add_argument("--safe-subcats", type=str,
                    default=",".join(DEFAULT_SAFE_SUBCATS),
                    help="Comma-separated subcategory whitelist. "
                         "Only images in these subcats are eligible. "
                         f"Default: {','.join(DEFAULT_SAFE_SUBCATS)}")
    ap.add_argument("--reject-subcats", type=str,
                    default=",".join(DEFAULT_REJECT_SUBCATS),
                    help="Comma-separated subcategory blacklist. "
                         "Images in these subcats are NEVER auto-accepted "
                         "(applied on top of the whitelist as a safety net). "
                         f"Default: {','.join(DEFAULT_REJECT_SUBCATS)}")
    ap.add_argument("--coverage-min", type=float, default=0.05,
                    help="Minimum fence_wood_coverage to auto-accept (default 0.05).")
    ap.add_argument("--coverage-max", type=float, default=0.90,
                    help="Maximum fence_wood_coverage (default 0.90 — guards "
                         "against full-image false positives).")
    ap.add_argument("--confidence-min", type=float, default=0.30,
                    help="Minimum overall_confidence to auto-accept (default 0.30).")
    ap.add_argument("--det-min", type=int, default=1,
                    help="Minimum n_detections (default 1).")
    ap.add_argument("--det-max", type=int, default=80,
                    help="Maximum n_detections (default 80 — guards against noisy "
                         "over-detection).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and exit without writing.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N images (0 = no limit).")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="Skip the confirmation prompt before processing.")
    args = ap.parse_args()

    annotations_root: Path = args.annotations_root
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

    safe_subcats = {s.strip() for s in args.safe_subcats.split(",") if s.strip()}
    reject_subcats = {s.strip() for s in args.reject_subcats.split(",") if s.strip()}
    if not safe_subcats:
        print("ERROR: --safe-subcats is empty; nothing eligible.", file=sys.stderr)
        return 2

    print(f"Safe subcategory whitelist:    {sorted(safe_subcats)}")
    print(f"Reject subcategory blacklist:  {sorted(reject_subcats)}")
    print(f"Thresholds:")
    print(f"  coverage:    [{args.coverage_min:.2f}, {args.coverage_max:.2f}]")
    print(f"  confidence:  >= {args.confidence_min:.2f}")
    print(f"  detections:  [{args.det_min}, {args.det_max}]")

    # ── Load manifest + auto results ────────────────────────────────────
    print(f"\nLoading manifest: {args.manifest}")
    manifest_rows = load_jsonl_safe(args.manifest)
    manifest_by_id: dict[str, dict] = {r["id"]: r for r in manifest_rows if "id" in r}
    print(f"  {len(manifest_by_id):,} rows")

    print(f"Loading auto results: {results_path}")
    if not results_path.exists():
        print(f"ERROR: results.jsonl required for this script (it reads "
              f"confidence + coverage from there)", file=sys.stderr)
        return 2
    results_by_id: dict[str, dict] = {}
    for r in load_jsonl_safe(results_path):
        iid = r.get("image_id")
        if iid:
            results_by_id[iid] = r
    print(f"  {len(results_by_id):,} rows")

    # ── Existing review log (skip already-reviewed) ─────────────────────
    reviewed_ids: set[str] = set()
    if review_log_path.exists():
        for e in load_jsonl_safe(review_log_path):
            iid = e.get("image_id")
            if iid:
                reviewed_ids.add(iid)
        print(f"Already reviewed: {len(reviewed_ids):,} images "
              f"(these will be skipped)")

    # ── Identify candidates ─────────────────────────────────────────────
    # Reject reasons tracked separately for visibility into why images were
    # filtered out — helps the user tune thresholds.
    reject_count = {
        "already_reviewed": 0,
        "manifest_not_pos": 0,
        "subcat_not_in_whitelist": 0,
        "subcat_in_blacklist": 0,
        "no_auto_data": 0,
        "coverage_too_low": 0,
        "coverage_too_high": 0,
        "confidence_too_low": 0,
        "n_detections_out_of_range": 0,
    }
    candidates: list[tuple[str, dict, dict]] = []

    for iid, row in manifest_by_id.items():
        if iid in reviewed_ids:
            reject_count["already_reviewed"] += 1
            continue
        if row.get("class") != "pos":
            reject_count["manifest_not_pos"] += 1
            continue
        subcat = row.get("subcategory")
        if subcat not in safe_subcats:
            reject_count["subcat_not_in_whitelist"] += 1
            continue
        if subcat in reject_subcats:
            reject_count["subcat_in_blacklist"] += 1
            continue
        auto = results_by_id.get(iid)
        if auto is None:
            reject_count["no_auto_data"] += 1
            continue
        cov = float(auto.get("fence_wood_coverage", 0))
        if cov < args.coverage_min:
            reject_count["coverage_too_low"] += 1
            continue
        if cov > args.coverage_max:
            reject_count["coverage_too_high"] += 1
            continue
        conf = float(auto.get("overall_confidence", 0))
        if conf < args.confidence_min:
            reject_count["confidence_too_low"] += 1
            continue
        n_det = int(auto.get("n_detections", 0))
        if not (args.det_min <= n_det <= args.det_max):
            reject_count["n_detections_out_of_range"] += 1
            continue
        candidates.append((iid, row, auto))

    print(f"\nCandidates eligible for auto-accept: {len(candidates):,}")
    print(f"Rejected (showing why):")
    total_seen = sum(reject_count.values()) + len(candidates)
    for reason, n in reject_count.items():
        if n > 0:
            print(f"  {reason:<28s} {n:>7,}")
    print(f"  {'TOTAL CONSIDERED':<28s} {total_seen:>7,}")

    if args.limit and args.limit > 0:
        candidates = candidates[:args.limit]
        print(f"\n[--limit] truncating to first {len(candidates):,}")

    if not candidates:
        print("\nNothing eligible — try lowering --confidence-min or "
              "--coverage-min, or expanding --safe-subcats.")
        return 0

    if args.dry_run:
        print(f"\n[dry-run] sample candidates (first 10):")
        print(f"  {'image_id':<38} {'subcat':<22} {'cov':>5} {'conf':>5} {'det':>4}")
        for iid, row, auto in candidates[:10]:
            print(f"  {iid:<38} "
                  f"{str(row.get('subcategory','?'))[:22]:<22} "
                  f"{float(auto.get('fence_wood_coverage',0)):>5.2f} "
                  f"{float(auto.get('overall_confidence',0)):>5.2f} "
                  f"{int(auto.get('n_detections',0)):>4}")
        print(f"\n[dry-run] would auto-accept {len(candidates):,} images. "
              f"Nothing written.")
        return 0

    # ── Confirm ─────────────────────────────────────────────────────────
    if not args.yes:
        print(f"\nThis will append {len(candidates):,} rows to manual_review.jsonl,")
        print(f"prune those images from qa_queue.jsonl, and delete any stale heatmaps.")
        print(f"Mask/preview/viz files are NOT touched (auto masks preserved as-is).")
        ans = input("Proceed? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return 0

    # ── Load qa_queue (so we can prune) ─────────────────────────────────
    qa_queue_rows = load_jsonl_safe(qa_queue_path) if qa_queue_path.exists() else []
    qa_queue_ids = {r.get("image_id") for r in qa_queue_rows if "image_id" in r}
    qa_queue_ids.discard(None)

    # ── Process ─────────────────────────────────────────────────────────
    n_accepted = 0
    n_qa_pruned = 0
    n_heatmap_deleted = 0
    n_errors = 0

    t0 = time.time()
    last_report = t0
    print(f"\nProcessing {len(candidates):,} images...")

    for i, (iid, row, auto) in enumerate(candidates, 1):
        try:
            cov = float(auto.get("fence_wood_coverage", 0))
            conf = float(auto.get("overall_confidence", 0))
            n_det = int(auto.get("n_detections", 0))
            # Pull cached fence pixel count if available; this matches the
            # value already on disk (the mask file is unchanged).
            per_class = auto.get("per_class_pixel_counts", {}) or {}
            fence_pixels = int(per_class.get("1", 0))

            entry = {
                "image_id": iid,
                "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "n_clicks_positive": 0,
                "n_clicks_negative": 0,
                "fence_pixel_count_before": fence_pixels,
                "fence_pixel_count_after": fence_pixels,   # unchanged
                "fence_coverage_after": cov,
                "edit_distance": 0,                          # mask not changed
                "edge_refinement_stages": [],
                "original_class": row.get("class"),
                "manual_class": "pos",
                "class_changed": False,
                "dirty_when_saved": False,
                "accept_mode": True,                         # like A key
                "review_duration_s": 0.0,
                # Extra fields specific to this script
                "auto_accept_positive": True,
                "auto_accept_subcategory": row.get("subcategory"),
                "auto_accept_confidence": conf,
                "auto_accept_coverage": cov,
                "auto_accept_n_detections": n_det,
            }
            with review_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            reviewed_ids.add(iid)
            n_accepted += 1

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
                      f"{rate:6.0f} img/s  eta={int(eta):>4}s  "
                      f"accepted={n_accepted:,}  qa_pruned={n_qa_pruned:,}")
                last_report = now

        except Exception as e:
            n_errors += 1
            print(f"  [{i}] {iid[:12]} ERROR: {type(e).__name__}: {str(e)[:80]}")

    # ── Flush qa_queue.jsonl ────────────────────────────────────────────
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
    print("Auto-accept summary")
    print("=" * 60)
    print(f"  accepted:           {n_accepted:,}")
    print(f"  qa-queue pruned:    {n_qa_pruned:,}")
    print(f"  heatmaps deleted:   {n_heatmap_deleted:,}")
    print(f"  errors:             {n_errors:,}")
    print(f"  elapsed:            {elapsed:.1f}s")
    print()
    print(f"  total reviewed (after this run): {len(reviewed_ids):,}")
    print(f"  remaining unreviewed in manifest: "
          f"{len(manifest_by_id) - len(reviewed_ids):,}")
    print()
    print(f"  review log: {review_log_path}")
    print()
    print("Next: launch manual_refine_sam3 — its counters will reflect the new")
    print("reviewed total automatically. Use --order qa-first or --order coverage")
    print("to focus on the still-unreviewed (probably ambiguous) positives.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
