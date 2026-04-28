"""Annotation CLI — run the Grounded-SAM 2 pipeline over a manifest.

Usage:
    python -m annotation.cli \
        --manifest dataset/manifest.jsonl \
        --out-root dataset/annotations_v1 \
        --schema configs/annotation_schema.yaml

    python -m annotation.cli --dry-run        # validate schema + show plan
    python -m annotation.cli --limit 10       # annotate 10 images (smoke test)
    python -m annotation.cli --resume         # skip already-annotated images
    python -m annotation.cli --only-positives # skip negative-class rows
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from annotation.pipeline import AnnotationPipeline
from annotation.schema import load_schema


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def fmt_eta(elapsed_s: float, done: int, total: int) -> str:
    if done == 0:
        return "--:--"
    rate = done / max(elapsed_s, 1e-6)
    remain = (total - done) / max(rate, 1e-6)
    m, s = divmod(int(remain), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path,
                    default=Path("dataset/manifest.jsonl"))
    ap.add_argument("--schema", type=Path,
                    default=Path("configs/annotation_schema_binary.yaml"),
                    help="Annotation schema YAML. Defaults to binary (fence_wood "
                         "vs background) which is the production schema. Use "
                         "configs/annotation_schema.yaml for the legacy 24-class "
                         "version.")
    ap.add_argument("--out-root", type=Path,
                    default=Path("dataset/annotations_v1"),
                    help="Output root: masks/, viz/, heatmaps/, results.jsonl, qa_queue.jsonl")
    ap.add_argument("--device", type=str, default=None,
                    help="'cuda', 'cuda:0', 'cpu', or auto-detect (default)")
    ap.add_argument("--amp", type=str, default="none",
                    choices=["none", "fp16", "bf16"],
                    help="Mixed-precision inference: 'none' (default, full fp32), "
                         "'bf16' (Ampere+ GPUs, ~2x faster), 'fp16' (older GPUs). "
                         "Ignored on CPU.")
    ap.add_argument("--low-vram", dest="low_vram", action="store_true",
                    default=None,
                    help="Enable memory-efficient SDPA + per-image cleanup + "
                         "OOM auto-retry. Zero quality trade-off. "
                         "Auto-enabled on GPUs with <=8 GB VRAM.")
    ap.add_argument("--no-low-vram", dest="low_vram", action="store_false",
                    help="Force-disable --low-vram even if auto-detect would enable it.")
    # --- Accuracy enhancements (from DINO-upgrade / two-pass / scene / TTA) ---
    ap.add_argument("--two-pass-fence", dest="two_pass_fence", action="store_true",
                    default=True,
                    help="Enable two-pass spatial filter: suppress occluder/"
                         "construction/fg-obj detections that overlap strong "
                         "fence detections. Default ON — safe accuracy boost.")
    ap.add_argument("--no-two-pass-fence", dest="two_pass_fence",
                    action="store_false")
    ap.add_argument("--scene-filter", dest="scene_filter", action="store_true",
                    default=True,
                    help="Run a CLIP-based scene classifier to flag OOD images "
                         "(interior rooms, documents, abstract close-ups). "
                         "Adds ~1s/image, catches data-quality issues.")
    ap.add_argument("--no-scene-filter", dest="scene_filter",
                    action="store_false")
    ap.add_argument("--tta", dest="tta", action="store_true",
                    default=True,
                    help="Test-time augmentation: horizontal flip + detect + "
                         "merge with NMS. ~2x detection runtime, +3-8%% recall. "
                         "ON by default; use --no-tta to disable.")
    ap.add_argument("--no-tta", dest="tta", action="store_false",
                    help="Disable TTA (faster, ~3-8%% less recall).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip images already in results.jsonl")
    ap.add_argument("--retry-missing", action="store_true",
                    help="With --resume, also re-process any image whose row is "
                         "in results.jsonl but whose mask file is missing on "
                         "disk (errored rows, unflushed async saves, etc.).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Load schema + manifest, print plan, exit")
    ap.add_argument("--limit", type=int, default=0,
                    help="Annotate at most N images (0 = all)")
    ap.add_argument("--only-positives", action="store_true")
    ap.add_argument("--only-negatives", action="store_true")
    ap.add_argument("--no-viz", action="store_true",
                    help="Skip saving colorized overlay PNGs")
    ap.add_argument("--save-heatmap", action="store_true",
                    help="Save confidence heatmap alongside each mask")
    ap.add_argument("--subcategory", type=str, default=None,
                    help="Only annotate rows with this subcategory")
    ap.add_argument("--qa-sample-rate", type=float, default=None,
                    help="Override sample_rate_for_qa from schema")
    ap.add_argument("--qa-seed", type=int, default=42)
    args = ap.parse_args()

    if args.only_positives and args.only_negatives:
        print("ERROR: --only-positives and --only-negatives are mutually exclusive",
              file=sys.stderr)
        return 2

    # ── Load + validate schema ────────────────────────────────────────
    print(f"Loading schema: {args.schema}")
    try:
        schema = load_schema(args.schema)
    except Exception as e:
        print(f"ERROR: schema load failed: {e}", file=sys.stderr)
        return 2
    print(f"  {len(schema.classes)} classes, {schema.num_classes} total (with bg)")
    print(f"  pipeline: grounding_dino={schema.pipeline.grounding_dino_model}")
    print(f"  pipeline: sam2={schema.pipeline.sam2_model}")

    # ── Load manifest ─────────────────────────────────────────────────
    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    rows = load_jsonl(args.manifest)
    print(f"Loaded {len(rows):,} manifest rows")

    # ── Filter ────────────────────────────────────────────────────────
    if args.only_positives:
        rows = [r for r in rows if r.get("class") == "pos"]
    elif args.only_negatives:
        rows = [r for r in rows if r.get("class") == "neg"]
    if args.subcategory:
        rows = [r for r in rows if r.get("subcategory") == args.subcategory]
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]
    print(f"After filter: {len(rows):,} rows to annotate")

    # ── Set up output dirs ────────────────────────────────────────────
    mask_dir = args.out_root / "masks"
    viz_dir = None if args.no_viz else args.out_root / "viz"
    heat_dir = args.out_root / "heatmaps" if args.save_heatmap else None
    results_jsonl = args.out_root / "results.jsonl"
    qa_queue = args.out_root / "qa_queue.jsonl"

    print(f"\nOutput root: {args.out_root}")
    print(f"  masks:       {mask_dir}")
    print(f"  viz:         {viz_dir or '(disabled)'}")
    print(f"  heatmaps:    {heat_dir or '(disabled)'}")
    print(f"  results:     {results_jsonl}")
    print(f"  qa queue:    {qa_queue}")

    if args.dry_run:
        print("\n[dry-run] no annotation performed. Exiting.")
        return 0

    # ── Resolve --low-vram (auto-enable on small GPUs) ────────────────
    low_vram = args.low_vram
    if low_vram is None:
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                low_vram = vram_gb <= 8.0
                if low_vram:
                    print(f"[compute] auto-enabling --low-vram "
                          f"(detected {vram_gb:.1f} GB VRAM <= 8 GB threshold)")
            else:
                low_vram = False
        except Exception:
            low_vram = False

    # ── Build pipeline + warm up ──────────────────────────────────────
    print(f"\nInitializing Grounding DINO + SAM 2 (device={args.device or 'auto'}, "
          f"amp={args.amp}, low_vram={low_vram})")
    print(f"  two_pass_fence={args.two_pass_fence}  scene_filter={args.scene_filter}  "
          f"tta={args.tta}")
    pipe = AnnotationPipeline(
        schema=schema, device=args.device,
        save_viz=not args.no_viz, save_heatmap=args.save_heatmap,
        amp_dtype=args.amp,
        low_vram=low_vram,
        two_pass_fence=args.two_pass_fence,
        tta=args.tta,
        scene_filter=args.scene_filter,
    )
    try:
        pipe.warm_up()
    except Exception as e:
        print(f"ERROR: model load failed: {e}", file=sys.stderr)
        print("\nInstall the required packages:", file=sys.stderr)
        print("  pip install torch transformers Pillow", file=sys.stderr)
        print("  pip install 'git+https://github.com/facebookresearch/sam2.git'",
              file=sys.stderr)
        return 3
    print("Models ready.")

    # ── Annotate ──────────────────────────────────────────────────────
    t0 = time.time()
    last_report = t0

    def report(done: int, total: int, image_id: str, result=None, error=None):
        nonlocal last_report
        now = time.time()
        if now - last_report >= 5 or done == total:
            elapsed = now - t0
            eta = fmt_eta(elapsed, done, total)
            extra = ""
            if result is not None:
                extra = (f"conf={result.overall_confidence:.2f} "
                         f"n_inst={result.n_detections} "
                         f"{'[FLAG]' if result.needs_review else ''}")
            if error:
                extra = f"ERROR: {error[:60]}"
            print(f"  [{done:>5}/{total}] elapsed={elapsed:6.0f}s eta={eta} "
                  f"id={image_id[:8]} {extra}")
            last_report = now

    n_processed, n_flagged = pipe.annotate_manifest(
        manifest_rows=rows,
        mask_dir=mask_dir,
        viz_dir=viz_dir,
        heatmap_dir=heat_dir,
        results_jsonl=results_jsonl,
        qa_queue_jsonl=qa_queue,
        resume=args.resume,
        qa_sample_rate=args.qa_sample_rate,
        qa_seed=args.qa_seed,
        progress_callback=report,
        retry_missing=args.retry_missing,
    )

    elapsed = time.time() - t0
    print(f"\n=== Summary ===")
    print(f"  Processed: {n_processed:,}")
    print(f"  Flagged for QA: {n_flagged:,} "
          f"({100*n_flagged/max(n_processed,1):.1f}%)")
    print(f"  Elapsed: {elapsed:.0f}s "
          f"(avg {elapsed/max(n_processed,1):.2f}s/image)")
    print(f"\n  results -> {results_jsonl}")
    print(f"  QA queue -> {qa_queue}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
