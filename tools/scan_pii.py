#!/usr/bin/env python3
"""scan_pii.py — flag images containing identifiable human faces.

Runs Google Vision's Face Detection API over the dataset and emits a PII
report. The underlying concerns are GDPR / CCPA / client privacy policy:
if any user-identifiable individual appears in training data, they must
either (a) be informed and consent, (b) have their face blurred, or
(c) be excluded from the training set.

Strategy: most fence photos are empty yards → near-zero faces. The
`pos:humans_animals` subcategory is ~57 rows where faces are expected.
Scanning the whole 33k set is wasteful; we scan ONLY the rows likely to
have people (humans_animals subcategory + sample from other categories).

Actions:
  - Sends image bytes to Vision face_detection in parallel batches
  - For each row, records: n_faces, max_face_size_ratio, max_confidence
  - Writes dataset/PII_SCAN_REPORT.md + dataset/pii_scan_report.jsonl
  - Does NOT modify images (blur/crop) — that's a separate decision

Usage:
    python tools/scan_pii.py                          # scan humans_animals + 5% sample
    python tools/scan_pii.py --subcategory pos:humans_animals    # just that subcategory
    python tools/scan_pii.py --all                    # scan everything (costs ~$50)
    python tools/scan_pii.py --sample-rate 0.1        # scan 10% sample of other cats
    python tools/scan_pii.py --dry-run                # preview without API calls

Cost: ~$1.50 per 1000 images at Vision face_detection list price.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════
# Google Vision initialization
# ══════════════════════════════════════════════════════════════════════

def init_vision():
    """Return a Vision client. Raises if credentials aren't configured."""
    try:
        from google.cloud import vision
    except ImportError:
        raise RuntimeError("google-cloud-vision not installed. "
                           "pip install google-cloud-vision")
    return vision.ImageAnnotatorClient(), vision


# ══════════════════════════════════════════════════════════════════════
# Per-image face detection (thread worker)
# ══════════════════════════════════════════════════════════════════════

def detect_faces(client, vision_mod, path: Path) -> dict:
    """Return {n_faces, max_area_ratio, max_confidence, largest_bbox}."""
    try:
        data = path.read_bytes()
    except Exception as e:
        return {"error": f"read:{type(e).__name__}:{str(e)[:80]}",
                "n_faces": 0, "max_area_ratio": 0.0,
                "max_confidence": 0.0, "largest_bbox": None}

    image = vision_mod.Image(content=data)
    try:
        resp = client.face_detection(image=image)
    except Exception as e:
        return {"error": f"api:{type(e).__name__}:{str(e)[:80]}",
                "n_faces": 0, "max_area_ratio": 0.0,
                "max_confidence": 0.0, "largest_bbox": None}

    faces = list(resp.face_annotations or [])
    if not faces:
        return {"error": None, "n_faces": 0, "max_area_ratio": 0.0,
                "max_confidence": 0.0, "largest_bbox": None}

    # Compute relative size of each face vs image area
    try:
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            W, H = im.size
        img_area = max(W * H, 1)
    except Exception:
        img_area = 0

    max_area_ratio = 0.0
    max_conf = 0.0
    largest_bbox = None
    for f in faces:
        verts = f.bounding_poly.vertices
        if len(verts) >= 3:
            xs = [v.x or 0 for v in verts]
            ys = [v.y or 0 for v in verts]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            area = w * h
            ratio = (area / img_area) if img_area else 0.0
            if ratio > max_area_ratio:
                max_area_ratio = ratio
                largest_bbox = {"x": min(xs), "y": min(ys), "w": w, "h": h}
        conf = float(getattr(f, "detection_confidence", 0.0) or 0.0)
        if conf > max_conf:
            max_conf = conf

    return {
        "error": None,
        "n_faces": len(faces),
        "max_area_ratio": round(max_area_ratio, 4),
        "max_confidence": round(max_conf, 3),
        "largest_bbox": largest_bbox,
    }


# ══════════════════════════════════════════════════════════════════════
# Selection — which images to scan
# ══════════════════════════════════════════════════════════════════════

def pick_rows_to_scan(manifest: list[dict], mode: str, subcategory: str | None,
                      sample_rate: float, seed: int) -> list[dict]:
    """mode: 'all' | 'targeted' | 'subcategory'."""
    if mode == "all":
        return list(manifest)
    if mode == "subcategory":
        return [r for r in manifest
                if f"{r['class']}:{r['subcategory']}" == subcategory]

    # 'targeted' — always scan humans_animals + human-like subcats, sample others
    rng = random.Random(seed)
    people_subcats = {
        "pos:humans_animals", "pos:incidental_background",
        "neg:neg_slatted_furniture",   # often has people on chairs
        "neg:neg_gate_door",           # sometimes has people
        "neg:neg_pure_random",         # has restaurant/cafe/home scenes with people
    }
    out: list[dict] = []
    for r in manifest:
        key = f"{r['class']}:{r['subcategory']}"
        if key in people_subcats:
            out.append(r)
        elif rng.random() < sample_rate:
            out.append(r)
    return out


# ══════════════════════════════════════════════════════════════════════
# Report builder
# ══════════════════════════════════════════════════════════════════════

def build_report_md(scanned_n: int, faces_n: int, significant_n: int,
                    thresholds: dict, by_subcat: dict, results_flagged: list[dict],
                    total_manifest: int) -> str:
    L: list[str] = []
    L.append("# PII / Face Detection Report")
    L.append("")
    L.append(f"_Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_")
    L.append("")
    L.append("## Scan summary")
    L.append(f"- **Manifest total**: {total_manifest:,}")
    L.append(f"- **Images scanned**: {scanned_n:,} "
             f"({100*scanned_n/max(total_manifest,1):.1f}% of manifest)")
    L.append(f"- **Images with >=1 face detected**: {faces_n:,} "
             f"({100*faces_n/max(scanned_n,1):.1f}% of scanned)")
    L.append(f"- **Significant-face images** (face area >= "
             f"{100*thresholds['min_area_ratio']:.0f}% of image AND confidence "
             f">= {thresholds['min_confidence']:.1f}): **{significant_n:,}**")
    L.append("")
    L.append("## Why this matters")
    L.append("")
    L.append("Under GDPR (EU) and CCPA (CA), identifiable human faces in training "
             "data require either:")
    L.append("- Documented consent from the individual, OR")
    L.append("- Face blurring / bbox cropping, OR")
    L.append("- Exclusion from the training set")
    L.append("")
    L.append("Incidental, distant, or non-identifying faces (small area, low "
             "confidence) are generally lower-risk. The **significant-face** "
             "count above is the number requiring action.")
    L.append("")
    L.append("## Distribution by subcategory (face-containing images)")
    L.append("")
    L.append("| Subcategory | Faces found | Of total scanned in cat |")
    L.append("|-------------|-------------|------------------------|")
    for key, (n_face, n_total) in sorted(by_subcat.items(),
                                          key=lambda kv: -kv[1][0]):
        if n_face == 0:
            continue
        L.append(f"| `{key}` | {n_face} | {n_face}/{n_total} "
                 f"({100*n_face/max(n_total,1):.1f}%) |")
    L.append("")
    L.append("## Recommended actions")
    L.append("")
    L.append("1. **Review significant-face images** listed in "
             "`dataset/pii_scan_report.jsonl` (filter `n_faces > 0 "
             "and max_area_ratio >= 0.05`).")
    L.append("2. For each: decide blur / exclude / accept-as-de-minimis")
    L.append("3. For blur: use bounding box from `largest_bbox` to apply "
             "Gaussian blur or mosaic to the face region before training.")
    L.append("4. For exclude: filter out the `id` from `manifest.jsonl` "
             "before training. Document the removal count for audit.")
    L.append("5. For accept: document rationale (e.g. background crowd, "
             "not individually identifiable) in deployment sign-off notes.")
    L.append("")
    if results_flagged:
        L.append("## Top flagged images (largest face size)")
        L.append("")
        L.append("| ID | Class:Subcat | N faces | Area ratio | Confidence | Path |")
        L.append("|----|-------------|---------|------------|------------|------|")
        for r in sorted(results_flagged, key=lambda x: -x["max_area_ratio"])[:25]:
            L.append(f"| `{r['id'][:8]}` | {r['class']}:{r['subcategory']} | "
                     f"{r['n_faces']} | {100*r['max_area_ratio']:.1f}% | "
                     f"{r['max_confidence']:.2f} | `{r['path']}` |")
        L.append("")
    L.append("## Raw data")
    L.append("")
    L.append("Per-image results: `dataset/pii_scan_report.jsonl` (one JSON line "
             "per scanned image, with bbox and confidence).")
    return "\n".join(L)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path,
                    default=Path("dataset/manifest.jsonl"))
    ap.add_argument("--out-md", type=Path,
                    default=Path("dataset/PII_SCAN_REPORT.md"))
    ap.add_argument("--out-jsonl", type=Path,
                    default=Path("dataset/pii_scan_report.jsonl"))
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true",
                       help="Scan EVERY image (expensive; ~$50 for 33k)")
    group.add_argument("--subcategory", type=str,
                       help="Scan just one subcategory (e.g. pos:humans_animals)")
    ap.add_argument("--sample-rate", type=float, default=0.05,
                    help="For 'targeted' mode, fraction of other-subcategories "
                         "to sample (default 0.05 = 5%%)")
    ap.add_argument("--min-area-ratio", type=float, default=0.02,
                    help="Face area/image area threshold for 'significant' "
                         "(default 0.02 = 2%% of image)")
    ap.add_argument("--min-confidence", type=float, default=0.5,
                    help="Min face detection confidence for 'significant'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--parallelism", type=int, default=4,
                    help="Concurrent Vision API calls")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show selection + cost estimate; no API calls")
    args = ap.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    # Select rows to scan
    print(f"Loading manifest: {args.manifest}")
    manifest = [json.loads(l) for l in args.manifest.open("r", encoding="utf-8")
                if l.strip()]
    print(f"  total rows: {len(manifest):,}")

    mode = "all" if args.all else ("subcategory" if args.subcategory else "targeted")
    to_scan = pick_rows_to_scan(manifest, mode, args.subcategory,
                                 args.sample_rate, args.seed)
    print(f"  mode: {mode}")
    print(f"  rows selected for scan: {len(to_scan):,}")

    cost_est = 1.50 * len(to_scan) / 1000
    print(f"  estimated Vision API cost: ${cost_est:.2f}")

    if args.dry_run:
        print("\n[dry-run] no API calls made")
        # Show distribution of what would be scanned
        by_cat = Counter(f"{r['class']}:{r['subcategory']}" for r in to_scan)
        print("  breakdown:")
        for k, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
            print(f"    {k:40s}  {n:>6,}")
        return 0

    # Init Vision client
    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip('"')
    if not cred_path or not Path(cred_path).exists():
        print("ERROR: GOOGLE_APPLICATION_CREDENTIALS not set or file missing.",
              file=sys.stderr)
        return 2
    try:
        client, vision_mod = init_vision()
        print(f"  Vision client initialized")
    except Exception as e:
        print(f"ERROR: Vision init failed: {e}", file=sys.stderr)
        return 2

    # Count rows per subcategory (for report) BEFORE scanning
    cat_totals: Counter = Counter(f"{r['class']}:{r['subcategory']}" for r in to_scan)

    # Parallel scan
    results: list[dict] = []
    print(f"\nScanning ({args.parallelism} threads)...")
    done = 0
    with ThreadPoolExecutor(max_workers=args.parallelism) as ex:
        futs = {ex.submit(detect_faces, client, vision_mod, Path(r["path"])): r
                for r in to_scan}
        for fut in as_completed(futs):
            row = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"error": f"future:{type(e).__name__}:{str(e)[:80]}",
                       "n_faces": 0, "max_area_ratio": 0.0,
                       "max_confidence": 0.0, "largest_bbox": None}
            merged = {
                "id": row["id"], "class": row["class"],
                "subcategory": row["subcategory"], "path": row["path"],
                **res,
            }
            results.append(merged)
            done += 1
            if done % 200 == 0 or done == len(to_scan):
                n_face_so_far = sum(1 for r in results if r["n_faces"] > 0)
                print(f"  {done:,}/{len(to_scan):,} scanned; "
                      f"{n_face_so_far:,} with faces")

    # Write raw jsonl
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out_jsonl.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.out_jsonl)
    print(f"\nWrote: {args.out_jsonl}")

    # Compute summary stats
    faces_n = sum(1 for r in results if r["n_faces"] > 0)
    significant = [
        r for r in results
        if r["n_faces"] > 0
        and r["max_area_ratio"] >= args.min_area_ratio
        and r["max_confidence"] >= args.min_confidence
    ]

    # Per-subcategory breakdown
    by_subcat: dict[str, tuple[int, int]] = {}
    for key, total in cat_totals.items():
        n_face_cat = sum(1 for r in results
                         if f"{r['class']}:{r['subcategory']}" == key
                         and r["n_faces"] > 0)
        by_subcat[key] = (n_face_cat, total)

    # Build markdown report
    md = build_report_md(
        scanned_n=len(results),
        faces_n=faces_n,
        significant_n=len(significant),
        thresholds={"min_area_ratio": args.min_area_ratio,
                    "min_confidence": args.min_confidence},
        by_subcat=by_subcat,
        results_flagged=significant,
        total_manifest=len(manifest),
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    print(f"Wrote: {args.out_md}")

    # Summary to stdout
    print(f"\n=== Summary ===")
    print(f"  Scanned:             {len(results):,}")
    print(f"  With any face:       {faces_n:,} "
          f"({100*faces_n/max(len(results),1):.1f}%)")
    print(f"  Significant faces:   {len(significant):,} "
          f"(area>={100*args.min_area_ratio:.0f}% + conf>={args.min_confidence:.1f})")
    print(f"  Action required:     review the {len(significant):,} flagged images "
          f"in {args.out_jsonl} (filter by n_faces>0 and area>={args.min_area_ratio:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
