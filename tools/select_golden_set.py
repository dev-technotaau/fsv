#!/usr/bin/env python3
"""select_golden_set.py — Pick a small, diverse, hand-mask-able subset.

Goal: produce a "golden" benchmark of N images (default 100) spanning all
major subcategories, drawn exclusively from the TEST split (so the golden
set is held out from training + is representative of deployment inputs).

The golden set is then hand-masked by a senior annotator (or the project lead)
to serve as:
  1. Ground-truth benchmark for Gemini auto-label quality (IoU)
  2. Calibration set for inter-annotator agreement
  3. Regression test during model development
  4. Final deployment sign-off metric

Outputs:
  dataset/golden_set/manifest.jsonl       — the selected rows
  dataset/golden_set/GOLDEN_SET_README.md — how to use it
  dataset/golden_set/images/              — (optional with --copy) copies of the images
  dataset/golden_set/masks/               — (empty dir; reviewer fills with PNGs)

Selection is STRATIFIED + SEEDED:
  - Draws from test.jsonl only (never train/val)
  - Proportional across subcategories (clamp to at least 1 per top-30 subcat)
  - Seed-controlled so rerunning produces the same golden set

Usage:
    python tools/select_golden_set.py                    # 100 images, test split
    python tools/select_golden_set.py --n 150            # bigger set
    python tools/select_golden_set.py --copy             # also copy image files
    python tools/select_golden_set.py --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def load_jsonl(p: Path) -> list[dict]:
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl_atomic(rows: list[dict], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)


def stable_hash_seed(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:4], "big")


# ══════════════════════════════════════════════════════════════════════
# Stratified selection
# ══════════════════════════════════════════════════════════════════════

def select_stratified(
    rows: list[dict],
    n_target: int,
    seed: int,
    pos_only: bool,
    ensure_minority: bool,
) -> list[dict]:
    """Pick `n_target` rows stratified across (class, subcategory).

    - If pos_only=True, negatives are excluded (golden set is for fence mask
      quality; neg masks are trivially empty so less valuable for IAA).
    - If ensure_minority=True, EVERY subcategory with ≥1 test row gets at
      least 1 sample in the golden set (subject to n_target budget).
    """
    pool = rows
    if pos_only:
        pool = [r for r in rows if r.get("class") == "pos"]

    # Group by subcategory
    by_subcat: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_subcat[f"{r['class']}:{r['subcategory']}"].append(r)

    total = sum(len(v) for v in by_subcat.values())
    if total == 0:
        return []
    # Proportional target per subcategory
    targets: dict[str, int] = {}
    for key, items in by_subcat.items():
        raw = n_target * len(items) / total
        targets[key] = max(1, int(round(raw))) if ensure_minority else int(round(raw))

    # Adjust to exactly n_target total (add/trim from biggest bucket if off)
    diff = n_target - sum(targets.values())
    if diff != 0:
        sorted_keys = sorted(targets.keys(), key=lambda k: -len(by_subcat[k]))
        for key in sorted_keys:
            if diff == 0:
                break
            if diff > 0:
                targets[key] += 1
                diff -= 1
            else:
                if targets[key] > 1:
                    targets[key] -= 1
                    diff += 1

    # Seed-stable per-subcategory sampling
    selected: list[dict] = []
    for key in sorted(targets.keys()):
        items = by_subcat[key]
        k = min(len(items), targets[key])
        if k <= 0:
            continue
        rng = random.Random(seed ^ stable_hash_seed(key))
        items_sorted = sorted(items, key=lambda r: r["id"])
        selected.extend(rng.sample(items_sorted, k))

    # Deterministic final ordering
    selected.sort(key=lambda r: r["id"])
    return selected


# ══════════════════════════════════════════════════════════════════════
# Readme
# ══════════════════════════════════════════════════════════════════════

def build_readme(stats: dict) -> str:
    lines = []
    lines.append("# Golden Set — Hand-Masked Benchmark")
    lines.append("")
    lines.append(f"_Generated: {stats['generated_at']}_  ")
    lines.append(f"_Source split: `{stats['source_split']}` "
                 f"(SHA-256 prefix `{stats['source_sha256'][:12]}…`)_")
    lines.append("")
    lines.append("## What this is")
    lines.append("")
    lines.append(f"A curated subset of **{stats['n']} images** from the test split, "
                 f"stratified across {stats['n_subcats']} subcategories. "
                 f"These images need **pixel-perfect hand-drawn masks** "
                 f"by a senior annotator.")
    lines.append("")
    lines.append("Once masked, the golden set serves as:")
    lines.append("")
    lines.append("1. **Ground-truth benchmark** for Gemini auto-label quality — compute "
                 "mean IoU of auto-labels against golden masks. Threshold: > 0.70 "
                 "per-image mean.")
    lines.append("2. **Inter-annotator agreement (IAA) target** — when a second "
                 "annotator reviews these images, their IoU vs. the golden masks "
                 "measures calibration. Target: > 0.90.")
    lines.append("3. **Regression test during training** — compute model-vs-golden IoU "
                 "every epoch. If it drops on a version, something regressed.")
    lines.append("4. **Deployment sign-off** — final reported metric to client is "
                 "`test.jsonl` IoU; golden set is the sanity-floor that should "
                 "always exceed the test-set average.")
    lines.append("")
    lines.append("## Contents")
    lines.append("")
    lines.append("```")
    lines.append(f"golden_set/")
    lines.append(f"  manifest.jsonl       # the {stats['n']} selected rows")
    lines.append(f"  images/              # {'copies of source images' if stats['copied_images'] else 'NOT populated — use original paths from manifest'}")
    lines.append(f"  masks/               # REVIEWER FILLS IN — PNGs named <id>.png")
    lines.append(f"  GOLDEN_SET_README.md # this file")
    lines.append(f"  selection_info.json  # audit (seed, source hash, etc.)")
    lines.append("```")
    lines.append("")
    lines.append("## Subcategory distribution")
    lines.append("")
    lines.append("| Subcategory | Count |")
    lines.append("|-------------|-------|")
    for key, n in sorted(stats["distribution"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{key}` | {n} |")
    lines.append("")
    lines.append("## Mask file format")
    lines.append("")
    lines.append("- **Format**: 8-bit single-channel PNG")
    lines.append("- **Dimensions**: EXACTLY match source image dimensions")
    lines.append("- **Pixel values**: `0` = background (not fence), `255` = fence")
    lines.append("- **Naming**: `<manifest_id>.png` — e.g. `a1b2c3d4-....png`")
    lines.append("- **Annotation software**: CVAT, Label Studio, LabelMe, or any "
                 "tool that exports a PNG mask")
    lines.append("")
    lines.append("## Before you start masking")
    lines.append("")
    lines.append("1. Read [`dataset/ANNOTATION_GUIDELINES.md`](../ANNOTATION_GUIDELINES.md) "
                 "end-to-end")
    lines.append("2. Calibrate on 10 easy images first (clear cedar fences)")
    lines.append("3. Do a second pass on your first 10 — you'll see inconsistencies")
    lines.append("4. Then proceed with the remaining set")
    lines.append("5. Budget ~**3–5 hours** for 100 images at pixel-perfect quality")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- Seed: `{stats['seed']}`")
    lines.append(f"- Source split SHA-256: `{stats['source_sha256']}`")
    lines.append(f"- Re-generate (same set): `{stats['command']}`")
    lines.append("")
    lines.append("_Any changes to the source split → golden set becomes invalid _  ")
    lines.append("_(source hash mismatch) and should be regenerated + re-masked._")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=Path,
                    default=Path("dataset/splits/test.jsonl"),
                    help="Which split to draw from (default: test)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("dataset/golden_set"))
    ap.add_argument("--n", type=int, default=100,
                    help="Number of images to select (default 100)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-neg", action="store_true",
                    help="Also sample from negatives (default: pos only)")
    ap.add_argument("--no-ensure-minority", action="store_true",
                    help="Allow subcategories to have 0 in golden set. Default "
                         "is to include ≥1 of each.")
    ap.add_argument("--copy", action="store_true",
                    help="Copy selected image files into golden_set/images/. "
                         "Otherwise the manifest references their original paths.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.source.exists():
        print(f"ERROR: source not found: {args.source}", file=sys.stderr)
        return 2

    # Hash source for reproducibility audit
    h = hashlib.sha256()
    with args.source.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    src_sha = h.hexdigest()

    rows = load_jsonl(args.source)
    print(f"Loaded {len(rows):,} rows from {args.source}")

    selected = select_stratified(
        rows, args.n, args.seed,
        pos_only=(not args.include_neg),
        ensure_minority=(not args.no_ensure_minority),
    )
    print(f"Selected {len(selected)} for golden set")

    # Distribution report
    dist: dict[str, int] = defaultdict(int)
    for r in selected:
        dist[f"{r['class']}:{r['subcategory']}"] += 1
    print("\nDistribution:")
    for key, n in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f"  {key:42s}  {n:>3}")

    if args.dry_run:
        print("\n[dry-run] no files written")
        return 0

    # Overwrite protection
    manifest_path = args.out_dir / "manifest.jsonl"
    if manifest_path.exists() and not args.force:
        print(f"ERROR: {manifest_path} exists. Use --force.", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "images").mkdir(exist_ok=True)
    (args.out_dir / "masks").mkdir(exist_ok=True)

    # Write manifest
    write_jsonl_atomic(selected, manifest_path)
    print(f"  wrote: {manifest_path}")

    # Write selection audit info
    sel_info = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "seed": args.seed,
        "n": len(selected),
        "source_split": str(args.source),
        "source_sha256": src_sha,
        "include_neg": args.include_neg,
        "ensure_minority": not args.no_ensure_minority,
        "copied_images": args.copy,
        "n_subcats": len(dist),
        "distribution": dict(dist),
    }
    info_path = args.out_dir / "selection_info.json"
    info_path.write_text(json.dumps(sel_info, indent=2), encoding="utf-8")
    print(f"  wrote: {info_path}")

    # Readme
    readme_path = args.out_dir / "GOLDEN_SET_README.md"
    readme_path.write_text(build_readme(sel_info), encoding="utf-8")
    print(f"  wrote: {readme_path}")

    # Optionally copy image files
    if args.copy:
        n_copied = 0
        for r in selected:
            src = Path(r["path"])
            if not src.exists():
                print(f"  warn: source missing: {src}", file=sys.stderr)
                continue
            dst = args.out_dir / "images" / f"{r['id']}{src.suffix}"
            shutil.copy2(src, dst)
            n_copied += 1
        print(f"  copied {n_copied} image files → {args.out_dir/'images'}")

    print(f"\nNEXT STEP: senior annotator draws per-pixel masks in "
          f"{args.out_dir}/masks/, one PNG per manifest row named <id>.png.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
