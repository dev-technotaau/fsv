#!/usr/bin/env python3
"""standardize_images.py — add resolution_tier field + emit HQ subset manifests.

PHASE 1 OF TRAINING (all 33k at 512 input):   use manifest.jsonl + splits/*.jsonl
PHASE 2 OF TRAINING (HQ subset at 1024):      use manifest_hq.jsonl + splits/*_hq.jsonl

Actions:
  1. Compute `resolution_tier` for every row from its width/height
     - ULTRA    shorter edge >= 2048
     - HD       shorter edge 1536-2047
     - STANDARD shorter edge 1024-1535
     - LOW      shorter edge 800-1023
  2. Write updated manifest.jsonl + splits/*.jsonl in-place (atomic, with .bak)
  3. Emit HQ subset (default threshold >= 1024 shorter edge):
       dataset/manifest_hq.jsonl
       dataset/splits/train_hq.jsonl
       dataset/splits/val_hq.jsonl
       dataset/splits/test_hq.jsonl
  4. Write dataset/RESOLUTION_REPORT.md + dataset/resolution_report.json

Optional:
  --verify-files  : decode each image via PIL (catches corruption beyond what
                    prepare_dataset.py already verified). Parallel; ~2 min on 33k.

Usage:
    python tools/standardize_images.py                    # tier + HQ subset
    python tools/standardize_images.py --hq-threshold 1280
    python tools/standardize_images.py --verify-files     # also decode-check
    python tools/standardize_images.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════
# Tier classification
# ══════════════════════════════════════════════════════════════════════

TIER_THRESHOLDS = [
    ("ULTRA",    2048),
    ("HD",       1536),
    ("STANDARD", 1024),
    ("LOW",      0),
]


def tier_of(width: int, height: int) -> str:
    se = min(width or 0, height or 0)
    for name, threshold in TIER_THRESHOLDS:
        if se >= threshold:
            return name
    return "LOW"


# ══════════════════════════════════════════════════════════════════════
# I/O (atomic, handles read-only test.jsonl)
# ══════════════════════════════════════════════════════════════════════

def load_jsonl(p: Path) -> list[dict]:
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl_atomic(rows: list[dict], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    was_readonly = p.exists() and not os.access(p, os.W_OK)
    if was_readonly:
        p.chmod(stat.S_IWRITE | stat.S_IREAD)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)
    if was_readonly:
        p.chmod(0o444)


def backup(p: Path) -> None:
    if p.exists() and not p.with_suffix(p.suffix + ".bak").exists():
        shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))


# ══════════════════════════════════════════════════════════════════════
# Optional file verification (parallel)
# ══════════════════════════════════════════════════════════════════════

def _verify_worker(path_str: str) -> tuple[str, str | None]:
    """Return (path, None) if OK, (path, reason) if issue. Never raises."""
    try:
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = False
    except ImportError:
        return (path_str, "pil-missing")
    p = Path(path_str)
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            im.load()
            mode = im.mode
            exif = im.getexif() if hasattr(im, "getexif") else {}
        issues: list[str] = []
        if mode not in ("RGB", "L"):
            issues.append(f"mode={mode}")
        if exif and any(exif.values()):
            issues.append("has-exif")
        return (path_str, ";".join(issues) if issues else None)
    except Exception as e:
        return (path_str, f"decode:{type(e).__name__}")


def verify_files(paths: list[Path], workers: int) -> dict[str, str]:
    """Returns {path → issue_string} for images that failed verification."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    issues: dict[str, str] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_verify_worker, str(p)): p for p in paths}
        for fut in as_completed(futs):
            path_str, issue = fut.result()
            if issue:
                issues[path_str] = issue
            done += 1
            if done % 2000 == 0:
                print(f"  verified {done:,}/{len(paths):,}...")
    return issues


# ══════════════════════════════════════════════════════════════════════
# Reports
# ══════════════════════════════════════════════════════════════════════

def build_report_md(tiers: Counter, by_class: dict[str, Counter],
                    cum: list[dict], hq_threshold: int,
                    hq_counts: dict, total: int) -> str:
    L: list[str] = []
    L.append("# Dataset Resolution Report")
    L.append("")
    L.append(f"_Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_")
    L.append("")
    L.append(f"**Total images**: {total:,}  ")
    L.append(f"**HQ threshold**: shorter-edge ≥ {hq_threshold}px")
    L.append("")
    L.append("## Tier distribution")
    L.append("")
    L.append("| Tier | Shorter edge | Count | % | Pos | Neg |")
    L.append("|------|--------------|-------|---|-----|-----|")
    descriptions = {
        "ULTRA":    ">= 2048",
        "HD":       "1536-2047",
        "STANDARD": "1024-1535",
        "LOW":      "800-1023",
    }
    for t in ["ULTRA", "HD", "STANDARD", "LOW"]:
        n = tiers[t]
        p = by_class["pos"][t]
        ng = by_class["neg"][t]
        L.append(f"| **{t}** | {descriptions[t]} | {n:,} | "
                 f"{100*n/max(total,1):.1f}% | {p:,} | {ng:,} |")
    L.append("")
    L.append("## Cumulative — images retained at each threshold")
    L.append("")
    L.append("| Shorter edge ≥ | Count | % of total | Pos | Neg |")
    L.append("|----------------|-------|-----------|-----|-----|")
    for row in cum:
        L.append(f"| {row['threshold']}px | {row['count']:,} | "
                 f"{row['pct']:.1f}% | {row['pos']:,} | {row['neg']:,} |")
    L.append("")
    L.append("## HQ subset (for Phase 2 fine-tune)")
    L.append("")
    L.append(f"- **Threshold**: shorter-edge ≥ {hq_threshold}px")
    L.append(f"- **HQ total**: {hq_counts['all']:,} images "
             f"({100*hq_counts['all']/max(total,1):.1f}% of full set)")
    L.append(f"  - pos: {hq_counts['pos']:,}")
    L.append(f"  - neg: {hq_counts['neg']:,}")
    L.append("- **HQ manifest**: `dataset/manifest_hq.jsonl`")
    L.append("- **HQ splits**: `dataset/splits/{train,val,test}_hq.jsonl`")
    L.append("")
    L.append("## Training strategy (Option C)")
    L.append("")
    L.append("1. **Phase 1 (pretrain)**: train at 512×512 on all "
             f"{total:,} images using `manifest.jsonl` + `splits/{{train,val}}.jsonl`")
    L.append(f"2. **Phase 2 (finetune)**: fine-tune at 1024×1024 on HQ subset "
             f"({hq_counts['all']:,} images) using `manifest_hq.jsonl` + "
             f"`splits/{{train,val}}_hq.jsonl`")
    L.append("3. **Final eval**: run test_hq.jsonl at 1024 input for deployment "
             "sign-off metric")
    L.append("")
    L.append("No upscaling is performed — low-resolution images contribute at "
             "their native size to Phase 1 and are simply excluded from Phase 2.")
    return "\n".join(L)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path,
                    default=Path("dataset/manifest.jsonl"))
    ap.add_argument("--splits-dir", type=Path,
                    default=Path("dataset/splits"))
    ap.add_argument("--hq-threshold", type=int, default=1024,
                    help="Images with shorter edge >= this land in the HQ subset "
                         "(default 1024)")
    ap.add_argument("--verify-files", action="store_true",
                    help="Decode every image via PIL (parallel). Catches "
                         "corruption beyond prepare_dataset.py's integrity check.")
    ap.add_argument("--workers", type=int,
                    default=max(4, (os.cpu_count() or 8) - 2))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    print(f"Loading: {args.manifest}")
    manifest = load_jsonl(args.manifest)
    print(f"  rows: {len(manifest):,}")

    # ── Tier every row ─────────────────────────────────────────────────
    for r in manifest:
        r["resolution_tier"] = tier_of(r.get("width", 0), r.get("height", 0))

    # ── Compute distributions ─────────────────────────────────────────
    tiers: Counter = Counter(r["resolution_tier"] for r in manifest)
    by_class: dict[str, Counter] = {
        "pos": Counter(r["resolution_tier"] for r in manifest if r["class"] == "pos"),
        "neg": Counter(r["resolution_tier"] for r in manifest if r["class"] == "neg"),
    }
    cum: list[dict] = []
    for thr in [1024, 1280, 1536, 1792, 2048, 2560, 3072]:
        subset = [r for r in manifest
                  if min(r.get("width", 0), r.get("height", 0)) >= thr]
        cum.append({
            "threshold": thr,
            "count": len(subset),
            "pct": round(100 * len(subset) / max(len(manifest), 1), 2),
            "pos": sum(1 for r in subset if r["class"] == "pos"),
            "neg": sum(1 for r in subset if r["class"] == "neg"),
        })

    # ── HQ subset at the requested threshold ──────────────────────────
    hq_rows = [r for r in manifest
               if min(r.get("width", 0), r.get("height", 0)) >= args.hq_threshold]
    hq_ids = {r["id"] for r in hq_rows}
    hq_counts = {
        "all": len(hq_rows),
        "pos": sum(1 for r in hq_rows if r["class"] == "pos"),
        "neg": sum(1 for r in hq_rows if r["class"] == "neg"),
    }

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\nTier distribution:")
    for t in ["ULTRA", "HD", "STANDARD", "LOW"]:
        print(f"  {t:10s}  {tiers[t]:>6,}  "
              f"(pos={by_class['pos'][t]:,} / neg={by_class['neg'][t]:,})")
    print(f"\nHQ subset (>= {args.hq_threshold}px shorter edge): "
          f"{hq_counts['all']:,} images  "
          f"(pos={hq_counts['pos']:,} / neg={hq_counts['neg']:,})")

    # ── Optional file verification ────────────────────────────────────
    verify_issues: dict[str, str] = {}
    if args.verify_files:
        print(f"\nVerifying {len(manifest):,} files via PIL "
              f"({args.workers} workers)...")
        paths = [Path(r["path"]) for r in manifest if r.get("path")]
        verify_issues = verify_files(paths, args.workers)
        print(f"  issues: {len(verify_issues)}")
        if verify_issues:
            for p, iss in list(verify_issues.items())[:5]:
                print(f"    - {p}: {iss}")
            if len(verify_issues) > 5:
                print(f"    ... and {len(verify_issues)-5} more")

    if args.dry_run:
        print("\n[dry-run] no files written")
        return 0

    # ── Write full manifest + splits with new field ───────────────────
    print(f"\nWriting full manifest + splits with 'resolution_tier' field...")
    backup(args.manifest)
    write_jsonl_atomic(manifest, args.manifest)
    print(f"  {args.manifest}  (backup: {args.manifest.with_suffix('.jsonl.bak').name})")

    for split_name in ["train", "val", "test"]:
        p = args.splits_dir / f"{split_name}.jsonl"
        if not p.exists():
            continue
        rows = load_jsonl(p)
        by_id = {r["id"]: r for r in manifest}
        for r in rows:
            src = by_id.get(r["id"])
            if src:
                r["resolution_tier"] = src["resolution_tier"]
        backup(p)
        write_jsonl_atomic(rows, p)
        print(f"  {p}")

    # ── Write HQ subset manifest + HQ split files ─────────────────────
    hq_manifest_path = args.manifest.with_name("manifest_hq.jsonl")
    write_jsonl_atomic(hq_rows, hq_manifest_path)
    print(f"  {hq_manifest_path}  ({len(hq_rows):,} rows)")

    for split_name in ["train", "val", "test"]:
        p = args.splits_dir / f"{split_name}.jsonl"
        if not p.exists():
            continue
        rows = load_jsonl(p)
        hq_rows_split = [r for r in rows if r["id"] in hq_ids]
        out_hq = args.splits_dir / f"{split_name}_hq.jsonl"
        write_jsonl_atomic(hq_rows_split, out_hq)
        if split_name == "test":
            # Lock test_hq read-only too
            out_hq.chmod(0o444)
        print(f"  {out_hq}  ({len(hq_rows_split):,} rows"
              f"{'  [read-only]' if split_name == 'test' else ''})")

    # ── Write reports ─────────────────────────────────────────────────
    report_md = build_report_md(tiers, by_class, cum, args.hq_threshold,
                                 hq_counts, len(manifest))
    (args.manifest.parent / "RESOLUTION_REPORT.md").write_text(
        report_md, encoding="utf-8")
    print(f"  {args.manifest.parent / 'RESOLUTION_REPORT.md'}")

    report_json = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total": len(manifest),
        "tiers": dict(tiers),
        "by_class": {k: dict(v) for k, v in by_class.items()},
        "cumulative": cum,
        "hq_threshold": args.hq_threshold,
        "hq_counts": hq_counts,
        "verify_files_issues": verify_issues if args.verify_files else None,
    }
    (args.manifest.parent / "resolution_report.json").write_text(
        json.dumps(report_json, indent=2), encoding="utf-8")
    print(f"  {args.manifest.parent / 'resolution_report.json'}")

    print(f"\nDone. For Phase 1 training use manifest.jsonl + splits/*.jsonl.")
    print(f"     For Phase 2 fine-tune use manifest_hq.jsonl + splits/*_hq.jsonl.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
