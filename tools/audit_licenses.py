#!/usr/bin/env python3
"""audit_licenses.py — source-level licensing analysis of the dataset.

Reads dataset/manifest.jsonl, groups rows by source, classifies each source's
commercial-use risk, and emits:
  - dataset/LICENSE_AUDIT.md    (human-readable report)
  - dataset/licenses_per_source.json  (raw breakdown)

Also emits a filtered manifest containing ONLY legally-safe-for-commercial-use
rows (if --emit-safe is passed), for use in the production training run.

Classifications (based on each source's published terms as of 2025-2026):
  SAFE       — Free commercial use, no attribution legally required
  SAFE_ATTR  — Free commercial use, attribution required (CC-BY etc.)
  RISKY      — Scraped from search engines / social / third-party; copyright
               unverified. Legal advises AGAINST commercial use without removal
               or per-image clearance.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════
# License classification table
# ══════════════════════════════════════════════════════════════════════
# Risk tier, short rationale, required attribution (or None), license URL
LICENSE_TABLE: dict[str, tuple[str, str, str | None, str]] = {
    "pexels": (
        "SAFE",
        "Pexels License: free for commercial use, no attribution required",
        None,
        "https://www.pexels.com/license/",
    ),
    "unsplash": (
        "SAFE",
        "Unsplash License: free commercial use (excludes 'Unsplash+' tier)",
        None,
        "https://unsplash.com/license",
    ),
    "pixabay": (
        "SAFE",
        "Pixabay Content License: free for commercial use, no attribution",
        None,
        "https://pixabay.com/service/license-summary/",
    ),
    "wikimedia": (
        "SAFE_ATTR",
        "CC-BY / CC-BY-SA / PD variants — attribution required per file",
        "See per-row 'extra.license' field",
        "https://commons.wikimedia.org/wiki/Commons:Licensing",
    ),
    "flickr": (
        "SAFE_ATTR",
        "CC-BY licenses requested at scrape time — verify per-image",
        "See per-row 'extra.license' field",
        "https://www.flickr.com/help/guidelines",
    ),
    "reddit": (
        "RISKY",
        "Reddit posts retain uploader copyright; terms permit Reddit to use "
        "but not third parties",
        None,
        "https://www.redditinc.com/policies/user-agreement",
    ),
    "pw_google": (
        "RISKY",
        "Scraped via Google Images — images belong to third-party sources. "
        "Copyright unverified; commercial use not cleared.",
        None,
        "https://policies.google.com/terms",
    ),
    "pw_bing": (
        "RISKY",
        "Scraped via Bing Images — same as Google. Third-party copyright.",
        None,
        "https://www.microsoft.com/en-us/legal/terms-of-use",
    ),
    "pw_ddg": (
        "RISKY",
        "Scraped via DuckDuckGo Images — third-party sources, unverified.",
        None,
        "https://duckduckgo.com/terms",
    ),
    "pw_pinterest": (
        "RISKY",
        "Pinterest pins retain original uploader copyright. Pinterest TOS "
        "prohibits commercial scraping.",
        None,
        "https://policy.pinterest.com/en/terms-of-service",
    ),
    "pw_houzz": (
        "RISKY",
        "Houzz photos copyright by original companies / photographers. TOS "
        "prohibits unauthorized use.",
        None,
        "https://www.houzz.com/terms",
    ),
    "company_sites": (
        "RISKY",
        "Direct scrapes from fence/staining company sites. Copyright held "
        "by each company.",
        None,
        "Varies per company",
    ),
}
DEFAULT_ROW = (
    "UNKNOWN",
    "Unknown source — treat as RISKY until verified",
    None,
    "",
)


def classify(source: str) -> tuple[str, str, str | None, str]:
    return LICENSE_TABLE.get(source, DEFAULT_ROW)


# ══════════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════════

def load_jsonl(p: Path) -> list[dict]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def write_json(obj, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(p)


def write_jsonl(rows: list[dict], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)


# ══════════════════════════════════════════════════════════════════════
# Markdown report builder
# ══════════════════════════════════════════════════════════════════════

def build_report_md(stats: dict, by_tier: dict, per_source: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Dataset License Audit")
    lines.append("")
    lines.append(f"_Generated: {stats['generated_at']}_  ")
    lines.append(f"_Manifest: `{stats['manifest_path']}` "
                 f"(SHA-256 prefix: `{stats['manifest_sha256'][:12]}…`)_")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(f"- **Total images**: {stats['total']:,}")
    lines.append(f"- **SAFE for commercial use**: "
                 f"{by_tier.get('SAFE', 0):,} "
                 f"({100*by_tier.get('SAFE', 0)/max(stats['total'],1):.1f}%)")
    lines.append(f"- **SAFE with required attribution**: "
                 f"{by_tier.get('SAFE_ATTR', 0):,} "
                 f"({100*by_tier.get('SAFE_ATTR', 0)/max(stats['total'],1):.1f}%)")
    lines.append(f"- **RISKY (do not use commercially without clearance)**: "
                 f"{by_tier.get('RISKY', 0):,} "
                 f"({100*by_tier.get('RISKY', 0)/max(stats['total'],1):.1f}%)")
    lines.append(f"- **UNKNOWN sources**: {by_tier.get('UNKNOWN', 0):,}")
    lines.append("")
    lines.append("## Recommended production subset")
    lines.append("")
    safe_total = by_tier.get("SAFE", 0) + by_tier.get("SAFE_ATTR", 0)
    lines.append(f"For the deployed client model, train on the "
                 f"**{safe_total:,} SAFE + SAFE_ATTR rows only**. ")
    lines.append(f"This excludes {by_tier.get('RISKY', 0):,} RISKY rows "
                 f"whose copyright is unverified.")
    lines.append("")
    lines.append("Run with `--emit-safe` to generate `dataset/manifest_safe.jsonl` "
                 "containing only the legally-clean subset.")
    lines.append("")
    lines.append("## Per-source breakdown")
    lines.append("")
    lines.append("| Source | Count | Tier | Rationale | License URL |")
    lines.append("|--------|-------|------|-----------|-------------|")
    for row in sorted(per_source, key=lambda r: -r["count"]):
        tier = row["tier"]
        badge = {"SAFE": "🟢", "SAFE_ATTR": "🟡", "RISKY": "🔴",
                 "UNKNOWN": "⚪"}.get(tier, "")
        lines.append(f"| `{row['source']}` | {row['count']:,} | "
                     f"{badge} **{tier}** | {row['rationale']} | "
                     f"[link]({row['license_url']}) |")
    lines.append("")
    lines.append("## Attribution plan (for SAFE_ATTR sources)")
    lines.append("")
    lines.append("Each image from a `SAFE_ATTR` source carries its original "
                 "attribution metadata in the `extra.license`, `extra.owner`, "
                 "or `extra.photographer` field of the manifest row.")
    lines.append("")
    lines.append("At deployment time, generate an `ATTRIBUTION.md` by iterating "
                 "over the training manifest and listing unique (author, license) "
                 "pairs for all SAFE_ATTR images used.")
    lines.append("")
    lines.append("## Risks & legal notes")
    lines.append("")
    lines.append("- **Web-scraped sources** (pw_google, pw_bing, pw_pinterest, "
                 "pw_houzz, company_sites) DO NOT clear copyright. Use only for "
                 "internal R&D or clear per-image before deployment.")
    lines.append("- **Fair use / research exemptions** (US) may apply for "
                 "model training only, not deployment — consult client's legal "
                 "counsel for specifics.")
    lines.append("- **EU AI Act (2024)**: for high-risk systems, training data "
                 "provenance may be auditable. Maintain this report with the "
                 "model deliverable.")
    lines.append("- **GDPR**: if any images contain identifiable EU individuals, "
                 "run `tools/scan_pii.py` and address before deployment.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"_This report is a starting point; final legal sign-off "
                 f"must come from the client's counsel._")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path,
                    default=Path("dataset/manifest.jsonl"))
    ap.add_argument("--out-md", type=Path,
                    default=Path("dataset/LICENSE_AUDIT.md"))
    ap.add_argument("--out-json", type=Path,
                    default=Path("dataset/licenses_per_source.json"))
    ap.add_argument("--emit-safe", action="store_true",
                    help="Also write dataset/manifest_safe.jsonl containing "
                         "only SAFE + SAFE_ATTR rows")
    args = ap.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}")
        return 2

    # Compute manifest SHA for audit traceability
    import hashlib
    h = hashlib.sha256()
    with args.manifest.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    manifest_sha = h.hexdigest()

    rows = load_jsonl(args.manifest)
    print(f"Loaded {len(rows):,} rows from {args.manifest}")

    # Count by source + annotate tier
    source_counts: Counter[str] = Counter()
    for r in rows:
        source_counts[r.get("source", "unknown")] += 1

    per_source_records: list[dict] = []
    for src, cnt in source_counts.items():
        tier, rationale, attr_field, url = classify(src)
        per_source_records.append({
            "source": src, "count": cnt, "tier": tier,
            "rationale": rationale, "attribution_field": attr_field,
            "license_url": url,
        })

    by_tier = Counter(r["tier"] for r in per_source_records
                      for _ in range(r["count"]))

    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest_path": str(args.manifest),
        "manifest_sha256": manifest_sha,
        "total": len(rows),
    }

    # Write markdown
    md = build_report_md(stats, by_tier, per_source_records)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    print(f"  wrote: {args.out_md}")

    # Write JSON sibling
    write_json({
        "stats": stats,
        "by_tier": dict(by_tier),
        "per_source": per_source_records,
    }, args.out_json)
    print(f"  wrote: {args.out_json}")

    # Emit the safe subset manifest
    if args.emit_safe:
        safe_sources = {r["source"] for r in per_source_records
                        if r["tier"] in ("SAFE", "SAFE_ATTR")}
        safe_rows = [r for r in rows if r.get("source") in safe_sources]
        safe_path = args.manifest.with_name("manifest_safe.jsonl")
        write_jsonl(safe_rows, safe_path)
        print(f"  wrote: {safe_path}  ({len(safe_rows):,} rows, "
              f"{len(rows) - len(safe_rows):,} risky rows excluded)")

    print("\n=== Summary ===")
    for tier in ["SAFE", "SAFE_ATTR", "RISKY", "UNKNOWN"]:
        n = by_tier.get(tier, 0)
        if n:
            print(f"  {tier:10s}  {n:>7,}  ({100*n/max(len(rows),1):.1f}%)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
