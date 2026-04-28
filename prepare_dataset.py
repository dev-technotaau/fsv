#!/usr/bin/env python3
"""prepare_dataset.py — Phase 1+2 of the data pipeline.

Validates every image in data_scraped/ (positives) and data_scraped_neg/
(negatives), cross-checks for SHA duplicates between the two sets (must be
disjoint), classifies each image into a subcategory, and emits a unified
manifest.jsonl ready for the auto-label step.

Robust features:
  • Parallel PIL integrity check (catches truncated / corrupt JPEGs)
  • Orphan detection: metadata-without-file AND file-without-metadata
  • Cross-set SHA-dedup: any image in both pos and neg → removed from neg
  • Subcategory classification from the `query` text (18 pos + 10 neg cats)
  • Stable UUIDs per image for downstream tracking
  • --dry-run mode for audit-before-apply
  • Atomic writes (temp file + rename) — no half-finished outputs on Ctrl-C
  • Structured JSON reports: integrity.json, removed.jsonl
  • Idempotent: safe to re-run after fixes

Usage:
    python prepare_dataset.py                       # normal run (with changes)
    python prepare_dataset.py --dry-run             # report only, no changes
    python prepare_dataset.py --workers 32          # parallelism
    python prepare_dataset.py --out-dir dataset/    # output directory
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Optional

try:
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = False   # DON'T silently accept truncation
except ImportError:
    print("ERROR: Pillow required. pip install Pillow", file=sys.stderr)
    sys.exit(2)

# ══════════════════════════════════════════════════════════════════════
# SUBCATEGORY TAXONOMY (ordered — first match wins)
# ══════════════════════════════════════════════════════════════════════

_KW = lambda *kw: tuple(k.lower() for k in kw)

POS_SUBCATS: list[tuple[tuple[str, ...], str]] = [
    (_KW("blur", "bokeh", "tilt shift", "panning", "shimmer", "out of focus"), "motion_focus"),
    (_KW("backlit", "silhouette", "night", "twilight", "moonlight", "dappled",
         "contre-jour", "flash photography", "rim light", "golden hour",
         "harsh shadow", "dramatic shadow", "blue hour"), "lighting"),
    (_KW("worms eye", "aerial", "drone", "oblique", "fisheye",
         "wide angle distorted", "receding horizon", "low angle",
         "telephoto compression", "rooftop above"), "angle"),
    (_KW("behind dense", "hidden", "obscured", "covered", "glimpsed through",
         "barely visible", "buried behind", "nearly buried", "mostly obscured",
         "overgrown"), "occlusion"),
    (_KW("stain drips", "half stained", "stain colors", "brush strokes",
         "paint tape", "power wash", "stain samples", "sanded",
         "polyurethane", "restain", "touching up", "post-stain"), "staining_process"),
    (_KW("broken", "rotten", "damaged", "demolished", "missing several boards",
         "blown over", "splitting", "cracking", "under repair", "construction",
         "storm", "partially finished", "half demolished", "hurricane",
         "tornado", "post holes", "panels stacked", "replacement new"), "damaged_construction"),
    (_KW("painted", "peeling paint", "chipped paint", "graffiti", "mural",
         "bleached", "primary colors", "white painted", "black painted",
         "red painted", "blue painted", "green painted", "grey painted"), "painted_color"),
    (_KW("moss", "algae", "lichen", "mushrooms growing", "mildew"), "biological_growth"),
    (_KW("reflected", "reflection", "puddle", "glistening", "flooded",
         "rain streaks", "water droplets", "water macro", "wet"), "reflection_water"),
    (_KW("meeting grass", "meeting brick", "meeting pool", "meeting stone",
         "base concrete", "above gravel", "base river rocks",
         "edge wood chips", "asphalt driveway edge", "mulched garden bed",
         "stone pavers", "pool deck wood transition"), "boundary_transition"),
    (_KW("hillside", "stepping down terrain", "on top retaining wall",
         "sloped driveway", "curving arc", "wrapping pool", "arched top",
         "degree corner", "135 degree"), "slope_curved"),
    (_KW("alley", "abandoned", "run down", "dumpsters", "industrial grunge",
         "garbage cans", "construction site", "junk yard"), "urban_rundown"),
    (_KW("portrait with", "wedding photo", "child playing with", "pet dog with",
         "family photo", "engagement photo"), "incidental_background"),
    (_KW("crowded urban", "cluttered", "chaotic yard", "busy garden",
         "scattered tools", "multiple outdoor structures", "farmers market",
         "workshop lumber", "greenhouse tomato", "community garden",
         "fire pit party"), "complex_background"),
    (_KW("macro extreme", "extreme close-up", "telephoto", "panorama",
         "wide panoramic", "tiny small part", "filling entire frame",
         "barely visible wide", "magnified nail"), "scale_extreme"),
    (_KW("with shed pergola", "log cabin exterior", "two outbuildings",
         "deck railing same wood", "pergola and shed", "retaining wall",
         "gazebo shed", "trellis arbor pergola", "driveway gate pergola"), "multi_structure"),
    (_KW("snow drift", "torrential rain", "ice storm", "hailstorm",
         "dense thick fog", "windstorm leaning", "wet after rain",
         "spring thaw", "covered frost", "monsoon rain"), "weather_extreme"),
    (_KW("two cedar fences meeting", "parallel wooden fences",
         "foreground different fence", "transitioning to vinyl",
         "different heights meeting", "curving around property",
         "90 degree corner", "meeting stone retaining", "gate integrated",
         "driveway meeting property line"), "multi_fence_corner"),

    # ══════════════════════════════════════════════════════════════════
    # ORIGINAL-CORPUS categories — catch the first 17k general positives
    # (use_static + custom + Gemini queries from the initial scrape).
    # Hard-positive categories above take priority via first-match-wins,
    # so specific occluded/damaged/lit shots stay in their proper bucket;
    # only the plain "cedar fence", "backyard fence" style queries land here.
    # ══════════════════════════════════════════════════════════════════
    (_KW(" dog ", " cat ", " bird ", "squirrel", "chickens", "rabbit",
         "horse behind", "person painting", "person staining",
         "person installing", "kid playing", "person gardening",
         "pet in fenced", "animals", "with pet", "puppy", "kitten",
         "livestock", "farm animals", "child near", "children",
         "family in", "kids in backyard"), "humans_animals"),
    (_KW("tree branches in front", "behind bushes", "climbing vines",
         "overhanging", "shrubs in front", "behind tall grass",
         "covered in ivy", "flower bushes", "rose bushes in front",
         "hidden by plants", "hedge in front", "tree trunk in front",
         "garden vegetables in front", "potted plants", "bird feeder",
         "hanging planters", "hose coiled", "garden tools leaning",
         "ladder leaning", "bicycle parked", "trash bin",
         "grill nearby", "snow on branches", "tall weeds"), "occlusion_mild"),
    (_KW("next to", "distractor", "similar material", "shed next",
         "house siding near", "pergola with", "telephone pole next",
         "gazebo in yard", "deck with", "playhouse", "trellis next",
         "arbor with", "log cabin with", "stairs next",
         "utility pole with", "bench against", "raised planter near",
         "retaining wall with"), "multi_structure"),  # catches DISTRACTORS
    (_KW("golden hour", "sunset light", "morning dew", "overcast day",
         "in rain", "after rain wet", "with frost", "with snow",
         "in fog", "midday shadow", "dappled shade", "dramatic shadows",
         "twilight", "blue hour"), "lighting"),      # catches CONDITIONS
    (_KW("broken fence panel", "rotting wooden fence", "old wooden fence",
         "fence post only", "fence panels stacked", "arbor gate",
         "double gate fence", "fence gate open", "fence gate closed",
         "fence around only part", "half finished fence",
         "missing board"), "damaged_construction"),  # catches VARIATIONS
    (_KW("close-up", "close up", "macro", "aerial view",
         "wide angle", "long fence perspective",
         "distant fence across", "fence corner view"), "scale_extreme"),  # catches SCALES
    (_KW("vinyl privacy fence", "white vinyl", "chain link fence",
         "wrought iron fence", "aluminum fence", "composite fence",
         "metal fence panels", "barbed wire fence",
         "wire mesh fence"), "style_nonwood"),       # catches NON_WOOD_STYLES
    (_KW("cedar fence", "cedar privacy", "cedar shadowbox",
         "cedar picket", "cedar stockade", "cedar horizontal",
         "cedar split rail", "cedar board on board", "cedar installation",
         "freshly stained cedar", "dark stained cedar", "natural cedar",
         "cedar weathered"), "style_cedar"),
    (_KW("wooden privacy", "wooden picket", "wooden shadowbox",
         "wooden horizontal", "wooden dog-ear", "wooden scalloped",
         "wooden lattice", "wooden fence panels", "redwood fence",
         "pressure treated", "pine wood fence", "bamboo fence",
         "rustic wooden", "weathered wood", "wooden fence"), "style_wood"),
    (_KW("backyard fence", "front yard fence", "garden fence",
         "suburban backyard", "residential fence", "fence with patio",
         "fence along sidewalk", "fence next to driveway",
         "fence with pergola", "fence with deck", "fence with pool",
         "fence around garden", "fence with raised garden",
         "fence with walkway", "fence property line",
         "fence bordering", "suburban home", "cottage fence",
         "farmhouse fence", "ranch fence", "neighborhood fence",
         "fence around yard", "fence surrounding", "fence around"), "scene_context"),

    # Final catch-all for fence queries that didn't match anything specific
    # (mostly Gemini-expansion queries with unique phrasings). Catches the big
    # "general_positive" bucket and gives it a meaningful subcategory.
    (_KW("wooden", " wood "), "fence_general_wood"),
    (_KW("fence"), "fence_general"),
]

NEG_SUBCATS: list[tuple[tuple[str, ...], str]] = [
    (_KW("railing", "handrail"), "neg_railing"),
    (_KW("pergola", "arbor", "trellis"), "neg_pergola_trellis"),
    (_KW("shake shingle siding", "siding exterior", "shingle wall",
         "tongue and groove", "barn wood", "log cabin wall",
         "log home interior"), "neg_siding"),
    (_KW("bamboo privacy", "reed wall"), "neg_bamboo_reed"),
    (_KW("room divider", "folding screen", "lattice panel decorative",
         "louvered exterior", "louvered door", "vent louver",
         "privacy screen outdoor"), "neg_divider_louver"),
    (_KW("shutters", "blinds"), "neg_shutter_blind"),
    (_KW("deck boards", "plank flooring", "ceiling beams", "herringbone",
         "reclaimed barn wood", "wall art vertical planks",
         "rustic wood plank wall", "gazebo"), "neg_wood_panel"),
    (_KW("crate stacked", "pallet stack", "firewood", "pallet wall art",
         "garden shed", "potting shed", "picnic table",
         "outdoor storage box", "raised garden bed", "awning slatted"), "neg_wood_yard_object"),
    (_KW("headboard", "chair slat", "chair back"), "neg_furniture_back"),
    (_KW("vinyl", "pvc", "chain link", "wrought iron", "aluminum",
         "steel tube fence", "wire mesh fence", "barbed wire",
         "electric fence", "composite vinyl", "metal slat fence",
         "glass pool fence", "plastic garden border", "post and rail metal",
         "temporary orange plastic", "snow fence plastic",
         "metal industrial", "iron rod fence", "galvanized steel mesh",
         "hog wire", "farm gate metal"), "neg_nonwood_fence"),
    (_KW("brick wall", "stone wall", "dry stone wall",
         "retaining wall stone", "retaining wall concrete", "concrete wall",
         "cinder block", "fieldstone", "limestone", "granite wall",
         "flagstone", "stucco", "adobe", "castle stone", "sea wall",
         "sound barrier", "boulder retaining", "modular block",
         "ivy covered", "concrete block wall", "pavestone",
         "gabion", "dry stack stone"), "neg_masonry"),
    (_KW("hedge", "cypress", "arborvitae", "bamboo grove",
         "row of", "cornfield", "corn field stalks", "sunflower field",
         "wheat field", "ornamental grass", "evergreen tree row",
         "birch tree grove", "prairie grass", "sugar cane",
         "sedge grass", "papyrus", "espalier", "cattail reeds",
         "fern wall"), "neg_natural"),
    (_KW("open suburban backyard", "open country field", "open lawn",
         "beachfront", "rural property open", "open pasture",
         "open land", "modern minimalist backyard", "open pool deck",
         "golf course green", "sports field", "open meadow",
         "public park", "outdoor patio garden", "backyard furniture no",
         "open landscape", "beach dunes", "countryside rolling",
         "open savanna", "outdoor basketball"), "neg_empty_outdoor"),
    (_KW("gate standalone", "garden gate closed", "rustic wooden gate",
         "arched wooden gate", "ornate wooden gate", "driveway gate",
         "barn door sliding", "carriage garage", "church door",
         "castle gate", "torii gate", "cellar door",
         "dutch door"), "neg_gate_door"),
    (_KW("pedestrian bridge", "footbridge", "boardwalk", "dock",
         "pier", "walkway garden", "trestle bridge", "covered bridge",
         "suspension bridge", "jetty", "rope bridge"), "neg_bridge_dock"),
    (_KW("mailbox post", "sign post", "flag pole", "utility pole",
         "lamp post", "clothesline post", "single fence post",
         "gate post alone", "sundial post", "weathervane"), "neg_isolated_post"),
    (_KW("adirondack", "park bench", "garden bench", "dining chair back",
         "slat outdoor lounger", "swing bench", "porch swing",
         "patio dining set", "folding chair beach", "slatted wood ottoman",
         "park bench row", "deck chair horizontal"), "neg_slatted_furniture"),
    (_KW("mountain", "beach", "forest hiking", "desert", " sunrise", " sunset",
         "cherry blossom", "wildflower", "lake", "waterfall", "coastline",
         "tropical island", "rainforest", "savanna", "tundra", "arctic",
         "city skyline", "downtown", "skyscraper", "cobblestone",
         "european old town", "subway", "airport", "train station",
         "highway road", "bridge suspension", "harbor", "marina",
         "modern kitchen", "kitchen interior", "living room",
         "bathroom", "home office", "restaurant dining", "cafe",
         "coffee shop", "library", "museum", "gym", "yoga studio",
         "hospital", "indoor photography", "landscape photography",
         "street photography", "nature photography", "autumn leaves",
         "winter landscape", "spring bloom", "ocean waves"), "neg_pure_random"),

    # Final negative catch-all for anything that didn't hit a specific bucket
    (_KW("outdoor", "scene", "landscape", "indoor"), "neg_general"),
]


def classify(query: str, class_: str) -> str:
    """Assign a subcategory to an image given its query text + class."""
    q = (query or "").lower()
    table = POS_SUBCATS if class_ == "pos" else NEG_SUBCATS
    for kws, label in table:
        if any(kw in q for kw in kws):
            return label
    # fallback — indicates query text didn't match any known subcategory
    return "general_positive" if class_ == "pos" else "negative_uncategorized"


# ══════════════════════════════════════════════════════════════════════
# INTEGRITY CHECK
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CheckResult:
    path: str
    ok: bool
    reason: str = ""
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    sha256: str = ""


def _verify_file(path_str: str) -> CheckResult:
    """Worker: open + decode image, return structured result. Never throws."""
    path = Path(path_str)
    try:
        size = path.stat().st_size
    except Exception as e:
        return CheckResult(path_str, False, f"stat:{type(e).__name__}")
    if size == 0:
        return CheckResult(path_str, False, "zero-bytes", size_bytes=0)

    # SHA of actual file bytes (don't trust metadata's sha — catches corruption)
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest()
    except Exception as e:
        return CheckResult(path_str, False, f"read:{type(e).__name__}", size_bytes=size)

    # PIL verify — catches truncation / bad JPEG
    try:
        with Image.open(path) as im:
            im.verify()
    except Exception as e:
        return CheckResult(path_str, False, f"pil-verify:{type(e).__name__}",
                           size_bytes=size, sha256=sha)

    # Actually decode to confirm pixels are readable (verify() is a shallow check)
    try:
        with Image.open(path) as im:
            im.load()
            w, h_ = im.size
    except Exception as e:
        return CheckResult(path_str, False, f"pil-load:{type(e).__name__}",
                           size_bytes=size, sha256=sha)

    return CheckResult(path_str, True, "", width=w, height=h_,
                       size_bytes=size, sha256=sha)


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def write_jsonl_atomic(rows: Iterator[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def iter_images(dir_: Path) -> Iterator[Path]:
    if not dir_.exists():
        return
    for p in dir_.iterdir():
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            yield p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pos-dir", type=Path, default=Path("data_scraped"),
                    help="Positive images root (contains images/ + metadata.jsonl)")
    ap.add_argument("--neg-dir", type=Path, default=Path("data_scraped_neg"),
                    help="Negative images root")
    ap.add_argument("--out-dir", type=Path, default=Path("dataset"),
                    help="Output dir for manifest + reports")
    ap.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) - 2),
                    help="Parallel workers for file integrity check")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report findings but don't delete files or write manifest")
    ap.add_argument("--reclassify-nonwood-as-neg", action="store_true",
                    help="Move pos:style_nonwood rows to neg:neg_nonwood_fence "
                         "(label-only — files stay where they are). Recommended "
                         "for cedar staining models where non-wood fences shouldn't "
                         "be treated as positives.")
    ap.add_argument("--subsample-pure-random", type=int, default=0,
                    help="Randomly keep only N rows from neg:neg_pure_random, "
                         "drop the rest (0 = no subsampling). Useful to rebalance "
                         "the negative set toward hard-negatives.")
    ap.add_argument("--subsample-seed", type=int, default=42,
                    help="Random seed for --subsample-pure-random (default 42, "
                         "deterministic across runs)")
    args = ap.parse_args()

    pos_images = args.pos_dir / "images"
    neg_images = args.neg_dir / "images"
    pos_meta = args.pos_dir / "metadata.jsonl"
    neg_meta = args.neg_dir / "metadata.jsonl"

    print(f"═══ Phase 1+2 Dataset Preparation ═══")
    print(f"  positives: {pos_images}  (metadata: {pos_meta.name})")
    print(f"  negatives: {neg_images}  (metadata: {neg_meta.name})")
    print(f"  out dir:   {args.out_dir}")
    print(f"  dry-run:   {args.dry_run}")
    print(f"  workers:   {args.workers}")
    print()

    # ── Load metadata ──────────────────────────────────────────────────
    pos_rows = load_jsonl(pos_meta)
    neg_rows = load_jsonl(neg_meta)
    print(f"[load] positives: {len(pos_rows):,} metadata rows")
    print(f"[load] negatives: {len(neg_rows):,} metadata rows")

    # Index metadata by filename for quick lookup
    pos_by_name = {Path(r["path"]).name: r for r in pos_rows if r.get("path")}
    neg_by_name = {Path(r["path"]).name: r for r in neg_rows if r.get("path")}

    # ── Scan images dir, detect orphans ────────────────────────────────
    pos_files = list(iter_images(pos_images))
    neg_files = list(iter_images(neg_images))
    print(f"[scan] positives on disk: {len(pos_files):,} files")
    print(f"[scan] negatives on disk: {len(neg_files):,} files")

    pos_orphan_files = [p for p in pos_files if p.name not in pos_by_name]
    neg_orphan_files = [p for p in neg_files if p.name not in neg_by_name]
    pos_orphan_meta = [n for n in pos_by_name if n not in {p.name for p in pos_files}]
    neg_orphan_meta = [n for n in neg_by_name if n not in {p.name for p in neg_files}]

    print(f"[orphan] pos files without metadata: {len(pos_orphan_files)}")
    print(f"[orphan] neg files without metadata: {len(neg_orphan_files)}")
    print(f"[orphan] pos metadata without file:  {len(pos_orphan_meta)}")
    print(f"[orphan] neg metadata without file:  {len(neg_orphan_meta)}")

    # ── Parallel integrity check ───────────────────────────────────────
    all_files = [(p, "pos") for p in pos_files] + [(p, "neg") for p in neg_files]
    print(f"\n[integrity] checking {len(all_files):,} files with {args.workers} workers…")
    results: dict[str, tuple[CheckResult, str]] = {}
    done = 0
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            fut_map = {ex.submit(_verify_file, str(p)): (p, cls) for p, cls in all_files}
            for fut in as_completed(fut_map):
                p, cls = fut_map[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = CheckResult(str(p), False, f"worker:{type(e).__name__}")
                results[str(p)] = (r, cls)
                done += 1
                if done % 1000 == 0 or done == len(all_files):
                    print(f"  {done:,}/{len(all_files):,} checked", flush=True)
    except KeyboardInterrupt:
        print("\n[abort] Ctrl-C — partial results discarded.")
        return 130

    bad = [r for (r, _) in results.values() if not r.ok]
    good = [(r, cls) for (r, cls) in results.values() if r.ok]
    print(f"[integrity] bad: {len(bad)}   good: {len(good):,}")

    # ── Cross-set SHA dedup: any POS that matches a neg by SHA → drop FROM POS ──
    # Analysis of actual cross-dupes showed these are almost always cases where
    # stock APIs served loose matches (pure-neg scenes returned for pos queries)
    # OR non-wood fences wrongly classified as positive. The neg classification
    # is almost always more accurate — so we delete from pos, keep in neg.
    neg_shas = {r.sha256 for (r, cls) in good if cls == "neg"}
    cross_dupes = [(r, cls) for (r, cls) in good
                   if cls == "pos" and r.sha256 in neg_shas]
    print(f"[cross-dedup] positives colliding with negatives (will remove from pos): "
          f"{len(cross_dupes)}")

    # ── Determine removals ─────────────────────────────────────────────
    to_remove: list[dict] = []
    for r in bad:
        to_remove.append({"path": r.path, "reason": f"integrity:{r.reason}"})
    for r, _ in cross_dupes:
        to_remove.append({"path": r.path, "reason": "cross-set-dup"})
    for p in pos_orphan_files:
        to_remove.append({"path": str(p), "reason": "orphan-file-pos"})
    for p in neg_orphan_files:
        to_remove.append({"path": str(p), "reason": "orphan-file-neg"})

    # ── Apply removals (unless dry-run) ────────────────────────────────
    actually_removed = 0
    if args.dry_run:
        print(f"\n[dry-run] would remove {len(to_remove)} files (no action taken)")
    else:
        print(f"\n[apply] removing {len(to_remove)} files…")
        removed_set = {x["path"] for x in to_remove}
        for path in removed_set:
            try:
                Path(path).unlink(missing_ok=True)
                actually_removed += 1
            except Exception as e:
                print(f"  warn: failed to remove {path}: {e}", file=sys.stderr)
        print(f"[apply] removed {actually_removed} files")

    # ── Build unified manifest ─────────────────────────────────────────
    # Strategy: start from the COMPLETE original metadata row, then add
    # provenance (id, class, subcategory) + override dims/sha with values
    # verified from disk during the integrity check. Every original field
    # (dhash, source-specific extras like flickr license, wikimedia license,
    # reddit subreddit/score, etc.) is preserved.
    manifest: list[dict] = []
    cross_dupe_paths = {r.path for r, _ in cross_dupes}
    for (r, cls) in good:
        if r.path in cross_dupe_paths:
            continue    # excluded (removed from pos — neg copy is the correct label)
        name = Path(r.path).name
        src_row = dict((pos_by_name if cls == "pos" else neg_by_name).get(name, {}))
        query = src_row.get("query") or ""
        subcat = classify(query, cls)

        # Preserve all original fields, then merge the new ones on top.
        row = dict(src_row)                      # full original metadata (path, source, query,
                                                 #   origin_url, origin_page, title, dhash, extra, …)
        # --- new provenance fields ---
        row["id"] = str(uuid.uuid4())
        row["class"] = cls                       # pos | neg
        row["subcategory"] = subcat
        # --- override with VERIFIED values from disk (integrity check) ---
        row["width"] = r.width
        row["height"] = r.height
        row["bytes"] = r.size_bytes
        row["sha256"] = r.sha256
        # --- flatten vision fields from extra for convenience (original kept in extra) ---
        extra = row.get("extra") or {}
        row["vision_label"] = extra.get("vision_label")
        row["vision_conf"] = extra.get("vision_conf")
        row["vision_checked"] = extra.get("vision_checked")

        manifest.append(row)

    # ── Optional post-transforms ───────────────────────────────────────
    transforms_report: dict = {}

    if args.reclassify_nonwood_as_neg:
        reclassed = 0
        for row in manifest:
            if row["class"] == "pos" and row["subcategory"] == "style_nonwood":
                row["class"] = "neg"
                row["subcategory"] = "neg_nonwood_fence"
                reclassed += 1
        print(f"[reclassify] moved {reclassed} style_nonwood rows from pos → neg")
        transforms_report["reclassified_nonwood_to_neg"] = reclassed

    if args.subsample_pure_random > 0:
        import random as _random
        rng = _random.Random(args.subsample_seed)
        pure_rows = [r for r in manifest if r["subcategory"] == "neg_pure_random"]
        if len(pure_rows) > args.subsample_pure_random:
            to_drop_ids = {r["id"] for r in rng.sample(
                pure_rows,
                len(pure_rows) - args.subsample_pure_random,
            )}
            before = len(manifest)
            manifest = [r for r in manifest if r["id"] not in to_drop_ids]
            dropped = before - len(manifest)
            print(f"[subsample] kept {args.subsample_pure_random} of "
                  f"{len(pure_rows)} neg_pure_random rows; dropped {dropped}")
            transforms_report["pure_random_kept"] = args.subsample_pure_random
            transforms_report["pure_random_dropped"] = dropped
            # Note: --dry-run leaves files in place; real run also leaves files
            # (subsampling is a manifest-level op, not a deletion). The dropped
            # files just aren't in the manifest → ignored by downstream pipeline.
        else:
            print(f"[subsample] neg_pure_random has only {len(pure_rows)} rows "
                  f"(≤ {args.subsample_pure_random}) — no subsampling applied")

    # Subcategory distribution — useful for verifying label balance
    subcat_counts: dict[str, int] = {}
    for m in manifest:
        key = f"{m['class']}:{m['subcategory']}"
        subcat_counts[key] = subcat_counts.get(key, 0) + 1

    # ── Write outputs ──────────────────────────────────────────────────
    report = {
        "pos_dir": str(pos_images),
        "neg_dir": str(neg_images),
        "total_files_scanned": len(all_files),
        "pos_files": len(pos_files),
        "neg_files": len(neg_files),
        "pos_metadata_rows": len(pos_rows),
        "neg_metadata_rows": len(neg_rows),
        "integrity_bad": len(bad),
        "integrity_good": len(good),
        "cross_set_duplicates": len(cross_dupes),
        "orphan_files_pos": len(pos_orphan_files),
        "orphan_files_neg": len(neg_orphan_files),
        "orphan_metadata_pos": len(pos_orphan_meta),
        "orphan_metadata_neg": len(neg_orphan_meta),
        "removed_total": len(to_remove),
        "removed_applied": actually_removed,
        "manifest_rows": len(manifest),
        "dry_run": args.dry_run,
        "transforms": transforms_report,
        "subcategory_distribution": dict(sorted(subcat_counts.items())),
    }

    if args.dry_run:
        out_report = args.out_dir / "integrity_DRYRUN.json"
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[dry-run] report → {out_report}")
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "integrity.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        write_jsonl_atomic(iter(to_remove), args.out_dir / "removed.jsonl")
        write_jsonl_atomic(iter(manifest), args.out_dir / "manifest.jsonl")
        print(f"\n[done] wrote:")
        print(f"  manifest   → {args.out_dir/'manifest.jsonl'}  ({len(manifest):,} rows)")
        print(f"  integrity  → {args.out_dir/'integrity.json'}")
        print(f"  removed    → {args.out_dir/'removed.jsonl'}  ({len(to_remove)} rows)")

    # ── Print summary ──────────────────────────────────────────────────
    print(f"\n═══ Summary ═══")
    print(f"  Total images kept: {len(manifest):,}")
    pos_count = sum(1 for m in manifest if m["class"] == "pos")
    neg_count = sum(1 for m in manifest if m["class"] == "neg")
    print(f"  Positives:  {pos_count:,}")
    print(f"  Negatives:  {neg_count:,}")
    print(f"  Ratio pos:neg = {pos_count/max(neg_count,1):.2f}:1")
    print(f"\n  Top subcategories:")
    for key, n in sorted(subcat_counts.items(), key=lambda kv: -kv[1])[:15]:
        print(f"    {key:42s} {n:>6,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
