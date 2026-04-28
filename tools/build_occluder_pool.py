"""tools/build_occluder_pool.py — Build the occluder cutout pool used by
CopyPasteOccluder during training.

THREE modes (combine freely):

  --procedural        Synthesize foliage / leaf clusters / branches
                      procedurally. ~200 by default. Zero downloads, zero
                      dependencies beyond PIL + numpy. Best for the common
                      "leaves in front of fence" case.

  --from-coco         Auto-download COCO 2017 val annotations (~240 MB,
                      one-time cached) and extract cutouts of fence-relevant
                      categories (person, bicycle, car, dog, cat, etc.) using
                      their built-in polygon segmentation masks. ~300 cutouts
                      by default. Internet required for first run; cached
                      after. NO rembg required — uses the dataset's masks.

  --from-images DIR   Run rembg on every image in DIR and save its foreground
                      as a transparent PNG. For when you have your own stock
                      photos and want them in the pool. Requires
                      `pip install rembg onnxruntime`.

Output: PNGs (RGBA, alpha-channel set) saved under <out_dir>, organized in
subdirectories by source / category. Default out_dir = dataset/occluders/.

`CopyPasteOccluder` recursively globs `*.png` from this directory at training
start, caches them in memory, and pastes 1-2 random occluders per training
image with random scale / rotation / position / color jitter.

Quick start (no internet, no extra deps):
    python -m tools.build_occluder_pool --procedural

Fully automatic (everything available, no manual cutout work):
    python -m tools.build_occluder_pool --procedural --from-coco

Add real photos you've sourced too:
    pip install rembg onnxruntime
    python -m tools.build_occluder_pool --procedural --from-coco --from-images ./stock

Inspect the current pool:
    python -m tools.build_occluder_pool --inspect
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ══════════════════════════════════════════════════════════════════════
# Procedural foliage generator
# ══════════════════════════════════════════════════════════════════════

# Color palettes — each tuple is (R, G, B) ranges; we sample a base, then
# perturb per-blob to avoid uniform color
LEAF_PALETTES = [
    # Healthy green
    {"base": (45, 110, 35), "spread": 35, "name": "leaf_green"},
    # Yellowing autumn
    {"base": (165, 145, 55), "spread": 40, "name": "leaf_yellow"},
    # Olive / muted
    {"base": (95, 105, 60), "spread": 25, "name": "leaf_olive"},
    # Dark forest
    {"base": (30, 75, 25), "spread": 20, "name": "leaf_dark"},
    # Reddish autumn
    {"base": (155, 80, 45), "spread": 35, "name": "leaf_red"},
    # Bare branch (brown)
    {"base": (85, 60, 40), "spread": 20, "name": "branch_brown"},
]


def _generate_leaf_cluster(
    size: int = 256, n_blobs: Optional[int] = None,
    palette: Optional[dict] = None, rng: Optional[np.random.Generator] = None,
) -> Image.Image:
    """Generate one synthetic foliage cluster as an RGBA PIL Image.
    The cluster is a few overlapping anti-aliased ellipses in random shades
    of green/brown, with feathered alpha edges. Looks like real leaf clumps."""
    if rng is None:
        rng = np.random.default_rng()
    if palette is None:
        palette = LEAF_PALETTES[int(rng.integers(0, len(LEAF_PALETTES)))]
    if n_blobs is None:
        n_blobs = int(rng.integers(8, 30))
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Cluster center + spread
    cx0 = size * float(rng.uniform(0.35, 0.65))
    cy0 = size * float(rng.uniform(0.35, 0.65))
    spread = size * float(rng.uniform(0.20, 0.42))
    base_r, base_g, base_b = palette["base"]
    pal_spread = palette["spread"]
    for _ in range(n_blobs):
        # Each blob is a small ellipse near the cluster center
        bx = cx0 + float(rng.normal(0, spread))
        by = cy0 + float(rng.normal(0, spread))
        bw = size * float(rng.uniform(0.05, 0.18))
        bh = size * float(rng.uniform(0.05, 0.18))
        # Slight random color variation around palette
        r = int(np.clip(base_r + rng.uniform(-pal_spread, pal_spread), 10, 230))
        g = int(np.clip(base_g + rng.uniform(-pal_spread, pal_spread), 10, 230))
        b = int(np.clip(base_b + rng.uniform(-pal_spread, pal_spread), 10, 230))
        a = int(rng.uniform(170, 240))
        bbox = (bx - bw, by - bh, bx + bw, by + bh)
        draw.ellipse(bbox, fill=(r, g, b, a))
    # Soft alpha edges (Gaussian blur on alpha channel only)
    rgb = img.convert("RGB")
    alpha = img.split()[-1]
    blur_radius = float(rng.uniform(2.0, 5.5))
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    img = Image.merge("RGBA", (*rgb.split(), alpha))
    # Crop to alpha bbox so we don't store a lot of empty space
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    return img


def _generate_branch(
    size: int = 256, rng: Optional[np.random.Generator] = None,
) -> Image.Image:
    """Generate a thin branching structure (vines / twigs) as RGBA."""
    if rng is None:
        rng = np.random.default_rng()
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    base_r, base_g, base_b = 80, 60, 40   # brown
    n_branches = int(rng.integers(3, 8))
    cx, cy = size / 2, size / 2
    for _ in range(n_branches):
        angle = float(rng.uniform(0, 2 * math.pi))
        length = float(rng.uniform(size * 0.3, size * 0.5))
        x_end = cx + math.cos(angle) * length
        y_end = cy + math.sin(angle) * length
        thickness = int(rng.uniform(2, 6))
        r = int(np.clip(base_r + rng.uniform(-15, 15), 30, 130))
        g = int(np.clip(base_g + rng.uniform(-15, 15), 30, 100))
        b = int(np.clip(base_b + rng.uniform(-10, 10), 20, 90))
        draw.line([(cx, cy), (x_end, y_end)], fill=(r, g, b, 220), width=thickness)
        # Small leaf clusters at the tip
        if rng.random() < 0.6:
            for _ in range(int(rng.integers(2, 6))):
                lx = x_end + float(rng.normal(0, 8))
                ly = y_end + float(rng.normal(0, 8))
                lw = float(rng.uniform(8, 18))
                lh = float(rng.uniform(8, 18))
                lr = int(np.clip(60 + rng.uniform(-20, 30), 30, 200))
                lg = int(np.clip(110 + rng.uniform(-30, 40), 50, 220))
                lb = int(np.clip(40 + rng.uniform(-15, 25), 20, 150))
                draw.ellipse((lx - lw, ly - lh, lx + lw, ly + lh),
                              fill=(lr, lg, lb, 220))
    rgb = img.convert("RGB")
    alpha = img.split()[-1]
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=1.5))
    img = Image.merge("RGBA", (*rgb.split(), alpha))
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    return img


def cmd_procedural(out_dir: Path, count: int, size: int, seed: int) -> int:
    out_dir = out_dir / "procedural"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    print(f"Generating {count} procedural occluders to {out_dir} ...")
    n_branch = max(20, count // 6)        # ~17% branches, rest leaf clusters
    n_leaf = count - n_branch
    t0 = time.time()
    for i in range(n_leaf):
        sz = int(rng.integers(160, size + 1))
        img = _generate_leaf_cluster(size=sz, rng=rng)
        out_path = out_dir / f"leaf_{i:04d}.png"
        img.save(out_path, optimize=False, compress_level=1)
    for i in range(n_branch):
        sz = int(rng.integers(180, size + 1))
        img = _generate_branch(size=sz, rng=rng)
        out_path = out_dir / f"branch_{i:04d}.png"
        img.save(out_path, optimize=False, compress_level=1)
    elapsed = time.time() - t0
    print(f"  Saved {n_leaf} leaf clusters + {n_branch} branches "
          f"({elapsed:.1f}s).")
    return n_leaf + n_branch


# ══════════════════════════════════════════════════════════════════════
# Real-photo extraction via rembg
# ══════════════════════════════════════════════════════════════════════

def cmd_from_images(src_dir: Path, out_dir: Path, max_size: int = 768,
                     min_area_frac: float = 0.04) -> int:
    """Use rembg to extract foreground from each image in src_dir, save as RGBA."""
    try:
        from rembg import remove, new_session
    except ImportError:
        print("ERROR: rembg not installed. Run: pip install rembg onnxruntime",
              file=sys.stderr)
        return 0

    out_dir = out_dir / "real"
    out_dir.mkdir(parents=True, exist_ok=True)
    src_paths: list[Path] = []
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        src_paths.extend(src_dir.rglob(f"*{ext}"))
        src_paths.extend(src_dir.rglob(f"*{ext.upper()}"))
    src_paths = sorted(set(src_paths))
    print(f"Found {len(src_paths)} source images in {src_dir}")
    if not src_paths:
        return 0

    print("Initializing rembg (downloads model on first run, ~170MB)...")
    session = new_session("u2net")

    n_ok = 0
    n_skip = 0
    t0 = time.time()
    for i, p in enumerate(src_paths):
        try:
            with open(p, "rb") as f:
                data = f.read()
            output = remove(data, session=session)
            img = Image.open(io.BytesIO(output)).convert("RGBA")

            # Crop to alpha bbox
            bbox = img.getbbox()
            if bbox is None:
                n_skip += 1
                continue
            img = img.crop(bbox)
            # Reject tiny/empty extractions (e.g., rembg failed)
            alpha = np.array(img.split()[-1])
            area_frac = (alpha > 32).sum() / max(1, alpha.size)
            if area_frac < min_area_frac:
                n_skip += 1
                continue
            # Cap dimensions
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_w = int(img.size[0] * ratio)
                new_h = int(img.size[1] * ratio)
                img = img.resize((new_w, new_h), Image.LANCZOS)
            out_path = out_dir / f"{p.stem}_occ.png"
            img.save(out_path, optimize=False, compress_level=1)
            n_ok += 1
            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.time() - t0 + 1e-6)
                eta = (len(src_paths) - i - 1) / max(rate, 1e-6)
                print(f"  [{i + 1:>4}/{len(src_paths):>4}]  "
                      f"{rate:5.1f} img/s  eta={int(eta):>4}s  "
                      f"saved={n_ok}  skipped={n_skip}")
        except Exception as e:
            print(f"  [{i:>4}] {p.name} failed: {type(e).__name__}: {str(e)[:80]}")
            n_skip += 1

    elapsed = time.time() - t0
    print(f"  Done. Saved {n_ok} cutouts ({n_skip} skipped) in {elapsed:.1f}s")
    return n_ok


# ══════════════════════════════════════════════════════════════════════
# Procedural "wooden non-fence" hard-negative generator
# ══════════════════════════════════════════════════════════════════════
#
# Produces RGBA cutouts of wooden objects that are NOT fences — wooden
# planks, raised garden beds, deck patches. These get pasted into
# BACKGROUND-only regions of training images by HardNegativeWoodPaste,
# explicitly teaching the model that wood texture alone is not enough
# to call something a fence (it needs scene context + shape).
#
# Why this matters: real fence images often share scenes with garden
# beds, decking, planters, fences-of-other-style, and tree trunks —
# all wooden, all NOT the target fence. Without explicit hard negatives,
# focal-BCE alone is not strong enough signal.

WOOD_PALETTES = [
    # Light cedar
    {"base": (200, 165, 120), "spread": 18, "name": "wood_cedar_light"},
    # Stained cedar (red-brown)
    {"base": (140,  85,  55), "spread": 18, "name": "wood_cedar_stained"},
    # Weathered gray-brown
    {"base": (130, 115,  95), "spread": 14, "name": "wood_weathered"},
    # Pressure-treated (greenish-tan)
    {"base": (160, 145,  95), "spread": 16, "name": "wood_pt"},
    # Dark walnut / treated brown
    {"base": ( 95,  65,  40), "spread": 16, "name": "wood_dark"},
    # Pine (yellow-tan)
    {"base": (210, 180, 130), "spread": 16, "name": "wood_pine"},
]


def _wood_color(palette: dict, rng: np.random.Generator,
                  brightness: float = 1.0) -> tuple[int, int, int]:
    base_r, base_g, base_b = palette["base"]
    s = palette["spread"]
    r = int(np.clip((base_r + rng.uniform(-s, s)) * brightness, 0, 255))
    g = int(np.clip((base_g + rng.uniform(-s, s)) * brightness, 0, 255))
    b = int(np.clip((base_b + rng.uniform(-s, s)) * brightness, 0, 255))
    return (r, g, b)


def _draw_wood_grain(draw: "ImageDraw.ImageDraw", x0: int, y0: int,
                      x1: int, y1: int, palette: dict,
                      rng: np.random.Generator,
                      vertical: bool = True) -> None:
    """Draw wood-grain texture (faint streaks) inside a rectangle."""
    base_r, base_g, base_b = palette["base"]
    n_streaks = int(rng.integers(8, 22))
    for _ in range(n_streaks):
        # Slight color variation per streak
        dr = int(rng.uniform(-25, 5))
        dg = int(rng.uniform(-20, 5))
        db = int(rng.uniform(-15, 5))
        col = (max(0, base_r + dr), max(0, base_g + dg), max(0, base_b + db),
               int(rng.uniform(60, 140)))
        if vertical:
            sx = int(rng.uniform(x0, x1))
            sy = y0 + int(rng.uniform(0, 10))
            ey = y1 - int(rng.uniform(0, 10))
            draw.line([(sx, sy), (sx + int(rng.uniform(-4, 4)), ey)],
                       fill=col, width=int(rng.integers(1, 3)))
        else:
            sy = int(rng.uniform(y0, y1))
            sx = x0 + int(rng.uniform(0, 10))
            ex = x1 - int(rng.uniform(0, 10))
            draw.line([(sx, sy), (ex, sy + int(rng.uniform(-4, 4)))],
                       fill=col, width=int(rng.integers(1, 3)))


def _generate_wood_plank(size: int = 256,
                          rng: Optional[np.random.Generator] = None,
                          ) -> Image.Image:
    """A single vertical wooden plank or short stack of horizontal boards.
    Looks like a rough piece of lumber lying around."""
    if rng is None:
        rng = np.random.default_rng()
    palette = WOOD_PALETTES[int(rng.integers(0, len(WOOD_PALETTES)))]
    # Plank is taller than wide (vertical) — but rotated randomly later
    h = size
    w = int(size * float(rng.uniform(0.18, 0.45)))
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Fill the plank with base color (slight gradient for depth)
    base = _wood_color(palette, rng, brightness=1.0)
    draw.rectangle((0, 0, w, h), fill=(*base, 230))
    # Wood grain
    _draw_wood_grain(draw, 0, 0, w, h, palette, rng, vertical=True)
    # Edge shading for 3D feel
    edge_dark = _wood_color(palette, rng, brightness=0.65)
    draw.rectangle((0, 0, 2, h), fill=(*edge_dark, 200))
    draw.rectangle((w - 2, 0, w, h), fill=(*edge_dark, 200))
    # Random rotation
    angle = float(rng.uniform(-90, 90))
    img = img.rotate(angle, resample=Image.BILINEAR, expand=True)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    return img


def _generate_wood_box(size: int = 256,
                        rng: Optional[np.random.Generator] = None,
                        ) -> Image.Image:
    """Garden bed / wooden planter box. Rectangular, horizontal plank texture,
    with subtle 3D edges so it looks like a 3D box from a low angle."""
    if rng is None:
        rng = np.random.default_rng()
    palette = WOOD_PALETTES[int(rng.integers(0, len(WOOD_PALETTES)))]
    w = size
    h = int(size * float(rng.uniform(0.40, 0.75)))
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Horizontal planks, 2-4 of them
    n_planks = int(rng.integers(2, 5))
    plank_h = h // n_planks
    for i in range(n_planks):
        y0 = i * plank_h
        y1 = (i + 1) * plank_h - 1
        col = _wood_color(palette, rng, brightness=float(rng.uniform(0.85, 1.05)))
        draw.rectangle((0, y0, w, y1), fill=(*col, 235))
        _draw_wood_grain(draw, 0, y0, w, y1, palette, rng, vertical=False)
        # Plank seam (dark line between planks)
        seam = _wood_color(palette, rng, brightness=0.5)
        draw.rectangle((0, y1 - 1, w, y1 + 1), fill=(*seam, 220))
    # Side shadow — gives it 3D feel
    side_dark = _wood_color(palette, rng, brightness=0.7)
    side_w = int(rng.integers(3, 8))
    draw.rectangle((0, 0, side_w, h), fill=(*side_dark, 180))
    draw.rectangle((w - side_w, 0, w, h), fill=(*side_dark, 180))
    # Slight rotation (planters usually sit at slight angles in photos)
    angle = float(rng.uniform(-10, 10))
    img = img.rotate(angle, resample=Image.BILINEAR, expand=True)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    return img


def _generate_wood_deck(size: int = 256,
                         rng: Optional[np.random.Generator] = None,
                         ) -> Image.Image:
    """Decking surface: many horizontal planks with visible seams."""
    if rng is None:
        rng = np.random.default_rng()
    palette = WOOD_PALETTES[int(rng.integers(0, len(WOOD_PALETTES)))]
    w = size
    h = int(size * float(rng.uniform(0.50, 0.95)))
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # 4-8 horizontal planks with clear seams
    n_planks = int(rng.integers(4, 9))
    plank_h = h // n_planks
    for i in range(n_planks):
        y0 = i * plank_h
        y1 = (i + 1) * plank_h - 2     # leave a 2px seam (which becomes dark)
        col = _wood_color(palette, rng, brightness=float(rng.uniform(0.88, 1.05)))
        draw.rectangle((0, y0, w, y1), fill=(*col, 235))
        _draw_wood_grain(draw, 0, y0, w, y1, palette, rng, vertical=False)
        # Dark seam
        seam = _wood_color(palette, rng, brightness=0.45)
        draw.rectangle((0, y1, w, y1 + 2), fill=(*seam, 220))
    # Random perspective-ish skew (subtle)
    skew = float(rng.uniform(-3, 3))
    if abs(skew) > 0.5:
        img = img.rotate(skew, resample=Image.BILINEAR, expand=True)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    return img


def cmd_wooden_negatives(out_dir: Path, count: int, size: int,
                          seed: int) -> int:
    """Generate procedural wooden non-fence cutouts (plank / box / deck)."""
    out_dir = out_dir / "wood"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    print(f"Generating {count} wooden non-fence cutouts -> {out_dir}")
    # Roughly 40% planks, 35% boxes/garden beds, 25% deck patches
    n_plank = int(count * 0.40)
    n_box = int(count * 0.35)
    n_deck = count - n_plank - n_box
    t0 = time.time()
    for i in range(n_plank):
        sz = int(rng.integers(180, size + 1))
        img = _generate_wood_plank(size=sz, rng=rng)
        img.save(out_dir / f"plank_{i:04d}.png", optimize=False, compress_level=1)
    for i in range(n_box):
        sz = int(rng.integers(180, size + 1))
        img = _generate_wood_box(size=sz, rng=rng)
        img.save(out_dir / f"box_{i:04d}.png", optimize=False, compress_level=1)
    for i in range(n_deck):
        sz = int(rng.integers(180, size + 1))
        img = _generate_wood_deck(size=sz, rng=rng)
        img.save(out_dir / f"deck_{i:04d}.png", optimize=False, compress_level=1)
    elapsed = time.time() - t0
    print(f"  Saved {n_plank} planks + {n_box} boxes + {n_deck} deck patches "
          f"({elapsed:.1f}s)")
    return count


# ══════════════════════════════════════════════════════════════════════
# COCO 2017 auto-download cutout extractor
# ══════════════════════════════════════════════════════════════════════

# COCO categories that make sense as fence-line occluders. Mapping is
# {coco_category_id: human_name_for_subdir}. Curated by hand from the
# COCO 2017 80-class taxonomy.
USEFUL_COCO_CATEGORIES: dict[int, str] = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    8: "truck",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    62: "chair",
    64: "potted_plant",
}

COCO_VAL_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_VAL_IMAGE_URL_FMT = "http://images.cocodataset.org/val2017/{:012d}.jpg"
COCO_INSTANCES_VAL_MEMBER = "annotations/instances_val2017.json"


def _download_with_progress(url: str, dest: Path,
                             chunk: int = 1 << 20) -> None:
    """Stream-download `url` to `dest` with a simple progress line."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  Downloading {url}")
    print(f"           -> {dest}")
    t0 = time.time()
    try:
        with urllib.request.urlopen(url) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            last_print = 0.0
            with open(tmp, "wb") as f:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)
                    now = time.time()
                    if now - last_print >= 0.5 or (total and done >= total):
                        rate = done / max(now - t0, 1e-6) / (1 << 20)
                        if total:
                            pct = 100.0 * done / total
                            print(
                                f"    {done / (1 << 20):8.1f} / "
                                f"{total / (1 << 20):.1f} MiB  "
                                f"({pct:5.1f}%)  {rate:5.1f} MiB/s",
                                end="\r", flush=True,
                            )
                        else:
                            print(
                                f"    {done / (1 << 20):8.1f} MiB  "
                                f"{rate:5.1f} MiB/s",
                                end="\r", flush=True,
                            )
                        last_print = now
        print()
        tmp.replace(dest)
    except (urllib.error.URLError, OSError) as e:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise RuntimeError(f"download failed: {e}") from e


def _ensure_coco_annotations(cache_dir: Path) -> Path:
    """Download (if needed) the COCO val2017 instance annotations JSON.
    Returns path to the extracted instances_val2017.json."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    json_path = cache_dir / "instances_val2017.json"
    if json_path.exists() and json_path.stat().st_size > 1_000_000:
        return json_path
    zip_path = cache_dir / "annotations_trainval2017.zip"
    if not zip_path.exists() or zip_path.stat().st_size < 100_000_000:
        _download_with_progress(COCO_VAL_ANNOTATIONS_URL, zip_path)
    print(f"  Extracting {COCO_INSTANCES_VAL_MEMBER} ...")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(COCO_INSTANCES_VAL_MEMBER) as src, \
                open(json_path.with_suffix(".json.tmp"), "wb") as dst:
            while True:
                buf = src.read(1 << 20)
                if not buf:
                    break
                dst.write(buf)
    json_path.with_suffix(".json.tmp").replace(json_path)
    print(f"  Wrote {json_path} ({json_path.stat().st_size / (1 << 20):.1f} MiB)")
    return json_path


def _polygon_to_mask(polygons: list[list[float]],
                      height: int, width: int) -> np.ndarray:
    """Render a list of COCO polygon segs to a binary uint8 mask (H, W)."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        if len(poly) < 6:
            continue  # need >= 3 vertices
        pts = [(float(poly[i]), float(poly[i + 1]))
               for i in range(0, len(poly) - 1, 2)]
        draw.polygon(pts, fill=255)
    return np.asarray(mask, dtype=np.uint8)


def cmd_from_coco(out_dir: Path, per_category: int = 30,
                   max_size: int = 512, min_area_frac: float = 0.04,
                   cache_dir: Optional[Path] = None,
                   seed: int = 42) -> int:
    """Auto-download COCO val2017 + extract cutouts using built-in poly masks."""
    cache_dir = cache_dir or (out_dir.parent / "_coco_cache")
    rng = np.random.default_rng(seed)
    out_root = out_dir / "coco"
    out_root.mkdir(parents=True, exist_ok=True)
    image_cache = cache_dir / "val2017"
    image_cache.mkdir(parents=True, exist_ok=True)

    print(f"COCO mode: target {per_category}/category × "
          f"{len(USEFUL_COCO_CATEGORIES)} categories "
          f"-> ~{per_category * len(USEFUL_COCO_CATEGORIES)} cutouts")
    print(f"  cache_dir: {cache_dir}")

    json_path = _ensure_coco_annotations(cache_dir)
    print(f"  Loading annotations JSON ({json_path.stat().st_size / (1 << 20):.0f} MiB)...")
    with open(json_path, "rb") as f:
        coco = json.load(f)

    # Build image_id -> (file_name, height, width)
    images_by_id = {img["id"]: img for img in coco["images"]}

    # Collect per-category candidate annotations (polygon segs only)
    by_cat: dict[int, list[dict]] = {cid: [] for cid in USEFUL_COCO_CATEGORIES}
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        if cid not in by_cat:
            continue
        seg = ann.get("segmentation")
        if not isinstance(seg, list) or not seg:
            continue  # skip RLE / empty
        if ann.get("iscrowd"):
            continue
        if ann.get("area", 0) < 1500:  # too small to be useful
            continue
        by_cat[cid].append(ann)

    print("  candidates per category:")
    for cid, name in USEFUL_COCO_CATEGORIES.items():
        print(f"    {name:<14} {len(by_cat[cid]):>5}")

    n_ok = 0
    n_skip = 0
    t0 = time.time()
    for cid, name in USEFUL_COCO_CATEGORIES.items():
        anns = by_cat[cid]
        if not anns:
            continue
        cat_dir = out_root / name
        cat_dir.mkdir(parents=True, exist_ok=True)
        # Random sample without replacement
        idxs = rng.permutation(len(anns))[:per_category]
        for k, ai in enumerate(idxs):
            ann = anns[int(ai)]
            img_meta = images_by_id.get(ann["image_id"])
            if img_meta is None:
                n_skip += 1
                continue
            file_name = img_meta["file_name"]
            H, W = img_meta["height"], img_meta["width"]
            img_local = image_cache / file_name
            try:
                if not img_local.exists() or img_local.stat().st_size < 1000:
                    url = COCO_VAL_IMAGE_URL_FMT.format(ann["image_id"])
                    _download_image_quiet(url, img_local)
                pil = Image.open(img_local).convert("RGB")
                if pil.size != (W, H):
                    W, H = pil.size  # trust actual file
                mask = _polygon_to_mask(ann["segmentation"], H, W)
                # Reject tiny masks (in case poly was off-image)
                area_frac = (mask > 0).sum() / float(mask.size)
                if area_frac < min_area_frac * 0.25:
                    n_skip += 1
                    continue
                rgb = np.asarray(pil, dtype=np.uint8)
                rgba = np.dstack([rgb, mask])
                cutout = Image.fromarray(rgba, mode="RGBA")
                bbox = cutout.getbbox()
                if bbox is None:
                    n_skip += 1
                    continue
                # Pad bbox slightly so soft alpha edges aren't clipped flat
                x0, y0, x1, y1 = bbox
                pad = 4
                x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
                x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
                cutout = cutout.crop((x0, y0, x1, y1))
                # Optional small alpha feathering (dataset masks are hard-edged)
                rgb_part = cutout.convert("RGB")
                alpha = cutout.split()[-1].filter(
                    ImageFilter.GaussianBlur(radius=0.8)
                )
                cutout = Image.merge("RGBA", (*rgb_part.split(), alpha))
                # Cap dimensions
                if max(cutout.size) > max_size:
                    ratio = max_size / max(cutout.size)
                    nw = max(1, int(cutout.size[0] * ratio))
                    nh = max(1, int(cutout.size[1] * ratio))
                    cutout = cutout.resize((nw, nh), Image.LANCZOS)
                # Final area check after crop+resize
                a = np.asarray(cutout.split()[-1])
                area_frac2 = (a > 32).sum() / float(a.size)
                if area_frac2 < min_area_frac:
                    n_skip += 1
                    continue
                out_path = cat_dir / f"{ann['image_id']:012d}_{ann['id']}.png"
                cutout.save(out_path, optimize=False, compress_level=1)
                n_ok += 1
            except Exception as e:
                n_skip += 1
                print(f"    [{name}] ann {ann.get('id')} failed: "
                      f"{type(e).__name__}: {str(e)[:80]}")
                continue
            if (n_ok + n_skip) % 25 == 0:
                rate = (n_ok + n_skip) / max(time.time() - t0, 1e-6)
                print(f"  [{name}] {k + 1}/{len(idxs)}  "
                      f"total saved={n_ok}  skipped={n_skip}  "
                      f"({rate:5.1f}/s)")

    elapsed = time.time() - t0
    print(f"  Done. Saved {n_ok} COCO cutouts ({n_skip} skipped) "
          f"in {elapsed:.1f}s")
    return n_ok


def _download_image_quiet(url: str, dest: Path) -> None:
    """Download a single small image with no progress chatter."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp, \
                open(tmp, "wb") as f:
            while True:
                buf = resp.read(1 << 16)
                if not buf:
                    break
                f.write(buf)
        tmp.replace(dest)
    except (urllib.error.URLError, OSError) as e:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise RuntimeError(f"image download failed: {e}") from e


# ══════════════════════════════════════════════════════════════════════
# Pool inspector
# ══════════════════════════════════════════════════════════════════════

def cmd_inspect(out_dir: Path) -> None:
    if not out_dir.exists():
        print(f"  {out_dir} does not exist.")
        return
    paths = sorted(out_dir.rglob("*.png"))
    print(f"  total occluder PNGs: {len(paths)}")
    if not paths:
        return
    by_subdir: dict[str, int] = {}
    for p in paths:
        rel = p.relative_to(out_dir).parent
        key = str(rel) if str(rel) else "(root)"
        by_subdir[key] = by_subdir.get(key, 0) + 1
    print("  per subdirectory:")
    for k in sorted(by_subdir.keys()):
        print(f"    {k:<24}  {by_subdir[k]:,}")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out-dir", type=Path, default=Path("dataset/occluders"),
                    help="Output directory for the occluder pool.")
    ap.add_argument("--procedural", action="store_true",
                    help="Generate procedural foliage occluders (no extra deps).")
    ap.add_argument("--procedural-count", type=int, default=200,
                    help="Number of procedural occluders to generate (default 200).")
    ap.add_argument("--procedural-size", type=int, default=320,
                    help="Max canvas size in px for each procedural occluder.")
    ap.add_argument("--wooden-negatives", action="store_true",
                    help="Generate procedural WOODEN NON-FENCE cutouts (planks, "
                         "garden beds, deck patches). These are pasted into "
                         "BACKGROUND regions of training images by "
                         "HardNegativeWoodPaste — explicit signal that wood "
                         "alone is not enough to call something a fence. "
                         "Critical for scenes with wooden garden boxes.")
    ap.add_argument("--wooden-negatives-count", type=int, default=200,
                    help="How many wooden non-fence cutouts to generate.")
    ap.add_argument("--wooden-negatives-out", type=Path,
                    default=Path("dataset/hard_negatives"),
                    help="Output dir for wooden non-fence pool.")
    ap.add_argument("--from-coco", action="store_true",
                    help="Auto-download COCO 2017 val + extract cutouts via "
                         "polygon segs (no rembg, no manual work).")
    ap.add_argument("--coco-per-category", type=int, default=30,
                    help="Cutouts to extract per COCO category (default 30; "
                         "x ~17 categories ≈ 510 cutouts).")
    ap.add_argument("--coco-cache-dir", type=Path, default=None,
                    help="Where to cache COCO downloads (default "
                         "<out_dir>/../_coco_cache).")
    ap.add_argument("--from-images", type=Path, default=None,
                    help="Source dir of real photos to run rembg on.")
    ap.add_argument("--max-size", type=int, default=768,
                    help="Max width/height for saved cutouts (default 768).")
    ap.add_argument("--min-area-frac", type=float, default=0.04,
                    help="Reject outputs whose foreground area is less than "
                         "this fraction (default 0.04 = 4%%).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inspect", action="store_true",
                    help="Just print pool stats and exit.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.inspect:
        print(f"Inspecting {args.out_dir}:")
        cmd_inspect(args.out_dir)
        return 0

    if (not args.procedural and not args.from_coco
            and args.from_images is None and not args.wooden_negatives):
        print("Nothing to do. Pass --procedural and/or --from-coco and/or "
              "--from-images <DIR> and/or --wooden-negatives.")
        ap.print_help()
        return 1

    total_added = 0
    if args.procedural:
        total_added += cmd_procedural(args.out_dir, args.procedural_count,
                                       args.procedural_size, args.seed)
    if args.wooden_negatives:
        total_added += cmd_wooden_negatives(
            args.wooden_negatives_out, args.wooden_negatives_count,
            args.procedural_size, args.seed,
        )
    if args.from_coco:
        total_added += cmd_from_coco(
            args.out_dir,
            per_category=args.coco_per_category,
            max_size=args.max_size,
            min_area_frac=args.min_area_frac,
            cache_dir=args.coco_cache_dir,
            seed=args.seed,
        )
    if args.from_images is not None:
        if not args.from_images.exists():
            print(f"ERROR: --from-images dir does not exist: {args.from_images}",
                  file=sys.stderr)
            return 2
        total_added += cmd_from_images(args.from_images, args.out_dir,
                                        max_size=args.max_size,
                                        min_area_frac=args.min_area_frac)

    print()
    print("=" * 60)
    print(f"Pool ready at {args.out_dir}/")
    cmd_inspect(args.out_dir)
    print()
    print("Test in training: CopyPasteOccluder will auto-detect the pool")
    print("on next training run (verifies via warning if pool is empty).")
    return 0 if total_added > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
