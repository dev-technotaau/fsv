# Multi-Class Annotation Schema — Quick Reference

**Schema version**: `1.0.0`  
**Schema file**: [`configs/annotation_schema.yaml`](../configs/annotation_schema.yaml)  
**20 classes + background (ID 0)**  
**Priority = higher wins at pixel overlap.**

This is the annotator's / reviewer's quick-reference card. For pipeline architecture see [`ANNOTATION_SYSTEM.md`](ANNOTATION_SYSTEM.md).

---

## Class map

| ID | Name | Priority | Tier | Paint stain here? | Example |
|----|------|----------|------|-------------------|---------|
| 0 | `background` | 0 | reserved | no | any unlabeled pixel |
| 1 | `fence_wood` | 50 | fence | **YES** | main wood / cedar fence body |
| 2 | `fence_wood_post` | 55 | fence | **YES** | vertical wood posts in fence line |
| 3 | `fence_wood_gate` | 52 | fence | **YES** | wood gate integrated in fence |
| 4 | `fence_vinyl` | 48 | fence | no | white PVC privacy fence |
| 5 | `fence_metal` | 48 | fence | no | chain link, wrought iron, aluminum |
| 6 | `fence_masonry` | 48 | fence | no | brick / stone / concrete fence |
| 7 | `fence_natural` | 45 | fence | no | hedge, bamboo as boundary |
| 8 | `pergola_trellis` | 40 | distractor | no | standalone pergola / arbor |
| 9 | `deck_railing` | 42 | distractor | no | deck / porch / balcony rails |
| 10 | `wood_siding` | 38 | distractor | no | wood siding on house / shed |
| 11 | `wall_masonry` | 35 | distractor | no | retaining wall, brick wall |
| 12 | `vegetation_occluder` | **80** | occluder | no | leaves / branches / vines in FRONT |
| 13 | `vegetation_background` | 20 | context | no | distant trees, background foliage |
| 14 | `human_animal` | **90** | occluder | no | person / pet / wildlife |
| 15 | `foreground_object` | 75 | occluder | no | car / bike / pots / furniture |
| 16 | `pole_wire` | 70 | occluder | no | utility pole / lamp post / wires |
| 17 | `ground` | 10 | context | no | grass / dirt / pavement |
| 18 | `sky` | 5 | context | no | sky / clouds |
| 19 | `building` | 30 | context | no | house / garage / shed |
| 20 | `water_feature` | 15 | context | no | pool / pond / fountain / hot tub |
| 21 | `construction_maintenance` | **76** | occluder | no | **stacked fence boards** / lumber / tools / debris / fence waste / paint cans / lawn mower / ladders |
| 22 | `wood_gate_standalone` | 40 | distractor | no | freestanding wooden gate with no attached fence (archway, garden entry) |
| 23 | `fence_glass` | 48 | fence | no | glass pool fence / transparent panels / plexiglass |
| 24 | `logo_watermark` | **82** | occluder | no | stock-photo watermarks / company logos / URLs / signatures / decals on fence |

---

## Color legend (for viz PNGs)

| Class | Color |
|-------|-------|
| `fence_wood` | `#8B5A2B` cedar brown |
| `fence_wood_post` | `#654321` dark brown |
| `fence_wood_gate` | `#A06432` light brown |
| `fence_vinyl` | `#F0F0F0` white |
| `fence_metal` | `#505050` dark grey |
| `fence_masonry` | `#B48264` terracotta |
| `fence_natural` | `#1E641E` hedge green |
| `pergola_trellis` | `#C89664` tan |
| `deck_railing` | `#96643C` medium brown |
| `wood_siding` | `#AF825A` warm tan |
| `wall_masonry` | `#B46450` red-brown |
| `vegetation_occluder` | `#28B428` bright green |
| `vegetation_background` | `#147814` dark green |
| `human_animal` | `#FF6464` pink-red |
| `foreground_object` | `#B4B432` yellow |
| `pole_wire` | `#787878` grey |
| `ground` | `#785A3C` earth brown |
| `sky` | `#87CEEB` sky blue |
| `building` | `#C8C8A0` beige |
| `water_feature` | `#4682B4` steel blue |

---

## Decision tree for borderline classes

**Is it a wood fence?**
- Solid wood material, acts as boundary, freestanding → `fence_wood` (or `fence_wood_post` for posts, `fence_wood_gate` for gates)
- Wood but attached to building → `wood_siding`
- Wood but horizontal & people sit on it → `deck_railing` (deck) or `wall_masonry` (if masonry underneath)
- Wood but decorative with climbing plants → `pergola_trellis`

**Is it a non-wood fence?**
- PVC / vinyl → `fence_vinyl`
- Any metal → `fence_metal` (doesn't matter which subtype — aluminum, wrought iron, chain link all = 5)
- Masonry as boundary → `fence_masonry`
- Living boundary (hedge, bamboo) → `fence_natural`

**Is it in front of a fence?**
- Any person / animal → `human_animal` (ALWAYS wins, even tiny sliver)
- Plants / branches / vines partially blocking fence → `vegetation_occluder`
- Wheelbarrow, pot, furniture, bike → `foreground_object`
- Utility pole or wire → `pole_wire`

**Is it scene context (not occluding fence)?**
- Tree far from fence → `vegetation_background`
- Grass / dirt / paving → `ground`
- Sky visible → `sky`
- House / shed in scene → `building`

**Gaps between fence slats**:
- The GAP itself is background / sky / whatever's visible through
- Model will automatically classify gap pixels as `sky` (priority 5) or `vegetation_background` (priority 20) or `ground` depending on what's behind
- The fence_wood mask stays pixel-accurate on just the wood, preserved by SAM 2's structure-aware segmentation

---

## Output format

`masks/<uuid>.png` files are **8-bit single-channel PNGs** where each pixel value is a class ID (0-19).

```python
import numpy as np
from PIL import Image

mask = np.array(Image.open("masks/a1b2c3.png"))
# mask is uint8, shape (H, W)
# Count pixels per class:
np.bincount(mask.ravel(), minlength=20)

# Binary fence-wood mask (for stainer):
staining_mask = np.isin(mask, [1, 2, 3])   # fence_wood OR post OR gate
```

---

## For annotators reviewing flagged images

Open `dataset/annotations_v1/viz/<uuid>.png` alongside the original image. The colors in the viz tell you what the pipeline decided. If the colors look wrong:

- **Cedar fence painted as `fence_metal`** — fix by changing class in the mask
- **Tree branches not classified as `vegetation_occluder`** — add the branch pixels to class 12
- **Human appears as background** — mask them as class 14 (safety-critical!)
- **Gaps between pickets incorrectly filled as `fence_wood`** — erase those pixels; they should be sky/background

Follow [`ANNOTATION_GUIDELINES.md`](ANNOTATION_GUIDELINES.md) for full review protocol.
