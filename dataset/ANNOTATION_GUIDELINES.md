# Fence Segmentation — Annotation Guidelines

**Purpose**: Produce consistent pixel-level segmentation masks of *wood fences* across ~33k images. These masks train a deployed cedar-staining visualizer. Consistency matters more than perfection — two annotators must produce near-identical masks on the same image.

**Mask type**: Binary per-pixel. `1` = wood-fence pixel, `0` = everything else.

**Tooling**: Gemini Vision auto-labels each image, human annotator reviews + corrects. This document tells the reviewer what "correct" means.

---

## 1. What counts as a *wood fence*?

### Include (mask = 1)

- **Wood boards forming a boundary barrier**: picket, stockade, shadowbox, board-on-board, horizontal slat, split rail, dog-ear, scalloped, lattice-top
- **Wood types**: cedar, redwood, pine, pressure-treated, bamboo-privacy-fence, composite-wood-blend
- **Painted wood fences** (white, red, black, grey, etc.) — still mask them; color doesn't disqualify
- **Stained or weathered wood fences** — mask them regardless of condition
- **Fence posts** — the vertical wood supports count as fence
- **Wood gates integrated into a fence line** — part of the fence
- **Partial / broken fences** — mask whatever wood is visible
- **Heavily occluded fences** — mask the portions you can actually see through foliage/snow/people

### Exclude (mask = 0)

- **Vinyl, chain-link, aluminum, wrought iron, wire mesh, barbed wire, PVC, composite-plastic** — not wood
- **Retaining walls, brick/stone walls, concrete walls** — not fences
- **Wooden decks, pergolas, arbors, trellises** — even if attached to a fence, only mask the fence itself
- **Wood siding on houses/sheds** — not a fence
- **Bamboo groves / hedges / natural "walls" of plants** — not fences
- **Wooden bridges, boardwalks, docks** — not fences
- **Slatted wood on outdoor furniture** (benches, Adirondack chairs) — not fences
- **Shutters, blinds, louvered panels on buildings** — not fences

### Ambiguous cases → DEFAULT RULE

> **If it's wood AND acts as a boundary/barrier AND is freestanding (not part of a building) → mask it. Otherwise → do not mask.**

Examples:
- Wood gate across a driveway, freestanding → **mask** (fence-like function)
- Wood fence that transitions into a wood retaining wall → **mask only the freestanding-barrier portion**, not the retaining-wall part
- Wood fence with a pergola arching over it → **mask the fence only**, not the pergola
- Wood post with no panels attached → **mask** (it's a fence post even if isolated)

---

## 2. Boundary precision

### Top edge
The TOP of a fence is often the clearest boundary. Mask up to the top edge of the highest wood element. If there's a decorative cap rail, include it.

### Bottom edge
Bottom edge is often grass/ground — the **dividing line**:
- Clear visible fence bottom → mask to that edge
- Fence disappears into grass → mask where wood stops being visible, don't guess underneath
- Fence bottom obscured by plants → **follow the plant line** — wherever you can see wood, that's mask; the rest isn't

### Sides
Mask only the fence itself, not adjacent structures. If a fence butts against a house, mask only up to where the fence ends.

### Through-fence gaps
- Picket fences have gaps between pickets — **mask only the pickets and rails**, not the gaps. The gap pixels should be `0`.
- Chain link-style OR pickets so thin that through-mask is impractical — treat fence as a single region but be consistent.
- Lattice → only mask the lattice wood strips, not the diamond-shaped gaps.

### Occlusion (plants / snow / animals in front)
- Foreground elements (leaves, snow, people, dogs) = **not fence** = mask `0`
- Only mask wood fence **visible between/behind** the occluders
- Small gaps in foliage where fence shows through → mask those small wood regions
- Do **not** mask "through" foliage (don't guess what's behind a big bush)

---

## 3. Common edge cases (reviewed in golden set)

### Cedar + pergola combo
Both are cedar. The pergola has wood posts and cross-beams; the fence has vertical boards. **Mask only the fence boards and fence posts**, NOT the pergola structure even though they're the same material.

### Long-distance shots
Fence shows as a thin horizontal line far from the camera. Still mask it — even if it's only a few pixels wide, a skinny mask line is correct.

### Construction / half-built
Mask the wood parts already installed. Don't mask piles of loose boards next to the fence.

### Stain drips / mid-application
If stain is being actively applied, mask the fence (wet or dry, unpainted or painted). Don't worry about the stain — that's a per-pixel color the model learns separately.

### Fence in mirror / reflection
If a fence is reflected in a pool / window / puddle — **do not mask the reflection**, only mask real fence in the physical scene.

### Multiple fences in one image
Mask all visible fences with the same label (class 1). Don't differentiate "fence A vs fence B".

### Fence partially off-frame
Mask what's inside the frame. If half a fence post is cut off at image edge, mask the half that's visible.

---

## 4. Review workflow (human on Gemini auto-labels)

For each image, the tool shows:
- The image
- Gemini's auto-generated mask overlaid in semi-transparent color
- Edit tools (brush, eraser, polygon)

**Reviewer checklist per image**:

1. **Is a fence present at all?**
   - No → Delete mask entirely, mark image as negative
   - Yes → continue

2. **Is the mask on the right thing?**
   - Mask is on a pergola / railing / fence-like-not-fence → erase completely, don't re-mask (image is neg-class material)
   - Mask is on a non-wood fence (vinyl, chain link) → erase, image is neg-class for our model

3. **Boundary accuracy**
   - Mask extends beyond fence edge → erase the overshoot
   - Mask under-covers fence → extend to correct boundary
   - Mask includes occluders (leaves in front) → erase the occluder pixels

4. **Occlusion handling**
   - Mask paints "through" a bush where fence is hidden → erase
   - Mask is only on clearly-visible wood → correct

5. **Time budget per image**: aim for **15–30 seconds**. If an image needs > 2 minutes, flag it for senior review instead of spending excessive time.

---

## 5. Consistency checks (weekly)

- **Inter-annotator agreement** (IAA): re-mask 30 random images per week with 2 annotators, measure IoU between them. Target: **≥ 0.90 IoU** between reviewers.
- **Golden set regression**: run annotators against the 100-image golden set monthly, measure IoU vs. reference masks. Target: **≥ 0.85 IoU** mean.

If either metric drops, re-read this document and re-calibrate in a team sync.

---

## 6. Things to NEVER do

- ❌ Don't "fix" Gemini's mistakes by over-masking — a too-small correct mask is better than a too-big wrong one
- ❌ Don't mask items the model shouldn't learn (pergolas, decks, fence-like siding)
- ❌ Don't guess pixels you can't see (behind occlusion, beyond frame)
- ❌ Don't skip "hard" images — flag them for senior review instead of low-effort masking
- ❌ Don't change these guidelines unilaterally — propose updates via project lead

---

## 7. Version history

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-04-17 | Initial release — covers 33k image scrape, pre-annotation phase |

---

_Questions or edge cases not covered → ping the project lead. Update this document as new patterns emerge from review._
