"""Multi-class mask fusion — priority-based + per-pixel conflict resolution.

Given a list of per-instance masks (each with class_id, priority, confidence),
produce a SINGLE multi-class segmentation map (H, W) uint8 where each pixel
takes the class_id of the winning mask under the configured resolution rule.

Priority logic:
  1. Higher-priority class wins over lower-priority class (e.g., human_animal
     at priority 90 beats fence_wood at priority 50 → human's pixel stays human).
  2. Within same priority, higher-confidence mask wins.
  3. Unlabeled pixels default to class_id 0 (background).

This is critical for gap preservation: SAM 2 returns a fence mask with natural
gaps between pickets (because it was trained to segment "the picket fence
structure", not "all pixels in the bounding box"). We preserve those gaps
by using SAM masks directly — no flood-fill or convex-hull modifications.
"""
from __future__ import annotations

import numpy as np

from annotation.sam2 import InstanceMask
from annotation.schema import Schema


# Fence preservation rules against higher-priority false positives.
# DINO's "bush/shrub" / "outdoor furniture" / "stacked fence boards" prompts
# often match cedar wood-grain texture at scores 0.40-0.65, so raw priority
# fusion paints cedar fences the wrong color. For the stain-visualizer use
# case, preserving a well-detected fence beats blindly trusting high-score
# occluder calls.
#
# Schema-driven: we derive fence_ids and occluder_ids from the schema's `tier`
# field instead of hardcoding. Works identically for both the binary schema
# (fence tier = {fence_wood}) and the legacy 24-class schema.
# Explicitly NOT blocked: human_animal (tier=occluder BUT priority 90) —
# real people must occlude fence, so we exclude it from the blocking set.
#
# Two-tier protection:
#   (1) STRONG fence (conf >= STRONG_FENCE_PROTECT_CONF): IMMUNE regardless
#       of occluder score.
#   (2) MODERATE fence (conf >= FENCE_PROTECT_CONF): protected only from
#       WEAK occluders (conf < OCCLUDER_OVERRIDE_CONF).
FENCE_PROTECT_CONF = 0.20          # moderate fence (vs weak occluder)
STRONG_FENCE_PROTECT_CONF = 0.25   # strong fence (vs any occluder)
OCCLUDER_OVERRIDE_CONF = 0.30      # below this, occluder is "weak"

# Occluder classes we want to BLOCK from overwriting well-detected fence.
# Derived from schema: every "occluder" tier class EXCEPT human_animal.
_EXCLUDE_FROM_BLOCKING_NAMES = {"human_animal"}

# When a "distractor" tier class (absorber) tries to overwrite a "fence"
# tier class via priority, REQUIRE this confidence margin to actually win.
# DINO matches cedar wood texture to BOTH "cedar fence" AND "wood siding"
# / "wooden chair" prompts. Without this margin, the absorber would always
# win via priority and the actual cedar fence would never get stained.
# Setting 0.10 means: absorber must score 10 percentage points higher than
# fence to override. Stone wall vs cedar fence: not_target ~0.45, fence_wood
# ~0.20 → absorber wins (margin met). Cedar fence: fence_wood ~0.40,
# not_target ~0.30 → fence_wood wins (margin not met).
ABSORBER_OVERRIDE_MARGIN = 0.07


def fuse_masks(
    instance_masks: list[InstanceMask],
    image_hw: tuple[int, int],
    schema: Schema,
    strategy: str = "winner_takes_pixel",
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse per-instance masks into a (H, W) uint8 class-map + (H, W) float32 confidence map.

    Returns:
        class_map: uint8 array (H, W), values in [0, num_classes)
        conf_map:  float32 array (H, W), winning mask's confidence per pixel
    """
    # Derive fence / occluder / distractor class IDs from the schema.
    fence_class_ids = frozenset(
        c.id for c in schema.classes if c.tier == "fence"
    )
    occluder_class_ids = frozenset(
        c.id for c in schema.classes
        if c.tier == "occluder" and c.name not in _EXCLUDE_FROM_BLOCKING_NAMES
    )
    # Distractor tier = "absorber" classes (e.g., not_target). They're given
    # higher priority than fence so they can route DINO's wood-like-but-not-
    # fence false positives away from staining. But this priority must NOT
    # be absolute — DINO matches cedar wood ALSO to "wood siding" / "wooden
    # chair" prompts. We require ABSORBER_OVERRIDE_MARGIN of additional conf
    # before letting an absorber overwrite a fence pixel.
    distractor_class_ids = frozenset(
        c.id for c in schema.classes if c.tier == "distractor"
    )

    H, W = image_hw
    class_map = np.zeros((H, W), dtype=np.uint8)
    conf_map = np.zeros((H, W), dtype=np.float32)
    prio_map = np.zeros((H, W), dtype=np.int32)   # highest priority claimed so far per pixel

    # Sort masks by priority ASCENDING — we paint low-priority first,
    # then higher-priority masks overwrite. Within same priority, more
    # confident masks painted later win.
    def sort_key(im: InstanceMask) -> tuple:
        cls = schema.by_id(im.class_id)
        pri = cls.priority if cls else 0
        # sort: priority ASC, then confidence ASC → last painted = highest priority + confidence
        return (pri, im.sam_score * im.detection_score)

    sorted_masks = sorted(instance_masks, key=sort_key)

    for im in sorted_masks:
        cls = schema.by_id(im.class_id)
        if cls is None:
            continue
        pri = cls.priority
        conf = float(im.sam_score * im.detection_score)

        # Apply this mask where:
        #   (a) the existing pixel priority is strictly lower, OR
        #   (b) equal priority but this mask has higher confidence
        mask_region = im.mask
        if mask_region.shape != (H, W):
            # Shouldn't happen — SAM 2 wrapper already resized — but guard anyway.
            continue

        higher_prio = (prio_map < pri) & mask_region
        equal_prio_better_conf = (prio_map == pri) & (conf_map < conf) & mask_region
        write_mask = higher_prio | equal_prio_better_conf

        # Fence preservation against occluder over-detection — two tiers.
        if im.class_id in occluder_class_ids:
            fence_present = np.isin(class_map, list(fence_class_ids))
            # Tier 1: strong fence is immune to any occluder, regardless of score
            strong_fence = fence_present & (conf_map >= STRONG_FENCE_PROTECT_CONF)
            is_protected = strong_fence
            # Tier 2: moderate fence is protected only from weak occluders
            if conf < OCCLUDER_OVERRIDE_CONF:
                moderate_fence = fence_present & (conf_map >= FENCE_PROTECT_CONF)
                is_protected = is_protected | moderate_fence
            write_mask = write_mask & ~is_protected

        # Absorber (distractor-tier) score-margin override:
        # When a distractor wants to overwrite a fence pixel, only allow the
        # overwrite if the distractor's confidence beats the fence's by at
        # least ABSORBER_OVERRIDE_MARGIN. Prevents cedar-fence-eaten-by-
        # not_target failures while still routing real walls/buildings to
        # the absorber.
        if im.class_id in distractor_class_ids:
            fence_present = np.isin(class_map, list(fence_class_ids))
            # Where fence currently wins AND incoming absorber doesn't beat
            # it by the required margin, BLOCK the overwrite.
            fence_wins_margin = fence_present & (conf_map + ABSORBER_OVERRIDE_MARGIN > conf)
            write_mask = write_mask & ~fence_wins_margin

        class_map[write_mask] = im.class_id
        conf_map[write_mask] = conf
        prio_map[write_mask] = pri

    return class_map, conf_map


def mask_coverage_ratio(class_map: np.ndarray, class_id: int) -> float:
    """Fraction of image pixels assigned to the given class."""
    return float((class_map == class_id).sum()) / class_map.size


def per_class_pixel_counts(class_map: np.ndarray, num_classes: int) -> dict[int, int]:
    """Histogram: class_id → pixel count."""
    counts = np.bincount(class_map.ravel(), minlength=num_classes)
    return {int(c): int(counts[c]) for c in range(num_classes)}
