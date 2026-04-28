"""Confidence scoring + QA-queue prioritization for human review.

After automatic annotation, each image gets a numeric confidence score plus
set of flags. Low-score images are pushed to the QA queue so human annotators
spend their time where the auto-pipeline is least certain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from annotation.fusion import per_class_pixel_counts, mask_coverage_ratio
from annotation.sam2 import InstanceMask
from annotation.schema import Schema


@dataclass
class QAScore:
    image_id: str
    image_class: str                     # "pos" | "neg" from manifest
    overall_confidence: float            # weighted mean of all kept masks
    n_detections: int
    fence_wood_coverage: float           # pixel ratio of fence_wood class
    fence_wood_confidence: float | None  # None if class not present
    flags: list[str] = field(default_factory=list)
    needs_review: bool = False
    per_class_counts: dict[int, int] = field(default_factory=dict)


def score_annotation(
    image_id: str,
    image_class: str,                    # manifest-level pos/neg
    instance_masks: list[InstanceMask],
    class_map: np.ndarray,
    schema: Schema,
    sample_for_qa: bool = False,
) -> QAScore:
    """Compute QA metrics + flags for a single annotated image."""
    flags: list[str] = []

    if not instance_masks:
        flags.append("no_detections")
        return QAScore(
            image_id=image_id, image_class=image_class,
            overall_confidence=0.0, n_detections=0,
            fence_wood_coverage=0.0, fence_wood_confidence=None,
            flags=flags, needs_review=True,
            per_class_counts=per_class_pixel_counts(class_map, schema.num_classes),
        )

    # Weighted mean confidence by mask area (so tiny spurious detections don't
    # drag the overall score — the important thing is the big masks' quality)
    total_area = sum(im.area_pixels for im in instance_masks)
    if total_area > 0:
        overall_conf = sum(
            (im.sam_score * im.detection_score) * im.area_pixels
            for im in instance_masks
        ) / total_area
    else:
        overall_conf = 0.0

    # fence_wood (class ID 1) analysis
    fence_wood_coverage = mask_coverage_ratio(class_map, 1)
    fence_wood_confs = [
        im.sam_score * im.detection_score
        for im in instance_masks if im.class_id == 1
    ]
    fence_wood_conf = (
        float(np.mean(fence_wood_confs)) if fence_wood_confs else None
    )

    qa_cfg = schema.qa
    if overall_conf < qa_cfg.min_overall_confidence:
        flags.append(f"low_overall_conf={overall_conf:.3f}")
    if image_class == "pos":
        if fence_wood_conf is None and qa_cfg.flag_if_fence_wood_missing_in_positive:
            flags.append("fence_wood_missing_in_positive")
        elif fence_wood_conf is not None and \
                fence_wood_conf < qa_cfg.min_fence_wood_confidence_for_positives:
            flags.append(f"weak_fence_wood_conf={fence_wood_conf:.3f}")

    if sample_for_qa:
        flags.append("random_qa_sample")

    return QAScore(
        image_id=image_id, image_class=image_class,
        overall_confidence=float(overall_conf),
        n_detections=len(instance_masks),
        fence_wood_coverage=float(fence_wood_coverage),
        fence_wood_confidence=fence_wood_conf,
        flags=flags,
        needs_review=bool(flags),
        per_class_counts=per_class_pixel_counts(class_map, schema.num_classes),
    )
