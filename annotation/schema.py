"""Load + validate the class schema YAML into typed dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ClassDef:
    id: int
    name: str
    tier: str                       # fence | distractor | occluder | context
    priority: int                   # higher wins on pixel overlap
    color: tuple[int, int, int]     # RGB for visualization
    is_staining_target: bool        # True only for paintable cedar/wood fence classes
    prompts: tuple[str, ...]
    box_threshold: float
    text_threshold: float


@dataclass(frozen=True)
class PipelineConfig:
    grounding_dino_model: str
    grounding_dino_max_prompt_len: int
    grounding_dino_batch_prompts: int
    sam2_model: str
    sam2_points_per_batch: int
    sam2_multimask_output: bool
    sam2_mask_threshold: float
    min_mask_area_pixels: int
    min_mask_area_ratio: float
    max_boxes_per_class_per_image: int
    iou_merge_threshold: float
    overlap_priority_strategy: str


@dataclass(frozen=True)
class QAConfig:
    min_overall_confidence: float
    min_fence_wood_confidence_for_positives: float
    flag_no_detections: bool
    flag_if_fence_wood_missing_in_positive: bool
    sample_rate_for_qa: float


@dataclass(frozen=True)
class Schema:
    version: str
    classes: tuple[ClassDef, ...]
    pipeline: PipelineConfig
    qa: QAConfig

    def by_id(self, class_id: int) -> ClassDef | None:
        for c in self.classes:
            if c.id == class_id:
                return c
        return None

    def by_name(self, name: str) -> ClassDef | None:
        for c in self.classes:
            if c.name == name:
                return c
        return None

    @property
    def staining_target_ids(self) -> tuple[int, ...]:
        """Class IDs that the staining visualizer should treat as paint-targets."""
        return tuple(c.id for c in self.classes if c.is_staining_target)

    @property
    def occluder_ids(self) -> tuple[int, ...]:
        return tuple(c.id for c in self.classes if c.tier == "occluder")

    @property
    def num_classes(self) -> int:
        # Includes background (ID 0)
        return max(c.id for c in self.classes) + 1


def load_schema(path: str | Path = "configs/annotation_schema.yaml") -> Schema:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    classes: list[ClassDef] = []
    seen_ids: set[int] = set()
    for c in raw["classes"]:
        cid = int(c["id"])
        if cid == 0:
            raise ValueError("Class ID 0 is reserved for background")
        if cid in seen_ids:
            raise ValueError(f"Duplicate class ID: {cid}")
        seen_ids.add(cid)
        color = tuple(c.get("color", [128, 128, 128]))
        if len(color) != 3:
            raise ValueError(f"Class {c['name']} color must be RGB triple")
        classes.append(ClassDef(
            id=cid,
            name=c["name"],
            tier=c["tier"],
            priority=int(c["priority"]),
            color=color,                           # type: ignore[arg-type]
            is_staining_target=bool(c.get("is_staining_target", False)),
            prompts=tuple(c["prompts"]),
            box_threshold=float(c.get("box_threshold", 0.30)),
            text_threshold=float(c.get("text_threshold", 0.22)),
        ))

    p = raw["pipeline"]
    pipeline = PipelineConfig(
        grounding_dino_model=p["grounding_dino"]["model"],
        grounding_dino_max_prompt_len=p["grounding_dino"]["max_text_prompt_length"],
        grounding_dino_batch_prompts=p["grounding_dino"]["batch_prompts_per_call"],
        sam2_model=p["sam2"]["model"],
        sam2_points_per_batch=p["sam2"]["points_per_batch"],
        sam2_multimask_output=bool(p["sam2"]["multimask_output"]),
        sam2_mask_threshold=float(p["sam2"]["mask_threshold"]),
        min_mask_area_pixels=int(p["min_mask_area_pixels"]),
        min_mask_area_ratio=float(p["min_mask_area_ratio"]),
        max_boxes_per_class_per_image=int(p["max_boxes_per_class_per_image"]),
        iou_merge_threshold=float(p["iou_merge_threshold"]),
        overlap_priority_strategy=p["overlap_priority_strategy"],
    )

    q = raw["qa"]
    qa = QAConfig(
        min_overall_confidence=float(q["min_overall_confidence"]),
        min_fence_wood_confidence_for_positives=float(
            q["min_fence_wood_confidence_for_positives"]),
        flag_no_detections=bool(q["flag_no_detections"]),
        flag_if_fence_wood_missing_in_positive=bool(
            q["flag_if_fence_wood_missing_in_positive"]),
        sample_rate_for_qa=float(q["sample_rate_for_qa"]),
    )

    return Schema(
        version=str(raw["version"]),
        classes=tuple(classes),
        pipeline=pipeline,
        qa=qa,
    )
