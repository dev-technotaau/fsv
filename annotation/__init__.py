"""Multi-class annotation pipeline (Grounded-SAM 2) for fence segmentation.

Public API:
    from annotation import AnnotationPipeline, load_schema
"""
from annotation.schema import ClassDef, Schema, load_schema
from annotation.pipeline import AnnotationPipeline

__all__ = ["ClassDef", "Schema", "load_schema", "AnnotationPipeline"]
