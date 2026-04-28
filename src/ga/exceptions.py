"""Typed exceptions for the GA engine."""
from __future__ import annotations


class GAError(Exception):
    """Base class for all GA-related errors."""


class ConfigError(GAError):
    """Configuration is malformed, inconsistent, or references missing resources."""


class AdapterError(GAError):
    """A model adapter failed to build, train, or evaluate."""


class AdapterNotImplementedError(AdapterError):
    """Adapter file exists but its build/train methods are stubs."""


class FitnessError(GAError):
    """Fitness evaluation failed (crash, timeout, OOM). Individual scores -inf."""


class CheckpointError(GAError):
    """Checkpoint save/load failed or state is corrupt."""


class PreflightError(GAError):
    """Environment/GPU/disk pre-flight check failed."""
