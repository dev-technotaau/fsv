"""Pydantic config models + YAML loader for the GA.

Two stage configs inherit from a base. YAML is the source of truth; CLI can
override any scalar via --set dotted.key=value.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, ConfigDict, field_validator

from .exceptions import ConfigError
from .registry import all_keys


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    images_dir: Path = Path("data/images")
    masks_dir: Path  = Path("data/masks")
    val_split: float = 0.15
    hard_eval_dir: Optional[Path] = None           # held-out "hard scenes" subset
    seed: int = 42


class FitnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: Literal["iou", "boundary_f1", "composite"] = "composite"
    iou_weight: float = 1.0
    boundary_f1_weight: float = 0.5
    tv_smoothness_penalty: float = 0.1          # penalizes blocky/jagged masks
    proxy_budget: Literal["epochs", "minutes", "iterations"] = "epochs"
    proxy_epochs: int = 10                      # short proxy training for GA fitness
    proxy_minutes: Optional[int] = None
    proxy_iterations: Optional[int] = None
    early_kill_iou: float = 0.10                # if IoU < this by mid-training, kill run
    early_kill_at_fraction: float = 0.3         # check at 30% of proxy budget


class GAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: Literal["stage1_model", "stage2_hyperparam"] = "stage1_model"

    # Population
    population_size: int = 12
    generations: int = 15
    elite_count: int = 2
    tournament_k: int = 3

    # Operators
    crossover_prob: float = 0.7
    mutation_prob: float = 0.3
    combo_mutation_rate: float = 0.05           # stage1 — chance to change combo_key
    param_mutation_rate: float = 0.3

    # Diversity / niching
    combo_pool: Optional[list[str]] = None      # None = all 18; else restrict to subset
    fixed_combo: Optional[str] = None           # stage2 — lock combo_key
    niching_quota: Optional[dict[str, int]] = None  # per-combo quota to preserve diversity

    # Misc
    seed: int = 42
    max_retries_per_individual: int = 1         # re-run crashed individuals this many times

    @field_validator("combo_pool")
    @classmethod
    def _check_pool(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is None:
            return v
        unknown = set(v) - set(all_keys())
        if unknown:
            raise ValueError(f"Unknown combo keys in combo_pool: {unknown}")
        return v

    @field_validator("fixed_combo")
    @classmethod
    def _check_fixed(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if v not in all_keys():
            raise ValueError(f"Unknown fixed_combo: {v}")
        return v


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_dir: Path = Path("runs/ga")
    checkpoint_every_gen: int = 1
    resume_from: Optional[Path] = None
    n_gpus: int = 1
    parallel_workers: int = 1                   # 1 = sequential; >1 runs individuals concurrently
    per_worker_gpu: Optional[list[int]] = None  # explicit gpu id per worker
    dry_run: bool = False
    log_to_tensorboard: bool = True
    log_to_csv: bool = True
    timeout_per_individual_minutes: int = 240   # kill run after this many minutes


class Config(BaseModel):
    """Top-level GA config."""
    model_config = ConfigDict(extra="forbid")
    data:    DataConfig    = Field(default_factory=DataConfig)
    fitness: FitnessConfig = Field(default_factory=FitnessConfig)
    ga:      GAConfig      = Field(default_factory=GAConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


# ---------- loaders ----------

def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _set_nested(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _coerce_cli_value(raw: str) -> Any:
    """Parse --set values: numbers, bools, null, plain strings."""
    low = raw.lower()
    if low in ("true", "yes"):   return True
    if low in ("false", "no"):   return False
    if low in ("null", "none"):  return None
    try:
        if "." in raw or "e" in low:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def load_config(
    path: Optional[Path] = None,
    overrides: Optional[list[str]] = None,
) -> Config:
    """Load a YAML config, apply CLI overrides, validate."""
    raw: dict[str, Any] = {}
    if path is not None:
        raw = load_yaml(path)
    for ov in overrides or []:
        if "=" not in ov:
            raise ConfigError(f"--set overrides must be dotted.key=value, got: {ov!r}")
        k, v = ov.split("=", 1)
        _set_nested(raw, k.strip(), _coerce_cli_value(v.strip()))
    try:
        return Config(**raw)
    except Exception as e:
        raise ConfigError(f"Config validation failed: {e}") from e
