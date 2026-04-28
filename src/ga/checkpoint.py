"""Checkpoint save/resume for the GA state.

Atomic writes via temp-file + rename. Pickles the RNG + population.
"""
from __future__ import annotations

import json
import os
import pickle
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .exceptions import CheckpointError
from .genome import Genome


CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: Path,
    *,
    generation: int,
    population: list[tuple[Genome, float]],     # list of (genome, fitness)
    rng: random.Random,
    config_dict: dict[str, Any],
    hall_of_fame: list[tuple[Genome, float, dict[str, float]]],
) -> None:
    """Atomic save of GA state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "version": CHECKPOINT_VERSION,
        "generation": generation,
        "population": [(g.to_dict(), fit) for g, fit in population],
        "rng_state": rng.getstate(),
        "config": config_dict,
        "hall_of_fame": [(g.to_dict(), fit, met) for g, fit, met in hall_of_fame],
    }
    # atomic write
    fd, tmp_path = tempfile.mkstemp(prefix="ga_ckpt_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise CheckpointError(f"Failed to save checkpoint {path}: {e}") from e

    # also a human-readable JSON sibling for quick inspection
    json_path = path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        human = dict(state)
        human.pop("rng_state", None)
        json.dump(human, f, indent=2, default=str)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise CheckpointError(f"Checkpoint not found: {path}")
    try:
        with path.open("rb") as f:
            state = pickle.load(f)
    except Exception as e:
        raise CheckpointError(f"Failed to load checkpoint {path}: {e}") from e
    if state.get("version") != CHECKPOINT_VERSION:
        raise CheckpointError(
            f"Checkpoint version {state.get('version')} != supported {CHECKPOINT_VERSION}"
        )
    state["population"] = [
        (Genome.from_dict(gd), fit) for gd, fit in state["population"]
    ]
    state["hall_of_fame"] = [
        (Genome.from_dict(gd), fit, met) for gd, fit, met in state["hall_of_fame"]
    ]
    return state
