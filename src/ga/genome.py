"""Genome representation + mutation/crossover operators.

A Genome is a dict-like object:
  {"combo_key": "01_dinov2_l_m2f", "params": {<param_name>: <value>, ...}}

Stage 1 (model search): `combo_key` mutates, `params` is re-sampled from combo's search space.
Stage 2 (hyperparam search): `combo_key` fixed, only `params` mutate/crossover.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field, asdict
from typing import Any

from .registry import COMBOS, get_combo, get_full_search_space, all_keys


# ---------- genome dataclass ----------

@dataclass
class Genome:
    combo_key: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Genome":
        return cls(combo_key=d["combo_key"], params=dict(d.get("params", {})))

    def stable_hash(self) -> str:
        """Deterministic hash for fitness caching. Rounds floats to 6 decimals
        so near-identical genomes collide."""
        norm = {"combo_key": self.combo_key, "params": {}}
        for k in sorted(self.params.keys()):
            v = self.params[k]
            if isinstance(v, float):
                v = round(v, 6)
            norm["params"][k] = v
        blob = json.dumps(norm, sort_keys=True).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:16]

    def short_label(self) -> str:
        return f"{self.combo_key}[{self.stable_hash()}]"


# ---------- sampling from search space ----------

def sample_value(spec: dict[str, Any], rng: random.Random) -> Any:
    t = spec["type"]
    if t == "float":
        low, high = spec["low"], spec["high"]
        if spec.get("log"):
            return math.exp(rng.uniform(math.log(low), math.log(high)))
        return rng.uniform(low, high)
    if t == "int":
        return rng.randint(spec["low"], spec["high"])
    if t == "choice":
        return rng.choice(spec["values"])
    if t == "bool":
        return rng.random() < 0.5
    raise ValueError(f"Unknown param spec type: {t}")


def sample_params(combo_key: str, rng: random.Random) -> dict[str, Any]:
    space = get_full_search_space(combo_key)
    return {name: sample_value(spec, rng) for name, spec in space.items()}


def random_genome(rng: random.Random, combo_pool: list[str] | None = None) -> Genome:
    pool = combo_pool or all_keys()
    key = rng.choice(pool)
    return Genome(combo_key=key, params=sample_params(key, rng))


# ---------- mutation ----------

def mutate_param(name: str, value: Any, spec: dict[str, Any], rng: random.Random) -> Any:
    """Mutate a single parameter. Gaussian for continuous, resample for categorical/bool."""
    t = spec["type"]
    if t == "float":
        low, high = spec["low"], spec["high"]
        if spec.get("log"):
            # mutate in log space: multiplicative noise
            log_v = math.log(max(value, 1e-20))
            sigma = 0.3 * (math.log(high) - math.log(low))
            new = math.exp(log_v + rng.gauss(0, sigma))
        else:
            sigma = 0.15 * (high - low)
            new = value + rng.gauss(0, sigma)
        return max(low, min(high, new))
    if t == "int":
        low, high = spec["low"], spec["high"]
        sigma = max(1.0, 0.15 * (high - low))
        new = int(round(value + rng.gauss(0, sigma)))
        return max(low, min(high, new))
    if t == "choice":
        return rng.choice(spec["values"])
    if t == "bool":
        return not bool(value)
    raise ValueError(f"Unknown param spec type: {t}")


def mutate_genome(
    g: Genome,
    rng: random.Random,
    *,
    combo_mutation_rate: float = 0.05,
    param_mutation_rate: float = 0.3,
    combo_pool: list[str] | None = None,
) -> Genome:
    """Mutate a genome. Combo-key change is rare (big effect), param tweak is common."""
    # Optional combo-key flip
    if rng.random() < combo_mutation_rate:
        pool = combo_pool or all_keys()
        other = [k for k in pool if k != g.combo_key] or pool
        new_key = rng.choice(other)
        return Genome(combo_key=new_key, params=sample_params(new_key, rng))

    # Param-only mutation
    space = get_full_search_space(g.combo_key)
    new_params = dict(g.params)
    for name, spec in space.items():
        if name not in new_params:
            new_params[name] = sample_value(spec, rng)
            continue
        if rng.random() < param_mutation_rate:
            new_params[name] = mutate_param(name, new_params[name], spec, rng)
    return Genome(combo_key=g.combo_key, params=new_params)


# ---------- crossover ----------

def crossover_uniform(a: Genome, b: Genome, rng: random.Random) -> tuple[Genome, Genome]:
    """Uniform crossover. If parents have different combo_keys, children inherit
    one parent's combo_key and all its params (param spaces differ across combos)."""
    if a.combo_key != b.combo_key:
        # can't meaningfully cross params from different spaces
        # swap identities 50/50
        if rng.random() < 0.5:
            return Genome(a.combo_key, dict(a.params)), Genome(b.combo_key, dict(b.params))
        return Genome(b.combo_key, dict(b.params)), Genome(a.combo_key, dict(a.params))

    # same combo_key — uniform crossover on params
    space = get_full_search_space(a.combo_key)
    c1_params, c2_params = {}, {}
    for name in space:
        va = a.params.get(name)
        vb = b.params.get(name)
        if rng.random() < 0.5:
            c1_params[name], c2_params[name] = va, vb
        else:
            c1_params[name], c2_params[name] = vb, va
        # if one parent was missing this param, ensure children both have valid values
        if c1_params[name] is None:
            c1_params[name] = sample_value(space[name], rng)
        if c2_params[name] is None:
            c2_params[name] = sample_value(space[name], rng)
    return (
        Genome(a.combo_key, c1_params),
        Genome(a.combo_key, c2_params),
    )
