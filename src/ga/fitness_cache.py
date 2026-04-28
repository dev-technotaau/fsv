"""Fitness cache — memoize (genome_hash) -> fitness_dict.

Persisted to disk as JSON Lines so concurrent workers can append safely.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Optional


class FitnessCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._mem: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._mem[rec["hash"]] = rec
                except json.JSONDecodeError:
                    # tolerate truncated last line
                    continue

    def get(self, genome_hash: str) -> Optional[dict[str, Any]]:
        return self._mem.get(genome_hash)

    def put(self, genome_hash: str, record: dict[str, Any]) -> None:
        with self._lock:
            rec = {"hash": genome_hash, **record}
            self._mem[genome_hash] = rec
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())

    def __contains__(self, genome_hash: str) -> bool:
        return genome_hash in self._mem

    def __len__(self) -> int:
        return len(self._mem)
