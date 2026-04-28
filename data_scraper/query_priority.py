"""Query-gap-aware priority queue.

Sources register the queries they'll emit; as images are saved, the coordinator
feeds yield counts back into this scheduler. Queries with low yield get
priority boosted so they're attempted first by sources that support
dynamic query pulls.

Use-case: you want a scraper that notices "we only have 3 images for 'fence
with dog'" and starts favoring that query across all remaining source iterations.
"""
from __future__ import annotations

import asyncio
import heapq
import itertools
import threading
from collections import defaultdict
from typing import Optional


class PriorityQueryScheduler:
    """Priority-by-shortage query dispenser.

    score(query) = -(saved_count) + small penalty per attempt → low-saved wins.
    Thread-safe.
    """

    def __init__(self, queries: list[str], target_per_query: int = 100):
        self.target_per_query = target_per_query
        self._lock = threading.Lock()
        self._saved: dict[str, int] = defaultdict(int)
        self._attempts: dict[str, int] = defaultdict(int)
        self._queries = list(queries)
        self._counter = itertools.count()   # tiebreaker for heap stability

    # ---------- feedback ----------
    def record_saved(self, query: Optional[str], n: int = 1) -> None:
        if not query:
            return
        with self._lock:
            self._saved[query] += n

    def record_attempt(self, query: Optional[str]) -> None:
        if not query:
            return
        with self._lock:
            self._attempts[query] += 1

    # ---------- pull ----------
    def next_query(self, exclude: Optional[set[str]] = None) -> Optional[str]:
        """Return the highest-priority query not in `exclude` (already-in-use by caller)."""
        with self._lock:
            # Build a lightweight priority list:
            #   priority = saved (lower is better) × 1000 + attempts (lower is better)
            best: tuple[int, int, str] | None = None
            for q in self._queries:
                if exclude and q in exclude:
                    continue
                saved = self._saved.get(q, 0)
                if saved >= self.target_per_query:
                    continue      # already met target
                score = saved * 1000 + self._attempts.get(q, 0)
                if best is None or score < best[0]:
                    best = (score, next(self._counter), q)
            return best[2] if best else None

    def reorder_for_shortage(self) -> list[str]:
        """Return queries sorted by (saved asc, attempts asc). Useful for sources that
        iterate linearly rather than pulling one-at-a-time."""
        with self._lock:
            return sorted(self._queries,
                          key=lambda q: (self._saved.get(q, 0), self._attempts.get(q, 0)))

    def snapshot(self) -> dict:
        with self._lock:
            below_target = sum(1 for q in self._queries if self._saved.get(q, 0) < self.target_per_query)
            return {
                "total_queries": len(self._queries),
                "below_target": below_target,
                "mean_saved_per_query": (
                    sum(self._saved.values()) / max(1, len(self._queries))
                ),
                "most_scarce": sorted(self._queries,
                                       key=lambda q: self._saved.get(q, 0))[:5],
            }
