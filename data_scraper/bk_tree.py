"""BK-tree for O(log N) Hamming-distance nearest-neighbor search.

Far faster than linear scan for dHash dedup when N > ~10k.

Usage:
    tree = BKTree()
    for dh in existing_dhashes:
        tree.add(dh)
    if tree.find_within(new_dh, threshold=5):
        # near-duplicate
"""
from __future__ import annotations

import threading
from typing import Optional


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class _Node:
    __slots__ = ("value", "children")

    def __init__(self, value: int):
        self.value = value
        self.children: dict[int, "_Node"] = {}


class BKTree:
    """Thread-safe BK-tree over 64-bit integer hashes (dHash output fits).
    Insertion: O(log N) average.
    Query `find_within(v, t)`: returns True early if any node within Hamming ≤ t.
    """
    def __init__(self, distance_fn=hamming):
        self._root: Optional[_Node] = None
        self._dist = distance_fn
        self._lock = threading.Lock()
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, value: int) -> None:
        with self._lock:
            if self._root is None:
                self._root = _Node(value)
                self._size = 1
                return
            node = self._root
            while True:
                d = self._dist(value, node.value)
                if d == 0:
                    return        # exact duplicate — don't insert
                child = node.children.get(d)
                if child is None:
                    node.children[d] = _Node(value)
                    self._size += 1
                    return
                node = child

    def find_within(self, value: int, threshold: int) -> bool:
        """Return True if any stored value has Hamming distance ≤ threshold."""
        if self._root is None:
            return False
        stack = [self._root]
        while stack:
            node = stack.pop()
            d = self._dist(value, node.value)
            if d <= threshold:
                return True
            lo = max(1, d - threshold)
            hi = d + threshold
            # Only recurse into children whose edge-distance is in [lo, hi]
            for edge, child in node.children.items():
                if lo <= edge <= hi:
                    stack.append(child)
        return False

    def bulk_load(self, values) -> None:
        """Load many values (for warm-starting from SQLite)."""
        for v in values:
            self.add(v)
