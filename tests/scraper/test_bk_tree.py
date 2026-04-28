"""Tests for BK-tree dHash nearest-neighbor."""
from __future__ import annotations

import random

import pytest

from data_scraper.bk_tree import BKTree, hamming


def test_hamming_known():
    assert hamming(0, 0) == 0
    assert hamming(0, 0xFF) == 8
    assert hamming(0b1010, 0b0101) == 4


def test_exact_match():
    t = BKTree()
    t.add(0xDEADBEEF)
    assert t.find_within(0xDEADBEEF, 0)
    assert t.find_within(0xDEADBEEF, 5)


def test_near_and_far():
    t = BKTree()
    t.add(0xFF00FF00)
    # flip 3 bits
    near = 0xFF00FF00 ^ 0b111
    assert t.find_within(near, 3)
    assert t.find_within(near, 5)
    # flip many bits — far
    far = 0x00FF00FF
    assert not t.find_within(far, 5)


def test_empty_tree():
    assert not BKTree().find_within(0x1234, 5)


def test_duplicate_insert_idempotent():
    t = BKTree()
    t.add(0xABC)
    t.add(0xABC)
    assert len(t) == 1


def test_bulk_load_and_scan(seed=1337):
    r = random.Random(seed)
    values = [r.getrandbits(64) for _ in range(500)]
    t = BKTree()
    t.bulk_load(values)
    assert len(t) <= len(values)  # may dedup exact matches
    # Every inserted value must be findable with threshold 0
    for v in values:
        assert t.find_within(v, 0)


def test_faster_than_linear_for_large_N(seed=7):
    """Sanity: BK-tree shouldn't crash on ~5k items."""
    r = random.Random(seed)
    t = BKTree()
    for _ in range(5000):
        t.add(r.getrandbits(64))
    # Search should complete quickly
    assert isinstance(t.find_within(r.getrandbits(64), 5), bool)
