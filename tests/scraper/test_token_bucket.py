"""Tests for adaptive TokenBucket."""
from __future__ import annotations

import asyncio
import time

import pytest

from data_scraper.sources.base import TokenBucket


@pytest.mark.asyncio
async def test_immediate_acquire_while_full():
    tb = TokenBucket(per_minute=600)  # 10/sec
    # Capacity starts full — first call should be near-instant
    t = time.monotonic()
    await tb.acquire()
    assert time.monotonic() - t < 0.1


@pytest.mark.asyncio
async def test_retry_after_seconds():
    tb = TokenBucket(per_minute=6000)
    tb.adjust_from_headers({"Retry-After": "0.2"})
    t = time.monotonic()
    await tb.acquire()
    assert time.monotonic() - t >= 0.15   # respected pause


@pytest.mark.asyncio
async def test_ratelimit_remaining_zero():
    tb = TokenBucket(per_minute=6000)
    import time as _t
    reset_in = _t.time() + 0.15
    tb.adjust_from_headers({
        "X-RateLimit-Remaining": "0",
        "X-RateLimit-Reset": str(reset_in),
    })
    t = _t.monotonic()
    await tb.acquire()
    # Should have waited most of 0.15s
    assert _t.monotonic() - t >= 0.10


@pytest.mark.asyncio
async def test_throttle_multiplier_reduces_rate():
    tb = TokenBucket(per_minute=600)    # 10/s
    tb.throttle_multiplier(0.1)         # now 1/s
    # drain initial bucket
    for _ in range(10):
        await tb.acquire()
    t = time.monotonic()
    await tb.acquire()
    # Next token takes ~1s with reduced rate
    assert time.monotonic() - t >= 0.5
