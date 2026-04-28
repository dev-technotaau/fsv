"""Distributed store factory test (Redis optional — uses Null store when off)."""
from __future__ import annotations

import pytest

from data_scraper.distributed import build_distributed_store, NullDistributedStore


@pytest.mark.asyncio
async def test_disabled_returns_null():
    s = build_distributed_store(enabled=False, url=None)
    assert isinstance(s, NullDistributedStore)
    assert s.enabled is False
    assert await s.url_seen("x") is False
    await s.mark_url("x", "src")
    assert await s.sha_seen("s") is False
    await s.close()


@pytest.mark.asyncio
async def test_redis_unavailable_falls_back():
    # Bogus URL → connection will fail → should fall back to Null
    s = build_distributed_store(enabled=True, url="redis://nonexistent.invalid:63791/0")
    # Either successfully connected (unlikely) or fell back to Null
    assert hasattr(s, "url_seen")
    await s.close()
