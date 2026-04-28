"""Tests for CircuitBreaker state machine."""
from __future__ import annotations

import asyncio

import pytest

from data_scraper.circuit_breaker import CircuitBreaker, CircuitOpenError, State


@pytest.mark.asyncio
async def test_closed_allows_calls():
    cb = CircuitBreaker("t", failure_threshold=3, cool_down_s=10)
    async def ok(): return "ok"
    assert await cb.call(ok) == "ok"
    assert cb.state is State.CLOSED


@pytest.mark.asyncio
async def test_opens_after_threshold():
    cb = CircuitBreaker("t", failure_threshold=3, cool_down_s=10)
    async def fail(): raise RuntimeError("nope")
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await cb.call(fail)
    assert cb.state is State.OPEN
    with pytest.raises(CircuitOpenError):
        await cb.call(fail)


@pytest.mark.asyncio
async def test_half_open_then_close_on_success():
    cb = CircuitBreaker("t", failure_threshold=2, cool_down_s=0.1)
    async def fail(): raise RuntimeError
    async def ok(): return "ok"
    # trip
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(fail)
    assert cb.state is State.OPEN
    # wait for half-open window
    await asyncio.sleep(0.15)
    assert await cb.call(ok) == "ok"
    assert cb.state is State.CLOSED


@pytest.mark.asyncio
async def test_half_open_reopens_on_failure():
    cb = CircuitBreaker("t", failure_threshold=2, cool_down_s=0.1)
    async def fail(): raise RuntimeError
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(fail)
    await asyncio.sleep(0.15)
    # probe fails → should go back to OPEN
    with pytest.raises(RuntimeError):
        await cb.call(fail)
    assert cb.state is State.OPEN
