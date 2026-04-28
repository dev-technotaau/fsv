"""Per-source async circuit breaker.

States:
  CLOSED   — normal. Failures counted.
  OPEN     — short-circuit all calls (immediately raise CircuitOpenError).
  HALF_OPEN — after cool-down, allow one probe. Success → CLOSED. Failure → OPEN again.
"""
from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Any, Callable


class CircuitOpenError(RuntimeError):
    pass


class State(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        cool_down_s: float = 60.0,
        half_open_max_probes: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cool_down_s = cool_down_s
        self.half_open_max_probes = half_open_max_probes
        self._state: State = State.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float = 0.0
        self._probes_in_flight = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> State:
        return self._state

    async def _maybe_half_open(self) -> None:
        if self._state is State.OPEN and time.monotonic() - self._opened_at >= self.cool_down_s:
            self._state = State.HALF_OPEN
            self._probes_in_flight = 0

    async def call(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """Wrap an async call. Raises CircuitOpenError if circuit is OPEN."""
        async with self._lock:
            await self._maybe_half_open()
            if self._state is State.OPEN:
                raise CircuitOpenError(f"circuit_open:{self.name}")
            if self._state is State.HALF_OPEN:
                if self._probes_in_flight >= self.half_open_max_probes:
                    raise CircuitOpenError(f"circuit_half_open_busy:{self.name}")
                self._probes_in_flight += 1
        try:
            result = await fn(*args, **kwargs)
        except Exception:
            async with self._lock:
                self._consecutive_failures += 1
                if self._state is State.HALF_OPEN:
                    self._probes_in_flight -= 1
                    self._state = State.OPEN
                    self._opened_at = time.monotonic()
                elif self._consecutive_failures >= self.failure_threshold:
                    self._state = State.OPEN
                    self._opened_at = time.monotonic()
            raise
        else:
            async with self._lock:
                if self._state is State.HALF_OPEN:
                    self._probes_in_flight -= 1
                self._state = State.CLOSED
                self._consecutive_failures = 0
            return result

    def force_open(self, cool_down_s: float) -> None:
        """Externally trip the breaker (e.g., on a 429 with long Retry-After)."""
        self._state = State.OPEN
        self._opened_at = time.monotonic() - (self.cool_down_s - cool_down_s)

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self._state.value,
            "consecutive_failures": self._consecutive_failures,
            "time_in_state_s": time.monotonic() - self._opened_at if self._state != State.CLOSED else 0,
        }
