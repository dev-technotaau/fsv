"""Source interface + adaptive-rate TokenBucket + helpers for per-query progress."""
from __future__ import annotations

import abc
import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional


@dataclass
class URLCandidate:
    """A candidate image URL discovered by a source."""
    url: str
    source: str
    query: Optional[str] = None
    origin_page: Optional[str] = None
    title: Optional[str] = None
    extra: dict = field(default_factory=dict)


class TokenBucket:
    """Async rate-limiter with adaptive mode.

    `adjust_from_headers(headers)` lets us shrink the rate dynamically when the
    server tells us we're going too fast (429 / X-RateLimit-Remaining=0).
    """
    def __init__(self, per_minute: int):
        self.capacity = max(1, per_minute)
        self._tokens = float(per_minute)
        self._last = time.monotonic()
        self._rate_per_second = per_minute / 60.0
        self._lock = asyncio.Lock()
        self._forced_wait_until: float = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            # Respect forced wait (set by adjust_from_headers on 429)
            now = time.monotonic()
            if self._forced_wait_until > now:
                await asyncio.sleep(self._forced_wait_until - now)
                now = time.monotonic()
            while True:
                elapsed = now - self._last
                self._last = now
                self._tokens = min(self.capacity, self._tokens + elapsed * self._rate_per_second)
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                sleep_s = max(0.05, (1 - self._tokens) / self._rate_per_second)
                await asyncio.sleep(sleep_s)
                now = time.monotonic()

    def adjust_from_headers(self, headers: dict) -> None:
        """Adapt from response headers:
          Retry-After          → pause until then
          X-RateLimit-Remaining → if 0 and Reset present, pause until Reset
          X-RateLimit-Reset    → epoch seconds or seconds-from-now
        """
        now = time.monotonic()

        def _get(k: str) -> Optional[str]:
            v = headers.get(k) or headers.get(k.lower())
            return v

        retry_after = _get("Retry-After")
        if retry_after:
            try:
                secs = float(retry_after)
            except ValueError:
                try:
                    from email.utils import parsedate_to_datetime
                    secs = max(0, parsedate_to_datetime(retry_after).timestamp() - time.time())
                except Exception:
                    secs = 0
            self._forced_wait_until = max(self._forced_wait_until, now + secs)

        remaining = _get("X-RateLimit-Remaining") or _get("X-Rate-Limit-Remaining")
        reset = _get("X-RateLimit-Reset") or _get("X-Rate-Limit-Reset")
        if remaining == "0" and reset:
            try:
                v = float(reset)
                secs = max(0, v - time.time()) if v > 1_000_000_000 else v
                self._forced_wait_until = max(self._forced_wait_until, now + secs)
            except ValueError:
                pass

    def throttle_multiplier(self, factor: float) -> None:
        """Reduce rate by `factor` (e.g., 0.5 = halve). Permanent until reset."""
        self._rate_per_second = max(0.01, self._rate_per_second * factor)


class Source(abc.ABC):
    """Abstract source. Subclasses yield URLCandidate via discover()."""
    name: str = "base"

    def __init__(self, queries: list[str], downloader, dedup, cfg, bucket: Optional[TokenBucket] = None):
        self.queries = queries
        self.downloader = downloader
        self.dedup = dedup
        self.cfg = cfg
        self.bucket = bucket or TokenBucket(cfg.rate_limit_per_minute)
        self.enabled = cfg.enabled

    async def throttle(self) -> None:
        await self.bucket.acquire()

    # ---------- per-query progress helpers ----------
    def get_start_page(self, query: str, default: int = 1) -> int:
        """Return the page to start from. Skips fully-completed queries (returns None)."""
        last_page, completed = self.dedup.query_progress(self.name, query)
        if completed:
            return -1          # sentinel: skip entirely
        return max(default, last_page + 1)

    def update_progress(self, query: str, page: int, completed: bool = False) -> None:
        self.dedup.update_query_progress(self.name, query, page, completed)

    @abc.abstractmethod
    async def discover(self) -> AsyncIterator[URLCandidate]:
        if False:
            yield  # type: ignore

    async def close(self) -> None:
        pass
