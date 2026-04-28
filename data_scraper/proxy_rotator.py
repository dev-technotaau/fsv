"""Proxy rotation. Loads a pool from config or env (SCRAPER_PROXIES=url1,url2,...),
rotates per-request, tracks failures, evicts bad proxies for a cool-down period.
"""
from __future__ import annotations

import itertools
import os
import threading
import time
from typing import Optional


class ProxyRotator:
    def __init__(self, proxies: Optional[list[str]] = None, failure_cool_down_s: float = 300.0):
        self.failure_cool_down_s = failure_cool_down_s
        self._lock = threading.Lock()
        self._proxies: list[str] = list(proxies or [])
        self._env_proxies()
        self._failures: dict[str, float] = {}            # proxy → cooldown_until_monotonic
        self._cycle = itertools.cycle(self._proxies) if self._proxies else None

    def _env_proxies(self) -> None:
        env = os.environ.get("SCRAPER_PROXIES", "").strip()
        if env:
            extras = [p.strip() for p in env.split(",") if p.strip()]
            for p in extras:
                if p not in self._proxies:
                    self._proxies.append(p)
            self._cycle = itertools.cycle(self._proxies) if self._proxies else None

    @property
    def enabled(self) -> bool:
        return bool(self._proxies)

    def next_proxy(self) -> Optional[str]:
        if not self.enabled or self._cycle is None:
            return None
        with self._lock:
            now = time.monotonic()
            for _ in range(len(self._proxies)):
                p = next(self._cycle)
                until = self._failures.get(p, 0)
                if until <= now:
                    return p
        return None

    def mark_failure(self, proxy: str) -> None:
        if not proxy:
            return
        with self._lock:
            self._failures[proxy] = time.monotonic() + self.failure_cool_down_s

    def mark_success(self, proxy: str) -> None:
        if not proxy:
            return
        with self._lock:
            self._failures.pop(proxy, None)

    def snapshot(self) -> dict:
        now = time.monotonic()
        return {
            "count": len(self._proxies),
            "active": sum(1 for p in self._proxies if self._failures.get(p, 0) <= now),
            "cooling": sum(1 for p in self._proxies if self._failures.get(p, 0) > now),
        }
