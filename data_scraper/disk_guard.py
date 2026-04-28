"""Disk-space guard. Checks free space at startup and periodically during run.

If free space drops below `min_free_gb`, signals the coordinator to stop gracefully.
"""
from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path
from typing import Callable

from .logger import get_logger

log = get_logger("disk_guard")


class DiskGuard:
    def __init__(self, path: Path, min_free_gb: float = 2.0, check_interval_s: float = 30.0):
        self.path = path
        self.min_free_gb = min_free_gb
        self.check_interval_s = check_interval_s
        self._stop_signal: asyncio.Event = asyncio.Event()

    def free_gb(self) -> float:
        try:
            du = shutil.disk_usage(str(self.path))
            return du.free / (1024 ** 3)
        except Exception:
            return float("inf")   # fail-open: don't stop if we can't check

    def check(self) -> bool:
        """Returns True if OK, False if low."""
        free = self.free_gb()
        if free < self.min_free_gb:
            log.error("disk_space_low", path=str(self.path), free_gb=round(free, 2),
                      min_gb=self.min_free_gb)
            return False
        return True

    async def monitor(self, should_stop: Callable[[], bool]) -> None:
        """Background task. Runs until should_stop() returns True or we run out of disk."""
        while not should_stop():
            if not self.check():
                self._stop_signal.set()
                return
            await asyncio.sleep(self.check_interval_s)

    @property
    def stop_signal(self) -> asyncio.Event:
        return self._stop_signal

    def preflight(self) -> None:
        """Raises RuntimeError if initial free space is already below threshold."""
        free = self.free_gb()
        if free < self.min_free_gb:
            raise RuntimeError(
                f"Disk preflight failed: {free:.2f} GB free < {self.min_free_gb} GB required "
                f"at {self.path}"
            )
        log.info("disk_preflight_ok", path=str(self.path), free_gb=round(free, 2))
