"""Batched SQLite writer — background task consumes a queue, flushes periodically.

Reduces single-writer lock contention from N workers each calling INSERT
down to one transaction every `flush_interval_s` or every `batch_size` items.
"""
from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional


class BatchedSQLiteWriter:
    """Writer task that batches writes. Thread-safe via asyncio.Queue + run_in_executor for DB I/O.

    Writes items shaped as (sql, params_tuple) tuples. The writer opens its own
    connection (WAL mode) and batches into transactions.
    """

    def __init__(self, db_path: Path, batch_size: int = 50,
                 flush_interval_s: float = 0.5, queue_maxsize: int = 5000):
        self.db_path = db_path
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._written = 0
        self._errors = 0

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def enqueue(self, sql: str, params: tuple) -> None:
        await self.queue.put((sql, params))

    async def drain(self) -> None:
        """Wait for queue to drain."""
        while not self.queue.empty():
            await asyncio.sleep(0.05)

    async def stop(self) -> None:
        self._stop.set()
        # Poison pill
        try:
            self.queue.put_nowait(None)  # type: ignore
        except asyncio.QueueFull:
            pass
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except asyncio.TimeoutError:
                self._task.cancel()

    @property
    def stats(self) -> dict[str, int]:
        return {"written": self._written, "errors": self._errors, "queued": self.queue.qsize()}

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        conn = await loop.run_in_executor(None, self._open_conn)
        try:
            buffer: list[tuple[str, tuple]] = []
            last_flush = time.monotonic()
            while not self._stop.is_set() or not self.queue.empty():
                timeout = max(0.05, self.flush_interval_s - (time.monotonic() - last_flush))
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    item = None  # periodic flush tick
                if item is None:
                    if buffer:
                        await loop.run_in_executor(None, self._flush, conn, buffer)
                        buffer.clear()
                    last_flush = time.monotonic()
                    continue
                buffer.append(item)
                if len(buffer) >= self.batch_size:
                    await loop.run_in_executor(None, self._flush, conn, buffer)
                    buffer.clear()
                    last_flush = time.monotonic()
            # final flush on shutdown
            if buffer:
                await loop.run_in_executor(None, self._flush, conn, buffer)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _open_conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _flush(self, conn, buffer: list[tuple[str, tuple]]) -> None:
        try:
            conn.execute("BEGIN")
            for sql, params in buffer:
                conn.execute(sql, params)
            conn.execute("COMMIT")
            self._written += len(buffer)
        except sqlite3.Error as e:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            self._errors += 1
            # Replay one by one to isolate bad rows; skip failures
            for sql, params in buffer:
                try:
                    conn.execute(sql, params)
                    self._written += 1
                except sqlite3.Error:
                    self._errors += 1
