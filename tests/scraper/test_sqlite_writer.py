"""Tests for BatchedSQLiteWriter."""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from data_scraper.sqlite_writer import BatchedSQLiteWriter


@pytest.mark.asyncio
async def test_batched_inserts(tmp_path: Path):
    db = tmp_path / "t.db"
    # create table
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t(x INTEGER)")
    conn.commit()
    conn.close()

    w = BatchedSQLiteWriter(db, batch_size=10, flush_interval_s=0.05)
    await w.start()
    for i in range(25):
        await w.enqueue("INSERT INTO t VALUES(?)", (i,))
    await asyncio.sleep(0.3)  # let batched flushes run
    await w.drain()
    await w.stop()

    conn = sqlite3.connect(db)
    count = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    conn.close()
    assert count == 25


@pytest.mark.asyncio
async def test_bad_row_isolated(tmp_path: Path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t(x INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    w = BatchedSQLiteWriter(db, batch_size=5, flush_interval_s=0.05)
    await w.start()
    for i in range(3):
        await w.enqueue("INSERT INTO t VALUES(?)", (i,))
    # This row conflicts with first
    await w.enqueue("INSERT INTO t VALUES(?)", (0,))
    for i in range(3, 6):
        await w.enqueue("INSERT INTO t VALUES(?)", (i,))
    await asyncio.sleep(0.3)
    await w.stop()

    conn = sqlite3.connect(db)
    count = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    conn.close()
    # 6 valid rows, 1 conflict skipped
    assert count == 6
