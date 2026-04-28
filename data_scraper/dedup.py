"""SQLite-backed dedup + query-progress state.

Features:
  - Exact dedup (SHA256) and near-dedup (dHash + BK-tree for O(log N) lookup)
  - urls_seen table (resume: skip URLs we've tried)
  - query_progress table (resume: skip pages we've completed per source×query)
  - failures table
  - repair(): scan images_dir, remove DB rows whose files are missing
"""
from __future__ import annotations

import hashlib
import io
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from .bk_tree import BKTree, hamming


# ============================================================
# Hashes
# ============================================================

def sha256_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _u64_to_i64(n: Optional[int]) -> Optional[int]:
    """SQLite INTEGER is signed 64-bit; the 64-bit unsigned dhash can exceed 2^63-1.
    Map unsigned→signed via two's-complement so the bit pattern round-trips losslessly."""
    if n is None:
        return None
    return n - (1 << 64) if n >= (1 << 63) else n


def _i64_to_u64(n: Optional[int]) -> Optional[int]:
    """Inverse of _u64_to_i64: signed→unsigned so in-memory Hamming math stays positive."""
    if n is None:
        return None
    return n + (1 << 64) if n < 0 else n


def dhash_of_bytes(b: bytes, hash_size: int = 8) -> Optional[int]:
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(b)).convert("L").resize(
            (hash_size + 1, hash_size), Image.BILINEAR)
        pixels = list(img.getdata())
        bits = 0
        idx = 0
        for r in range(hash_size):
            row = pixels[r * (hash_size + 1):(r + 1) * (hash_size + 1)]
            for c in range(hash_size):
                if row[c + 1] > row[c]:
                    bits |= (1 << idx)
                idx += 1
        return bits
    except Exception:
        return None


# ============================================================
# Schema
# ============================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256      TEXT    NOT NULL UNIQUE,
    dhash       INTEGER,
    path        TEXT    NOT NULL,
    source      TEXT    NOT NULL,
    query       TEXT,
    origin_url  TEXT,
    width       INTEGER,
    height      INTEGER,
    bytes       INTEGER,
    ts          INTEGER
);
CREATE INDEX IF NOT EXISTS idx_dhash  ON images(dhash);
CREATE INDEX IF NOT EXISTS idx_source ON images(source);

CREATE TABLE IF NOT EXISTS urls_seen (
    url         TEXT PRIMARY KEY,
    source      TEXT,
    ts          INTEGER
);

CREATE TABLE IF NOT EXISTS failures (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT,
    source      TEXT,
    reason      TEXT,
    retried     INTEGER DEFAULT 0,
    ts          INTEGER
);

CREATE TABLE IF NOT EXISTS query_progress (
    source      TEXT NOT NULL,
    query       TEXT NOT NULL,
    last_page   INTEGER NOT NULL DEFAULT 0,
    completed   INTEGER NOT NULL DEFAULT 0,
    ts          INTEGER,
    PRIMARY KEY (source, query)
);
"""


# ============================================================
# Store
# ============================================================

class DedupStore:
    """SQLite-backed dedup + progress state. Safe across threads."""

    def __init__(self, db_path: Path, phash_hamming_threshold: int = 5):
        self.db_path = db_path
        self.threshold = phash_hamming_threshold
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._bk = BKTree()
        self._bk_loaded = False
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_bk_loaded(self) -> None:
        if self._bk_loaded:
            return
        with self._lock:
            if self._bk_loaded:
                return
            with self._connect() as conn:
                rows = conn.execute("SELECT dhash FROM images WHERE dhash IS NOT NULL").fetchall()
            self._bk.bulk_load([_i64_to_u64(r[0]) for r in rows if r[0] is not None])
            self._bk_loaded = True

    # ---------- URL dedup ----------
    def url_seen(self, url: str) -> bool:
        with self._connect() as conn:
            return conn.execute("SELECT 1 FROM urls_seen WHERE url=? LIMIT 1", (url,)).fetchone() is not None

    def mark_url(self, url: str, source: str) -> None:
        with self._lock, self._connect() as conn:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO urls_seen(url, source, ts) VALUES(?, ?, ?)",
                    (url, source, int(time.time())),
                )
                conn.commit()
            except sqlite3.Error:
                pass

    # ---------- content dedup ----------
    def exists_sha256(self, sha: str) -> bool:
        with self._connect() as conn:
            return conn.execute("SELECT 1 FROM images WHERE sha256=? LIMIT 1", (sha,)).fetchone() is not None

    def near_duplicate(self, dhash: int) -> bool:
        self._ensure_bk_loaded()
        return self._bk.find_within(dhash, self.threshold)

    def save_image(self, *, sha256: str, dhash: Optional[int], path: str, source: str,
                   query: Optional[str], origin_url: Optional[str],
                   width: int, height: int, bytes_: int) -> bool:
        with self._lock, self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO images(sha256, dhash, path, source, query, origin_url, "
                    "width, height, bytes, ts) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (sha256, _u64_to_i64(dhash), path, source, query, origin_url,
                     width, height, bytes_, int(time.time())),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                return False
            if dhash is not None:
                self._bk.add(dhash)
            return True

    def delete_by_path(self, path: str) -> bool:
        """Remove a row by its saved file path. Used by vision-qa post-hoc cleanup.
        dhash stays in BK-tree (harmless — worst case prevents re-adding an identical image)."""
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM images WHERE path=?", (path,))
            conn.commit()
            return cur.rowcount > 0

    def log_failure(self, url: str, source: str, reason: str) -> None:
        with self._lock, self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO failures(url, source, reason, retried, ts) VALUES(?,?,?,0,?)",
                    (url, source, reason[:500], int(time.time())),
                )
                conn.commit()
            except sqlite3.Error:
                pass

    # ---------- query progress (resumable) ----------
    def query_progress(self, source: str, query: str) -> tuple[int, bool]:
        """Return (last_page, completed). last_page=0 if never attempted."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT last_page, completed FROM query_progress WHERE source=? AND query=?",
                (source, query),
            ).fetchone()
            if row is None:
                return (0, False)
            return (int(row[0]), bool(row[1]))

    def update_query_progress(self, source: str, query: str, last_page: int,
                              completed: bool = False) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO query_progress(source, query, last_page, completed, ts)
                   VALUES(?,?,?,?,?)
                   ON CONFLICT(source, query) DO UPDATE SET
                     last_page=excluded.last_page,
                     completed=excluded.completed,
                     ts=excluded.ts""",
                (source, query, last_page, 1 if completed else 0, int(time.time())),
            )
            conn.commit()

    def is_query_completed(self, source: str, query: str) -> bool:
        _, completed = self.query_progress(source, query)
        return completed

    # ---------- failure retry ----------
    def get_unretried_failures(self, limit: int = 1000) -> list[tuple[int, str, str, str]]:
        """Return list of (id, url, source, reason) for retry pass."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, url, source, reason FROM failures WHERE retried=0 AND url IS NOT NULL "
                "ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [tuple(r) for r in rows]

    def mark_failure_retried(self, failure_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE failures SET retried=1 WHERE id=?", (failure_id,))
            conn.commit()

    # ---------- repair / GC ----------
    def repair(self, images_dir: Path, dry_run: bool = False) -> dict:
        """Scan images_dir, remove DB rows whose files are missing.
        Also rebuilds BK-tree from scratch. Returns stats dict."""
        removed = 0
        kept = 0
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT id, path FROM images").fetchall()
            for row_id, path in rows:
                p = Path(path)
                if p.exists():
                    kept += 1
                else:
                    removed += 1
                    if not dry_run:
                        conn.execute("DELETE FROM images WHERE id=?", (row_id,))
            if not dry_run:
                conn.commit()
        if not dry_run:
            # rebuild BK-tree from surviving rows
            self._bk = BKTree()
            self._bk_loaded = False
            self._ensure_bk_loaded()
        return {"kept": kept, "removed": removed, "dry_run": dry_run}

    # ---------- stats ----------
    def count(self, source: Optional[str] = None) -> int:
        with self._connect() as conn:
            if source:
                return conn.execute("SELECT COUNT(*) FROM images WHERE source=?", (source,)).fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    def counts_by_source(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT source, COUNT(*) FROM images GROUP BY source").fetchall()
        return dict(rows)

    def url_seen_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM urls_seen").fetchone()[0]

    def failures_count(self, unretried_only: bool = False) -> int:
        with self._connect() as conn:
            if unretried_only:
                return conn.execute("SELECT COUNT(*) FROM failures WHERE retried=0").fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
