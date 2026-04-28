"""Tests for DedupStore — SHA, BK-tree integration, query progress, failures, repair."""
from __future__ import annotations

from pathlib import Path

import pytest

from data_scraper.dedup import DedupStore, sha256_of_bytes, dhash_of_bytes


def test_sha_exact_dedup(tmp_db_path, jpeg_bytes):
    store = DedupStore(tmp_db_path, phash_hamming_threshold=5)
    b = jpeg_bytes()
    sha = sha256_of_bytes(b)
    assert not store.exists_sha256(sha)
    ok = store.save_image(sha256=sha, dhash=dhash_of_bytes(b),
                          path="/tmp/x.jpg", source="test",
                          query="q", origin_url="http://x",
                          width=1024, height=768, bytes_=len(b))
    assert ok
    assert store.exists_sha256(sha)
    # Attempt re-insert with same sha → returns False
    ok2 = store.save_image(sha256=sha, dhash=None, path="/tmp/y.jpg",
                           source="test", query="q", origin_url="http://x",
                           width=1024, height=768, bytes_=len(b))
    assert ok2 is False


def test_url_seen(tmp_db_path):
    store = DedupStore(tmp_db_path)
    assert not store.url_seen("http://example.com/a")
    store.mark_url("http://example.com/a", "test")
    assert store.url_seen("http://example.com/a")
    # Second mark is idempotent
    store.mark_url("http://example.com/a", "test")
    assert store.url_seen_count() == 1


def test_near_duplicate_via_bktree(tmp_db_path, jpeg_bytes):
    store = DedupStore(tmp_db_path, phash_hamming_threshold=5)
    # Two visually similar images (same color, tiny size diff)
    b1 = jpeg_bytes(w=1024, h=768, color=(100, 150, 80))
    sha1 = sha256_of_bytes(b1)
    dh1 = dhash_of_bytes(b1)
    store.save_image(sha256=sha1, dhash=dh1, path="/tmp/1.jpg",
                     source="t", query="q", origin_url="u1",
                     width=1024, height=768, bytes_=len(b1))
    # Slightly different size → different sha, similar dhash
    b2 = jpeg_bytes(w=1025, h=768, color=(100, 150, 80))
    dh2 = dhash_of_bytes(b2)
    assert dh1 is not None and dh2 is not None
    # dHash should likely be near-identical; assert bktree detects
    assert store.near_duplicate(dh2)


def test_query_progress_roundtrip(tmp_db_path):
    store = DedupStore(tmp_db_path)
    assert store.query_progress("cse", "cedar fence") == (0, False)
    store.update_query_progress("cse", "cedar fence", last_page=3, completed=False)
    assert store.query_progress("cse", "cedar fence") == (3, False)
    store.update_query_progress("cse", "cedar fence", last_page=9, completed=True)
    assert store.is_query_completed("cse", "cedar fence")


def test_failures_roundtrip(tmp_db_path):
    store = DedupStore(tmp_db_path)
    store.log_failure("http://x/1", "src", "timeout")
    store.log_failure("http://x/2", "src", "bad_image")
    assert store.failures_count() == 2
    assert store.failures_count(unretried_only=True) == 2
    failures = store.get_unretried_failures(limit=10)
    assert len(failures) == 2
    for fid, *_ in failures:
        store.mark_failure_retried(fid)
    assert store.failures_count(unretried_only=True) == 0


def test_repair_removes_missing_files(tmp_db_path, tmp_images_dir, jpeg_bytes):
    store = DedupStore(tmp_db_path)
    # Insert 3 rows: 2 files exist, 1 doesn't
    for i in range(3):
        b = jpeg_bytes(color=(i * 50, 0, 0))
        path = tmp_images_dir / f"img_{i}.jpg"
        if i != 2:
            path.write_bytes(b)
        store.save_image(
            sha256=sha256_of_bytes(b),
            dhash=dhash_of_bytes(b),
            path=str(path), source="t", query="q", origin_url=f"u{i}",
            width=1024, height=768, bytes_=len(b),
        )
    # dry-run
    res = store.repair(tmp_images_dir, dry_run=True)
    assert res == {"kept": 2, "removed": 1, "dry_run": True}
    assert store.count() == 3  # no actual mutation
    # real run
    res = store.repair(tmp_images_dir, dry_run=False)
    assert res == {"kept": 2, "removed": 1, "dry_run": False}
    assert store.count() == 2
