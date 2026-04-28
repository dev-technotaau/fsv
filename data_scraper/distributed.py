"""Optional distributed coordination via Redis.

When `distributed.enabled` is True, workers share:
  - URL-seen set (SADD/SISMEMBER on redis set)
  - Query-progress dict (HSET/HGET)
  - SHA256-seen set (SADD/SISMEMBER)

This turns the scraper into a multi-machine workload: start N processes on
different machines, each pointing at the same Redis URL; they'll divide work
automatically via distinct URL discovery and see each other's saves.

IMPORTANT: image file storage is still local per machine. To share files
across nodes, put `storage.images_dir` on a shared filesystem (NFS, s3fs, etc.).

Fallback: if redis-py isn't installed or the connection fails, methods return
"I don't know" values that match single-machine behavior.
"""
from __future__ import annotations

import asyncio
from typing import Optional


class NullDistributedStore:
    """Used when distributed is disabled. All methods are no-ops."""
    enabled = False
    async def url_seen(self, url: str) -> bool: return False
    async def mark_url(self, url: str, source: str) -> None: pass
    async def sha_seen(self, sha: str) -> bool: return False
    async def mark_sha(self, sha: str) -> None: pass
    async def get_query_progress(self, source: str, query: str) -> int: return 0
    async def set_query_progress(self, source: str, query: str, page: int) -> None: pass
    async def close(self) -> None: pass


class RedisDistributedStore:
    """Redis-backed coordination. Requires `redis>=5.0` (async)."""
    enabled = True

    def __init__(self, url: str, key_prefix: str = "fence_scraper:"):
        try:
            import redis.asyncio as redis    # type: ignore
        except ImportError as e:
            raise ImportError("redis>=5.0 required for distributed mode: pip install redis") from e
        self._client = redis.from_url(url, decode_responses=True)
        self._prefix = key_prefix

    async def url_seen(self, url: str) -> bool:
        return bool(await self._client.sismember(self._prefix + "urls", url))

    async def mark_url(self, url: str, source: str) -> None:
        pipe = self._client.pipeline()
        pipe.sadd(self._prefix + "urls", url)
        pipe.hset(self._prefix + "url_source", url, source)
        await pipe.execute()

    async def sha_seen(self, sha: str) -> bool:
        return bool(await self._client.sismember(self._prefix + "shas", sha))

    async def mark_sha(self, sha: str) -> None:
        await self._client.sadd(self._prefix + "shas", sha)

    async def get_query_progress(self, source: str, query: str) -> int:
        key = self._prefix + "qprog"
        field = f"{source}::{query}"
        val = await self._client.hget(key, field)
        try:
            return int(val or 0)
        except (TypeError, ValueError):
            return 0

    async def set_query_progress(self, source: str, query: str, page: int) -> None:
        key = self._prefix + "qprog"
        field = f"{source}::{query}"
        await self._client.hset(key, field, str(page))

    async def close(self) -> None:
        try:
            await self._client.aclose()
        except Exception:
            pass


def build_distributed_store(enabled: bool, url: Optional[str],
                             key_prefix: str = "fence_scraper:"):
    """Factory. Falls back to NullDistributedStore on any init failure."""
    if not enabled or not url:
        return NullDistributedStore()
    try:
        return RedisDistributedStore(url, key_prefix=key_prefix)
    except Exception as e:
        # Logger imported lazily to avoid circular deps
        from .logger import get_logger
        get_logger("distributed").warn("redis_init_failed_falling_back", error=str(e))
        return NullDistributedStore()
