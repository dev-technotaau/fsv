"""Unsplash API — free. Docs: https://unsplash.com/documentation#search-photos
Rate: 50 req/hour (demo), 5000/hour (production — needs approval).
"""
from __future__ import annotations

from typing import AsyncIterator

from .base import Source, URLCandidate


class UnsplashSource(Source):
    name = "unsplash"
    ENDPOINT = "https://api.unsplash.com/search/photos"

    async def discover(self) -> AsyncIterator[URLCandidate]:
        if not self.enabled or not self.cfg.access_key:
            return
        headers = {
            "Authorization": f"Client-ID {self.cfg.access_key}",
            "Accept-Version": "v1",
        }
        for q in self.queries:
            start_page = self.get_start_page(q, default=1)
            if start_page < 0:
                continue
            page = start_page
            while page <= 6:   # 6 × 30 = 180 per query
                await self.throttle()
                params = {"query": q, "per_page": 30, "page": page,
                          "orientation": "landscape", "content_filter": "high"}
                data = await self.downloader.fetch_json(self.ENDPOINT, params=params, headers=headers)
                if not data or "results" not in data:
                    break
                results = data["results"]
                if not results:
                    break
                for r in results:
                    urls = r.get("urls", {})
                    # Prefer `regular` (1080px, ~300KB-1MB) over `full`/`raw` (can be 5-30MB).
                    url = urls.get("regular") or urls.get("full") or urls.get("raw")
                    if not url:
                        continue
                    if self.dedup.url_seen(url):
                        continue
                    self.dedup.mark_url(url, self.name)
                    yield URLCandidate(
                        url=url,
                        source=self.name,
                        query=q,
                        title=r.get("alt_description") or r.get("description"),
                        origin_page=r.get("links", {}).get("html"),
                        extra={
                            "width": r.get("width"),
                            "height": r.get("height"),
                            "likes": r.get("likes"),
                        },
                    )
                self.update_progress(q, page, completed=False)
                total_pages = data.get("total_pages", 1)
                if page >= total_pages:
                    break
                page += 1
            self.update_progress(q, page, completed=True)
