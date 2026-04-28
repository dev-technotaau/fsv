"""Pexels API — free. Docs: https://www.pexels.com/api/documentation/
Rate: 200 requests/hour, 20000/month.
"""
from __future__ import annotations

from typing import AsyncIterator

from .base import Source, URLCandidate


class PexelsSource(Source):
    name = "pexels"
    ENDPOINT = "https://api.pexels.com/v1/search"

    async def discover(self) -> AsyncIterator[URLCandidate]:
        if not self.enabled or not self.cfg.api_key:
            return
        headers = {"Authorization": self.cfg.api_key}
        for q in self.queries:
            start_page = self.get_start_page(q, default=1)
            if start_page < 0:
                continue
            page = start_page
            while page <= 8:        # 8 pages × 80 per-page = 640 results per query max
                await self.throttle()
                params = {"query": q, "per_page": 80, "page": page, "orientation": "landscape"}
                data = await self.downloader.fetch_json(self.ENDPOINT, params=params, headers=headers)
                if not data or "photos" not in data:
                    break
                photos = data["photos"]
                if not photos:
                    break
                for p in photos:
                    src = p.get("src", {})
                    # Prefer large2x (1880x1253, ~1-3MB) over original (can be 20-40MB).
                    # For seg training at 640px input, large2x is plenty and 10× faster to fetch.
                    url = src.get("large2x") or src.get("large") or src.get("original")
                    if not url:
                        continue
                    if self.dedup.url_seen(url):
                        continue
                    self.dedup.mark_url(url, self.name)
                    yield URLCandidate(
                        url=url,
                        source=self.name,
                        query=q,
                        title=p.get("alt"),
                        origin_page=p.get("url"),
                        extra={
                            "photographer": p.get("photographer"),
                            "width": p.get("width"),
                            "height": p.get("height"),
                        },
                    )
                self.update_progress(q, page, completed=False)
                if not data.get("next_page"):
                    break
                page += 1
            self.update_progress(q, page, completed=True)
