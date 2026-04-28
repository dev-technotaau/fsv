"""Pixabay API — free. Docs: https://pixabay.com/api/docs/
Rate: 100 requests/60s, 1M results/month.
"""
from __future__ import annotations

from typing import AsyncIterator

from .base import Source, URLCandidate


class PixabaySource(Source):
    name = "pixabay"
    ENDPOINT = "https://pixabay.com/api/"

    async def discover(self) -> AsyncIterator[URLCandidate]:
        if not self.enabled or not self.cfg.api_key:
            return
        for q in self.queries:
            start_page = self.get_start_page(q, default=1)
            if start_page < 0:
                continue
            page = start_page
            while page <= 5:   # up to 5 pages × 200 per_page = 1000 per query
                await self.throttle()
                params = {
                    "key": self.cfg.api_key,
                    "q": q,
                    "image_type": "photo",
                    "orientation": "horizontal",
                    "safesearch": "true",
                    "per_page": 200,
                    "page": page,
                    "min_width": 1024,
                    "min_height": 720,
                }
                data = await self.downloader.fetch_json(self.ENDPOINT, params=params)
                if not data or "hits" not in data:
                    break
                hits = data["hits"]
                if not hits:
                    break
                for h in hits:
                    # largeImageURL is highest res available via free API
                    url = h.get("largeImageURL") or h.get("webformatURL")
                    if not url:
                        continue
                    if self.dedup.url_seen(url):
                        continue
                    self.dedup.mark_url(url, self.name)
                    yield URLCandidate(
                        url=url,
                        source=self.name,
                        query=q,
                        title=h.get("tags"),
                        origin_page=h.get("pageURL"),
                        extra={
                            "user": h.get("user"),
                            "imageWidth": h.get("imageWidth"),
                            "imageHeight": h.get("imageHeight"),
                            "views": h.get("views"),
                        },
                    )
                self.update_progress(q, page, completed=False)
                # Pixabay caps total results at 500 without pagination hint
                if len(hits) < 200:
                    break
                page += 1
            self.update_progress(q, page, completed=True)
