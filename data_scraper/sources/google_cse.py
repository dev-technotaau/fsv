"""Google Custom Search API — paid.
Docs: https://developers.google.com/custom-search/v1/using_rest

Requires: api_key + cx (search engine ID).  Free tier: 100 queries/day, then $5/1000.
Each query returns up to 10 results; paginate via `start`.
"""
from __future__ import annotations

from typing import AsyncIterator

from .base import Source, URLCandidate


class GoogleCSESource(Source):
    name = "google_cse"
    ENDPOINT = "https://www.googleapis.com/customsearch/v1"

    async def discover(self) -> AsyncIterator[URLCandidate]:
        if not self.enabled or not self.cfg.api_key or not self.cfg.cx:
            return
        for q in self.queries:
            start_page = self.get_start_page(q, default=1)
            if start_page < 0:
                continue   # completed in a prior run
            last_page_reached = start_page - 1
            # Paginate: CSE `start` is 1,11,21,...
            for start in range(1 + (start_page - 1) * 10, 92, 10):
                await self.throttle()
                params = {
                    "key": self.cfg.api_key,
                    "cx": self.cfg.cx,
                    "q": q,
                    "searchType": "image",
                    "imgSize": "large",
                    "imgType": "photo",
                    "safe": "active",
                    "num": 10,
                    "start": start,
                }
                data = await self.downloader.fetch_json(self.ENDPOINT, params=params)
                if not data or "items" not in data:
                    break
                items = data["items"]
                if not items:
                    break
                for item in items:
                    url = item.get("link")
                    if not url:
                        continue
                    if self.dedup.url_seen(url):
                        continue
                    self.dedup.mark_url(url, self.name)
                    yield URLCandidate(
                        url=url,
                        source=self.name,
                        query=q,
                        title=item.get("title"),
                        origin_page=item.get("image", {}).get("contextLink"),
                        extra={
                            "displayLink": item.get("displayLink"),
                            "mime": item.get("mime"),
                            "fileFormat": item.get("fileFormat"),
                        },
                    )
                last_page_reached += 1
                self.update_progress(q, last_page_reached, completed=False)
            self.update_progress(q, last_page_reached, completed=True)
