"""Flickr API — free. Docs: https://www.flickr.com/services/api/
Rate: informal limit ~3600/hour.

We use flickr.photos.search with Creative Commons licenses so images are
safe for training / redistribution (licenses 1,2,3,4,5,6,7,9,10).
"""
from __future__ import annotations

from typing import AsyncIterator

from .base import Source, URLCandidate


class FlickrSource(Source):
    name = "flickr"
    ENDPOINT = "https://api.flickr.com/services/rest/"
    CC_LICENSES = "1,2,3,4,5,6,7,9,10"   # all Creative Commons variants
    # Prefer large/huge (~1-3MB) over original (can be 20-50MB).
    # For seg training, 1024-2048px is plenty. "o" (original) is fallback only.
    SIZE_PREF = ("l", "h", "b", "c", "o")  # large > huge > big > medium > original

    async def discover(self) -> AsyncIterator[URLCandidate]:
        if not self.enabled or not self.cfg.api_key:
            return
        for q in self.queries:
            start_page = self.get_start_page(q, default=1)
            if start_page < 0:
                continue
            page = start_page
            while page <= 6:  # 6 × 250 = 1500 per query
                await self.throttle()
                params = {
                    "method": "flickr.photos.search",
                    "api_key": self.cfg.api_key,
                    "text": q,
                    "license": self.CC_LICENSES,
                    "content_type": 1,      # photos only
                    "media": "photos",
                    "per_page": 250,
                    "page": page,
                    "extras": "url_o,url_l,url_h,url_b,url_c,owner_name,license",
                    "format": "json",
                    "nojsoncallback": 1,
                    "sort": "relevance",
                }
                data = await self.downloader.fetch_json(self.ENDPOINT, params=params)
                if not data or data.get("stat") != "ok":
                    break
                photos = data.get("photos", {})
                photo_list = photos.get("photo", [])
                if not photo_list:
                    break
                for p in photo_list:
                    url = None
                    for size in self.SIZE_PREF:
                        u = p.get(f"url_{size}")
                        if u:
                            url = u
                            break
                    if not url:
                        continue
                    if self.dedup.url_seen(url):
                        continue
                    self.dedup.mark_url(url, self.name)
                    yield URLCandidate(
                        url=url,
                        source=self.name,
                        query=q,
                        title=p.get("title"),
                        origin_page=f"https://www.flickr.com/photos/{p.get('owner')}/{p.get('id')}",
                        extra={
                            "owner": p.get("ownername"),
                            "license": p.get("license"),
                        },
                    )
                self.update_progress(q, page, completed=False)
                total_pages = photos.get("pages", 1)
                if page >= total_pages:
                    break
                page += 1
            self.update_progress(q, page, completed=True)
