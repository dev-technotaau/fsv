"""Wikimedia Commons — free, no API key. All images are CC-licensed / public domain.
Docs: https://commons.wikimedia.org/w/api.php
"""
from __future__ import annotations

from typing import AsyncIterator

from .base import Source, URLCandidate


class WikimediaSource(Source):
    name = "wikimedia"
    ENDPOINT = "https://commons.wikimedia.org/w/api.php"
    # Wikimedia REQUIRES a descriptive UA with contact info (https://w.wiki/4wJS).
    # Browser-like UAs return 403 with "Please respect our robot policy".
    UA = ("ninja-fence-scraper/0.1 "
          "(https://github.com/technotau/ninja-fence; send@technotau.com) "
          "httpx/python")

    # Commons catalogues files with broad descriptive names — our segmentation-tuned
    # prompts like "cedar fence behind climbing roses" match zero files there.
    # These broad terms align with how Commons contributors actually tag images.
    _BROAD_QUERIES = [
        # originals
        "wooden fence", "cedar fence", "picket fence", "garden fence",
        "backyard fence", "wood fence", "privacy fence", "rail fence",
        "stockade fence", "split rail fence", "shadowbox fence", "redwood fence",
        "rustic fence", "weathered fence", "fence gate", "fence post",
        "farm fence", "garden with fence", "yard fence",
        # more wood types
        "pine fence", "oak fence", "bamboo fence", "teak fence",
        "pressure treated fence", "treated wood fence",
        # more structures
        "lattice fence", "slat fence", "board fence", "board on board fence",
        "horizontal fence", "vertical fence", "louvered fence",
        "post and rail fence", "post and beam fence", "pale fence",
        "palisade fence", "hurdle fence", "wattle fence", "panel fence",
        # settings / contexts
        "rural fence", "suburban fence", "park fence", "paddock fence",
        "pasture fence", "boundary fence", "property fence",
        "front yard fence", "side yard fence", "countryside fence",
        # conditions
        "old fence", "broken fence", "new fence", "painted fence",
        "decaying fence", "restored fence", "rotten fence",
        "snowy fence", "wet fence", "rainy fence",
        # settings with animals / function
        "horse fence", "cattle fence", "sheep fence", "livestock fence",
        "chicken coop fence", "deer fence", "dog fence",
        # cultural / regional variants
        "Japanese fence", "Japanese wooden fence", "Korean fence",
        "English garden fence", "Scandinavian fence", "Norwegian fence",
        "Swiss alpine fence", "Nordic fence", "country cottage fence",
        # decorative / accessory
        "wooden fence flowers", "fence with vines", "ivy covered fence",
        "fence and hedge", "garden fence wooden", "backyard wooden fence",
        "wooden gate", "wooden garden gate", "rural gate",
        # materials near fence (distractors for segmentation)
        "fence and shed", "fence and barn", "fence and pergola",
    ]

    async def discover(self) -> AsyncIterator[URLCandidate]:
        if not self.enabled:
            return
        # Substitute in broad queries — ignore the specific per-segmentation ones
        # the coordinator passes us, since Commons doesn't catalogue at that granularity.
        effective_queries = self._BROAD_QUERIES
        for q in effective_queries:
            # Resume: skip queries already completed in a prior run
            if self.get_start_page(q, default=1) < 0:
                continue
            continue_token: dict = {}
            page_limit = 5
            for _ in range(page_limit):
                await self.throttle()
                params = {
                    "action": "query",
                    "format": "json",
                    "generator": "search",
                    # Plain text search — `filetype:bitmap` matches 0 for most fence queries.
                    "gsrsearch": q,
                    "gsrnamespace": 6,          # File: namespace
                    "gsrlimit": 50,
                    "prop": "imageinfo",
                    "iiprop": "url|size|mime|extmetadata",
                    "iiurlwidth": 1600,
                }
                params.update(continue_token)
                data = await self.downloader.fetch_json(
                    self.ENDPOINT, params=params, headers={"User-Agent": self.UA},
                )
                if not data:
                    break
                pages = data.get("query", {}).get("pages", {})
                if not pages:
                    break
                for _, page in pages.items():
                    infos = page.get("imageinfo")
                    if not infos:
                        continue
                    info = infos[0]
                    url = info.get("thumburl") or info.get("url")
                    mime = info.get("mime", "")
                    if not url or "image/svg" in mime or "image/gif" in mime:
                        continue
                    if self.dedup.url_seen(url):
                        continue
                    self.dedup.mark_url(url, self.name)
                    yield URLCandidate(
                        url=url,
                        source=self.name,
                        query=q,
                        title=page.get("title"),
                        origin_page=info.get("descriptionurl"),
                        extra={
                            "width": info.get("width"),
                            "height": info.get("height"),
                            "mime": mime,
                            "license": info.get("extmetadata", {}).get("LicenseShortName", {}).get("value"),
                        },
                    )
                if "continue" not in data:
                    break
                continue_token = data["continue"]
            # Mark query completed after all pages (or early break) done.
            self.update_progress(q, 1, completed=True)
