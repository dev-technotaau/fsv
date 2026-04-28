"""Scrape a configured list of fence-company gallery URLs for images.

Most fence / staining / maintenance company sites are static WordPress — their
gallery pages have direct `<img src="https://.../wp-content/uploads/.../photo.jpg">`
references with no lazy-loading tricks. Plain HTML + regex is enough, no
Playwright required.

Config format (in scraper.yaml):
    company_sites:
      enabled: true
      urls:
        - "https://example-fence.com/gallery"
        - "https://cedar-stain-pro.com/portfolio"
      ...
"""
from __future__ import annotations

import re
from typing import AsyncIterator
from urllib.parse import urljoin, urlparse

from ..logger import get_logger
from .base import Source, URLCandidate

_log = get_logger("source.company_sites")


# Match <img src="..." or data-src="..." pointing at an image
_IMG_TAG_RE = re.compile(
    r'<img\b[^>]*?\s(?:data-src|src|data-lazy-src|data-original)\s*=\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)
# Also catch srcset="url1 1x, url2 2x" — take the largest (last) entry
_SRCSET_RE = re.compile(
    r'srcset\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE,
)
_IMG_EXT_RE = re.compile(r"\.(jpe?g|png|webp)(\?|$)", re.IGNORECASE)
# Strip WordPress's -WxH size suffix ("/photo-300x200.jpg" → "/photo.jpg") for full-res
_WP_SIZE_RE = re.compile(r"-(\d+)x(\d+)(\.(?:jpe?g|png|webp))$", re.IGNORECASE)


def _extract_image_urls(html: str, base_url: str) -> list[str]:
    """Pull every image URL out of the HTML, resolve relative paths, upgrade
    WordPress size-suffixed URLs to full-res."""
    out: set[str] = set()

    for m in _IMG_TAG_RE.finditer(html):
        url = m.group(1).strip()
        if url.startswith("data:"):
            continue
        abs_url = urljoin(base_url, url)
        if _IMG_EXT_RE.search(abs_url):
            out.add(_WP_SIZE_RE.sub(r"\3", abs_url))

    for m in _SRCSET_RE.finditer(html):
        srcset = m.group(1)
        # Take the URL with the largest `Nw` or `Nx` descriptor
        best = None
        best_w = 0
        for part in srcset.split(","):
            bits = part.strip().split()
            if not bits:
                continue
            u = bits[0]
            w = 0
            if len(bits) > 1:
                d = bits[1].rstrip("wx").strip()
                try:
                    w = int(float(d))
                except ValueError:
                    w = 0
            if u.startswith("data:"):
                continue
            abs_u = urljoin(base_url, u)
            if _IMG_EXT_RE.search(abs_u) and w >= best_w:
                best = abs_u
                best_w = w
        if best:
            out.add(_WP_SIZE_RE.sub(r"\3", best))

    return sorted(out)


class CompanySitesSource(Source):
    name = "company_sites"

    async def discover(self) -> AsyncIterator[URLCandidate]:
        if not self.enabled:
            return
        urls: list[str] = list(getattr(self.cfg, "urls", []) or [])
        if not urls:
            _log.info("no_company_urls_configured")
            return

        for page_url in urls:
            # Resume: use the URL itself as the progress key
            if self.get_start_page(page_url, default=1) < 0:
                continue
            await self.throttle()
            try:
                html = await self.downloader.fetch_text(page_url)
            except Exception as e:
                _log.warn("fetch_failed", url=page_url, error=str(e)[:120])
                continue
            if not html:
                _log.debug("empty_html", url=page_url)
                self.update_progress(page_url, 1, completed=True)
                continue

            found = _extract_image_urls(html, page_url)
            host = urlparse(page_url).netloc
            emitted = 0
            for img_url in found:
                if self.dedup.url_seen(img_url):
                    continue
                self.dedup.mark_url(img_url, self.name)
                yield URLCandidate(
                    url=img_url, source=self.name, query=host,
                    origin_page=page_url,
                )
                emitted += 1
            _log.info("page_scraped", url=page_url, imgs_found=len(found),
                     yielded=emitted)
            self.update_progress(page_url, 1, completed=True)
