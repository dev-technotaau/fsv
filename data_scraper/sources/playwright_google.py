"""Playwright-driven Google Images with auto-restart on browser crash.

FREE but violates Google TOS — prefer `google_cse` for production.

Strategy: Google (2025+) no longer uses `/imgres?imgurl=…` anchors for image
results — the real source URLs are now embedded as JSON strings inside
`<script>` tags on the results page. We scroll to trigger lazy-loaded results,
then regex every script tag for `http(s)://…\\.(jpg|png|webp|gif)` URLs.
No clicking, no overlay parsing; fast and reliable against DOM changes.
"""
from __future__ import annotations

import re
from typing import AsyncIterator
from urllib.parse import quote_plus, urlparse

from ..logger import get_logger
from .base import Source, URLCandidate
from ._playwright_base import PlaywrightSupervisor, scroll_and_collect

_log = get_logger("source.pw_google")


# Extract image URLs from every <script> tag. Google's embedded JSON blobs
# contain the real hotlink-able source URLs for each result.
_SCRIPT_IMG_JS = r"""
() => {
  const urls = new Set();
  const pat = /https?:\/\/[^"\s\\]+\.(?:jpe?g|png|webp|gif)(?:\?[^"\s\\]*)?/gi;
  document.querySelectorAll('script').forEach(s => {
    const t = s.textContent || '';
    if (!t) return;
    const m = t.match(pat);
    if (m) m.forEach(u => urls.add(u));
  });
  return Array.from(urls);
}
"""

_IMG_EXT_RE = re.compile(r"\.(jpe?g|png|webp|gif)(\?|$)", re.I)
# Known image-serving hosts (may not have obvious extensions in URL).
_KNOWN_IMG_HOSTS = (
    "pinimg.com", "imgur.com", "flickr.com", "staticflickr.com",
    "upload.wikimedia.org", "cloudfront.net", "amazonaws.com",
    "shopify.com", "cdninstagram.com", "ytimg.com", "redd.it",
    "wp.com", "wordpress.com",
)


def _looks_like_image_url(u: str) -> bool:
    if _IMG_EXT_RE.search(u):
        return True
    host = urlparse(u).netloc.lower()
    return any(known in host for known in _KNOWN_IMG_HOSTS)


class PlaywrightGoogleSource(Source):
    name = "pw_google"

    async def discover(self) -> AsyncIterator[URLCandidate]:
        if not self.enabled:
            return
        sup = PlaywrightSupervisor(
            browser_kind=self.cfg.browser, headless=self.cfg.headless,
        )
        try:
            await sup.start()
        except Exception as e:
            _log.warn("playwright_unavailable", error=str(e))
            return

        self._sup = sup
        try:
            for q in self.queries:
                # Resume: skip queries already completed in a prior run
                if self.get_start_page(q, default=1) < 0:
                    continue
                await self.throttle()
                try:
                    urls = await sup.run_with_retry(self._scrape_query, q)
                except Exception as e:
                    _log.warn("query_failed_after_retries", query=q, error=str(e)[:200])
                    continue
                for u, origin in urls:
                    if not u:
                        continue
                    # Drop Google-hosted thumbnails and obvious non-images.
                    skip_hosts = ("gstatic.com", "google.com/images",
                                  "googleusercontent.com", "ggpht.com")
                    if any(s in u for s in skip_hosts):
                        continue
                    # Drop wiki/File: page URLs and other non-direct-image links.
                    if not _looks_like_image_url(u):
                        continue
                    if self.dedup.url_seen(u):
                        continue
                    self.dedup.mark_url(u, self.name)
                    yield URLCandidate(url=u, source=self.name, query=q, origin_page=origin)
                self.update_progress(q, 1, completed=True)
        finally:
            await sup.stop()

    async def _scrape_query(self, page, q: str) -> list[tuple[str, str]]:
        """Scroll the results page and decode `imgurl` params from every /imgres anchor."""
        search_url = f"https://www.google.com/search?q={quote_plus(q)}&udm=2&hl=en&safe=active"
        await page.goto(search_url, timeout=self.cfg.request_timeout_s * 1000)
        # Accept cookies if prompt appears
        try:
            await page.click("button:has-text('Accept all')", timeout=2000)
        except Exception:
            pass
        await scroll_and_collect(page, max_scrolls=self.cfg.max_scrolls,
                                  scroll_delay_s=self.cfg.scroll_delay_s)
        discovered: list[str] = await page.evaluate(_SCRIPT_IMG_JS) or []
        return [(u, search_url) for u in discovered]

    async def close(self) -> None:
        sup = getattr(self, "_sup", None)
        if sup is not None:
            try:
                await sup.stop()
            except Exception:
                pass
