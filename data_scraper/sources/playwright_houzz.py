"""Playwright-driven Houzz photo scraper.

Strategy:
  1. Load https://www.houzz.com/photos/query/<query> (URL-slugged, hyphenated)
  2. Scroll to load more photos
  3. Extract every `st.hzcdn.com` URL from <img> tags
  4. Upgrade the size params (`w360-h360`) to `w1600-h1600` for full-res
  5. Skip small avatar sizes (w32, w48, etc.)

Houzz URL format:
  https://st.hzcdn.com/fimgs/HASH-w<W>-h<H>-bX-pX--.jpg
Replace <W>-h<H> with 1600-h1600 to get the ~128KB original vs. 40KB thumb.
"""
from __future__ import annotations

import re
from typing import AsyncIterator

from ..logger import get_logger
from .base import Source, URLCandidate
from ._playwright_base import PlaywrightSupervisor, scroll_and_collect

_log = get_logger("source.pw_houzz")


# Pull st.hzcdn.com URLs and upgrade their size params to w1600-h1600.
# Pattern `-w<digits>-h<digits>-` is unique to Houzz's CDN image URLs.
_HOUZZ_IMG_JS = r"""
() => {
  const urls = new Set();
  const sizeRe = /-w([0-9]+)-h([0-9]+)-/;
  document.querySelectorAll('img').forEach(img => {
    let src = img.src || img.getAttribute('data-src') || '';
    if (!src || !src.includes('hzcdn.com')) return;
    const m = src.match(sizeRe);
    if (!m) return;                                // not a sized CDN image
    const w = parseInt(m[1], 10);
    if (w < 200) return;                           // skip avatars (w32, w48, etc)
    const full = src.replace(sizeRe, '-w1600-h1600-');
    urls.add(full);
  });
  return Array.from(urls);
}
"""


def _slug(q: str) -> str:
    """URL-slug a query for Houzz's /photos/query/<slug> format."""
    s = re.sub(r"[^a-z0-9]+", "-", q.lower()).strip("-")
    return s or "fence"


class PlaywrightHouzzSource(Source):
    name = "pw_houzz"

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
                if self.get_start_page(q, default=1) < 0:
                    continue
                await self.throttle()
                try:
                    urls = await sup.run_with_retry(self._scrape_query, q)
                except Exception as e:
                    _log.warn("query_failed_after_retries", query=q, error=str(e)[:200])
                    continue
                for u, origin in urls:
                    if not u or not u.startswith("http"):
                        continue
                    if self.dedup.url_seen(u):
                        continue
                    self.dedup.mark_url(u, self.name)
                    yield URLCandidate(url=u, source=self.name, query=q, origin_page=origin)
                self.update_progress(q, 1, completed=True)
        finally:
            await sup.stop()

    async def _scrape_query(self, page, q: str) -> list[tuple[str, str]]:
        slug = _slug(q)
        search_url = f"https://www.houzz.com/photos/query/{slug}"
        await page.goto(search_url, timeout=self.cfg.request_timeout_s * 1000)
        # Dismiss any signup/cookie modal
        for sel in ("button[aria-label='Close']",
                    "button:has-text('Accept All')",
                    "button:has-text('Got it')"):
            try:
                await page.click(sel, timeout=1500)
                break
            except Exception:
                continue
        await scroll_and_collect(page, max_scrolls=self.cfg.max_scrolls,
                                  scroll_delay_s=self.cfg.scroll_delay_s)
        discovered: list[str] = await page.evaluate(_HOUZZ_IMG_JS) or []
        return [(u, search_url) for u in discovered]

    async def close(self) -> None:
        sup = getattr(self, "_sup", None)
        if sup is not None:
            try:
                await sup.stop()
            except Exception:
                pass
