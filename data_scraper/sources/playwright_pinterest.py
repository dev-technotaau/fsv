"""Playwright-driven Pinterest pin scraper.

Strategy:
  1. Load https://www.pinterest.com/search/pins/?q=<query>
  2. Scroll to lazy-load more pins (up to cfg.max_scrolls)
  3. Extract every `i.pinimg.com` URL from <img src> attributes
  4. Rewrite the size path segment to `/originals/` to get the full-res image
     (Pinterest URLs are structured: /236x/AB/CD/EF/hash.jpg → /originals/AB/CD/EF/hash.jpg)

Pinterest's search page shows ~30-50 pins before scrolling, and ~100-300 after
20 scrolls. Images are direct hotlink-able (no signed URLs, no session tokens).
"""
from __future__ import annotations

from typing import AsyncIterator
from urllib.parse import quote_plus

from ..logger import get_logger
from .base import Source, URLCandidate
from ._playwright_base import PlaywrightSupervisor, scroll_and_collect

_log = get_logger("source.pw_pinterest")


# Extract pin images from i.pinimg.com and upgrade to /originals/.
# Pinterest uses TWO different URL patterns on the same page:
#   - Pin images:     /236x/…, /474x/…, /564x/…, /736x/…   ← what we want (upgrade to /originals/)
#   - Profile avatars: /60x60/…, /75x75/…                   ← reject (not real pins)
# Matching `/<digits>x/` with no trailing digits isolates pins cleanly.
_PIN_IMG_JS = r"""
() => {
  const urls = new Set();
  const pinSizeRe = /\/[0-9]+x\//;    // pin images only (/236x/, /474x/, etc)
  const avatarRe = /\/[0-9]+x[0-9]+\//; // avatars (/60x60/) — reject
  document.querySelectorAll('img').forEach(img => {
    let src = img.src || img.getAttribute('data-src') || '';
    if (!src) {
      const ss = img.getAttribute('srcset') || '';
      if (ss) src = ss.split(',')[0].trim().split(' ')[0];
    }
    if (!src || !src.includes('i.pinimg.com')) return;
    if (avatarRe.test(src)) return;    // skip profile avatars
    if (!pinSizeRe.test(src)) return;  // skip anything that isn't a sized pin
    urls.add(src.replace(pinSizeRe, '/originals/'));
  });
  return Array.from(urls);
}
"""


class PlaywrightPinterestSource(Source):
    name = "pw_pinterest"

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
        search_url = f"https://www.pinterest.com/search/pins/?q={quote_plus(q)}&rs=typed"
        await page.goto(search_url, timeout=self.cfg.request_timeout_s * 1000)
        # Dismiss the signup overlay if it appears
        for sel in ("button[aria-label='Close']",
                    "div[data-test-id='closeModal']",
                    "button:has-text('Close')"):
            try:
                await page.click(sel, timeout=1500)
                break
            except Exception:
                continue
        await scroll_and_collect(page, max_scrolls=self.cfg.max_scrolls,
                                  scroll_delay_s=self.cfg.scroll_delay_s)
        discovered: list[str] = await page.evaluate(_PIN_IMG_JS) or []
        return [(u, search_url) for u in discovered]

    async def close(self) -> None:
        sup = getattr(self, "_sup", None)
        if sup is not None:
            try:
                await sup.stop()
            except Exception:
                pass
