"""Playwright-driven Bing Images with auto-restart on browser crash."""
from __future__ import annotations

from typing import AsyncIterator
from urllib.parse import quote_plus

from ..logger import get_logger
from .base import Source, URLCandidate
from ._playwright_base import PlaywrightSupervisor, scroll_and_collect

_log = get_logger("source.pw_bing")


BING_JS = """
(() => {
  const urls = new Set();
  // Bing embeds the original image URL in each result anchor's `m` attribute as JSON
  document.querySelectorAll('a.iusc').forEach(a => {
    const m = a.getAttribute('m');
    if (!m) return;
    try {
      const meta = JSON.parse(m);
      if (meta.murl) urls.add(meta.murl);
      if (meta.turl && !meta.murl) urls.add(meta.turl);
    } catch (e) {}
  });
  document.querySelectorAll('img.mimg').forEach(img => {
    if (img.src && img.src.startsWith('http')) urls.add(img.src);
  });
  return Array.from(urls);
})()
"""


class PlaywrightBingSource(Source):
    name = "pw_bing"

    # Query diversifiers: each (suffix, filter_param) pair forces Bing to return
    # a different result set for the same base query. 4 variants × 319 queries = 1276
    # distinct "queries" as far as Bing's ranker is concerned. filterui codes:
    #   imagesize-large     → min ~500px
    #   imagesize-wallpaper → min ~1024px, landscape bias
    #   photo-photo         → real photos only (not clipart/line-art)
    # See https://github.com/gurugio/bing_image_downloader for the filterui reference.
    _VARIANTS = [
        ("", ""),                                      # baseline
        (" HD", "+filterui:imagesize-large"),          # HD + large
        (" original", "+filterui:imagesize-wallpaper"), # wallpaper tier
        (" photo", "+filterui:photo-photo"),           # real photos only
    ]

    def _expand(self, q: str) -> list[tuple[str, str, str]]:
        """Yield (variant_key, text_query, filter_param) per variant."""
        out = []
        for suffix, fparam in self._VARIANTS:
            variant_q = (q + suffix).strip()
            key = f"{q}||{suffix}||{fparam}"   # unique progress key per variant
            out.append((key, variant_q, fparam))
        return out

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
                for progress_key, text_q, fparam in self._expand(q):
                    # Resume: skip variants already completed in a prior run
                    if self.get_start_page(progress_key, default=1) < 0:
                        continue
                    await self.throttle()
                    try:
                        urls = await sup.run_with_retry(self._scrape_query, text_q, fparam)
                    except Exception as e:
                        _log.warn("query_failed_after_retries",
                                  query=text_q, error=str(e)[:200])
                        continue
                    for u, origin in urls:
                        if not u or self.dedup.url_seen(u):
                            continue
                        self.dedup.mark_url(u, self.name)
                        yield URLCandidate(url=u, source=self.name,
                                            query=text_q, origin_page=origin)
                    self.update_progress(progress_key, 1, completed=True)
        finally:
            await sup.stop()

    async def _scrape_query(self, page, q: str, fparam: str = "") -> list[tuple[str, str]]:
        # qft={param} enables Bing's "filter toggles" (size, type, etc).
        qft = f"&qft={quote_plus(fparam)}" if fparam else ""
        url = (f"https://www.bing.com/images/search?q={quote_plus(q)}"
               f"&form=QBLH&safe=strict{qft}")
        await page.goto(url, timeout=self.cfg.request_timeout_s * 1000)
        await scroll_and_collect(page, max_scrolls=self.cfg.max_scrolls,
                                  scroll_delay_s=self.cfg.scroll_delay_s)
        discovered: list[str] = await page.evaluate(BING_JS) or []
        return [(u, url) for u in discovered]

    async def close(self) -> None:
        sup = getattr(self, "_sup", None)
        if sup is not None:
            try:
                await sup.stop()
            except Exception:
                pass
