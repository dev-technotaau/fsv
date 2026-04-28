"""Playwright-driven DuckDuckGo Images with auto-restart on browser crash.

DDG's image search returns JSON if you hit https://duckduckgo.com/i.js?q=...&vqd=...
We fetch the vqd token from the main page, then call the JSON endpoint in a loop.
"""
from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator
from urllib.parse import quote_plus

from ..logger import get_logger
from .base import Source, URLCandidate
from ._playwright_base import PlaywrightSupervisor

_log = get_logger("source.pw_ddg")


class PlaywrightDDGSource(Source):
    name = "pw_ddg"

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
                await self.throttle()
                try:
                    items = await sup.run_with_retry(self._scrape_query, q)
                except Exception as e:
                    _log.warn("query_failed_after_retries", query=q, error=str(e)[:200])
                    continue
                for item in items:
                    url = item.get("image") or item.get("thumbnail")
                    if not url or self.dedup.url_seen(url):
                        continue
                    self.dedup.mark_url(url, self.name)
                    yield URLCandidate(
                        url=url, source=self.name, query=q,
                        origin_page=item.get("url"),
                        title=item.get("title"),
                        extra={"ddg_source": item.get("source")},
                    )
        finally:
            await sup.stop()

    async def _scrape_query(self, page, q: str) -> list[dict]:
        """Return list of DDG JSON result items for this query. Caller yields candidates."""
        page_url = f"https://duckduckgo.com/?q={quote_plus(q)}&iar=images&iax=images&ia=images"
        await page.goto(page_url, timeout=self.cfg.request_timeout_s * 1000)
        await asyncio.sleep(2.0)
        content = await page.content()
        m = re.search(r"vqd=[\"']?(\d-[\d-]+)[\"']?", content)
        if not m:
            return []
        vqd = m.group(1)

        out: list[dict] = []
        seen_next = ""
        for _ in range(6):
            await self.throttle()
            api = (
                f"https://duckduckgo.com/i.js?l=us-en&o=json&q={quote_plus(q)}"
                f"&vqd={vqd}&f=,,,&p=1&v7exp=a&s={seen_next or 0}"
            )
            resp = await page.evaluate(
                """async (url) => {
                    const r = await fetch(url, {credentials: 'include'});
                    if (!r.ok) return null;
                    return await r.json();
                }""",
                api,
            )
            if not resp or "results" not in resp:
                break
            for item in resp["results"]:
                if isinstance(item, dict):
                    out.append(item)
            nxt = resp.get("next")
            if not nxt:
                break
            m2 = re.search(r"s=(\d+)", nxt)
            if not m2:
                break
            seen_next = m2.group(1)
        return out

    async def close(self) -> None:
        sup = getattr(self, "_sup", None)
        if sup is not None:
            try:
                await sup.stop()
            except Exception:
                pass
