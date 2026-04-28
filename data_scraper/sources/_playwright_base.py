"""Shared helpers for Playwright sources + a supervisor with auto-restart.

PlaywrightSupervisor wraps browser lifecycle so that if the browser crashes
mid-scrape, we restart it and continue. Exponential backoff on repeated crashes.
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Optional

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
except ImportError:
    async_playwright = None
    Browser = BrowserContext = Page = None  # type: ignore

from ..logger import get_logger

log = get_logger("playwright")

STEALTH_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


class PlaywrightSupervisor:
    """Manages browser lifecycle with auto-restart on crash.

    Use `run_with_retry(task_fn)` for operations that should survive browser death.
    """

    def __init__(self, *, browser_kind: str = "chromium", headless: bool = True,
                 viewport: tuple[int, int] = (1600, 1000),
                 proxy: Optional[str] = None,
                 max_restarts: int = 5, restart_backoff_s: float = 5.0):
        self.browser_kind = browser_kind
        self.headless = headless
        self.viewport = viewport
        self.proxy = proxy
        self.max_restarts = max_restarts
        self.restart_backoff_s = restart_backoff_s
        self._pw = None
        self._browser = None
        self._ctx = None
        self._restart_count = 0
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if async_playwright is None:
            raise ImportError("playwright required: pip install playwright && playwright install chromium")
        async with self._lock:
            if self._ctx is not None:
                return
            self._pw = await async_playwright().start()
            launcher = getattr(self._pw, self.browser_kind)
            launch_kwargs: dict = {
                "headless": self.headless,
                "args": [
                    "--disable-blink-features=AutomationControlled",
                    "--disable-features=IsolateOrigins,site-per-process",
                    "--no-sandbox",
                ],
            }
            if self.proxy:
                launch_kwargs["proxy"] = {"server": self.proxy}
            self._browser = await launcher.launch(**launch_kwargs)
            self._ctx = await self._browser.new_context(
                user_agent=STEALTH_UA,
                viewport={"width": self.viewport[0], "height": self.viewport[1]},
                locale="en-US",
            )
            # Comprehensive stealth init — Google Images serves a blocker page to
            # detected Chromium automation. We patch the properties Google checks.
            await self._ctx.add_init_script("""
                // 1) navigator.webdriver → undefined (the obvious one)
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                // 2) Spoof plugins (headless Chromium has none — real Chrome has 3+)
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [
                        {name: 'PDF Viewer', filename: 'internal-pdf-viewer'},
                        {name: 'Chrome PDF Viewer', filename: 'internal-pdf-viewer'},
                        {name: 'Chromium PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai'},
                    ],
                });
                // 3) navigator.languages (headless defaults to empty array)
                Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
                // 4) window.chrome — headless Chromium doesn't inject this
                if (!window.chrome) {
                    window.chrome = {runtime: {}, app: {}, csi: () => {}, loadTimes: () => {}};
                }
                // 5) permissions.query spoof — headless returns 'denied' for notifications
                const origQuery = window.navigator.permissions ? window.navigator.permissions.query : null;
                if (origQuery) {
                    window.navigator.permissions.query = (params) =>
                        params.name === 'notifications'
                            ? Promise.resolve({state: Notification.permission})
                            : origQuery(params);
                }
                // 6) WebGL vendor/renderer — headless reports 'Google Inc.' literally
                const getParameter = WebGLRenderingContext.prototype.getParameter;
                WebGLRenderingContext.prototype.getParameter = function(p) {
                    if (p === 37445) return 'Intel Inc.';           // UNMASKED_VENDOR_WEBGL
                    if (p === 37446) return 'Intel Iris OpenGL Engine'; // UNMASKED_RENDERER_WEBGL
                    return getParameter.call(this, p);
                };
            """)
            log.info("playwright_started", browser=self.browser_kind, headless=self.headless,
                     proxy=bool(self.proxy))

    async def stop(self) -> None:
        async with self._lock:
            for obj in (self._ctx, self._browser):
                if obj is not None:
                    try:
                        await obj.close()
                    except Exception:
                        pass
            if self._pw is not None:
                try:
                    await self._pw.stop()
                except Exception:
                    pass
            self._ctx = None
            self._browser = None
            self._pw = None

    async def _is_alive(self) -> bool:
        if self._ctx is None or self._browser is None:
            return False
        try:
            return bool(self._browser.is_connected())  # type: ignore[attr-defined]
        except Exception:
            return False

    async def _ensure_alive(self) -> None:
        if not await self._is_alive():
            log.warn("playwright_browser_dead")
            await self._restart()

    async def _restart(self) -> None:
        if self._restart_count >= self.max_restarts:
            raise RuntimeError(f"playwright: max_restarts ({self.max_restarts}) exceeded")
        backoff = self.restart_backoff_s * (2 ** self._restart_count) + random.uniform(0, 1)
        self._restart_count += 1
        log.warn("playwright_restarting", attempt=self._restart_count, backoff_s=round(backoff, 1))
        await self.stop()
        await asyncio.sleep(backoff)
        await self.start()

    async def new_page(self) -> Page:
        await self._ensure_alive()
        assert self._ctx is not None
        return await self._ctx.new_page()

    async def run_with_retry(self, task_fn, *args, **kwargs):
        """Run task_fn(page, *args) with automatic browser restart on crash."""
        last_err: Optional[BaseException] = None
        for attempt in range(self.max_restarts + 1):
            try:
                await self._ensure_alive()
                assert self._ctx is not None
                page = await self._ctx.new_page()
                try:
                    return await task_fn(page, *args, **kwargs)
                finally:
                    try:
                        await page.close()
                    except Exception:
                        pass
            except Exception as e:
                last_err = e
                err_name = type(e).__name__
                if attempt >= self.max_restarts:
                    break
                if ("Target" in err_name or "Browser" in err_name or "Crash" in err_name
                        or "Connection" in err_name):
                    log.warn("playwright_crash_retry", error=str(e)[:120])
                    try:
                        await self._restart()
                    except Exception as re:
                        last_err = re
                        break
                    continue
                raise
        if last_err is not None:
            raise last_err


async def scroll_and_collect(page, *, max_scrolls: int, scroll_delay_s: float) -> None:
    """Scroll down repeatedly to trigger lazy-loaded images."""
    last_height = 0
    stuck = 0
    for i in range(max_scrolls):
        await page.evaluate("window.scrollBy(0, document.documentElement.clientHeight * 0.85)")
        await asyncio.sleep(scroll_delay_s + random.uniform(-0.2, 0.3))
        height = await page.evaluate("document.documentElement.scrollHeight")
        if height == last_height:
            stuck += 1
            if stuck >= 3:
                break
        else:
            stuck = 0
        last_height = height


# Back-compat shim for function-based API
async def launch_context(browser_kind: str = "chromium", headless: bool = True,
                         viewport=(1600, 1000)):
    sup = PlaywrightSupervisor(browser_kind=browser_kind, headless=headless, viewport=viewport)
    await sup.start()
    return (sup._pw, sup._browser, sup._ctx)
