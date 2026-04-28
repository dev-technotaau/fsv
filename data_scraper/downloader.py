"""Async HTTP downloader with retry, backoff, proxy rotation, circuit breaker,
and adaptive rate limiting from response headers."""
from __future__ import annotations

import asyncio
import random
from typing import Optional

try:
    import httpx
except ImportError:
    httpx = None

from .circuit_breaker import CircuitBreaker, CircuitOpenError
from .logger import get_logger
from .proxy_rotator import ProxyRotator

log = get_logger("downloader")


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]


class Downloader:
    """Async HTTP client with:
        - shared connection pool (httpx AsyncClient)
        - exponential-backoff retry
        - proxy rotation (optional, via ProxyRotator)
        - circuit breaker (per host, optional)
        - adaptive rate limiting from Retry-After / X-RateLimit-Reset
    """

    def __init__(self, timeout_s: int = 30, max_retries: int = 3, max_bytes: int = 25_000_000,
                 proxy_rotator: Optional[ProxyRotator] = None,
                 use_circuit_breakers: bool = True):
        if httpx is None:
            raise ImportError("httpx required: pip install httpx")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_bytes = max_bytes
        self.proxy_rotator = proxy_rotator
        self.use_circuit_breakers = use_circuit_breakers

        limits = httpx.Limits(max_connections=64, max_keepalive_connections=32)
        self.client = httpx.AsyncClient(
            timeout=timeout_s, limits=limits, follow_redirects=True, http2=False,
            headers={"User-Agent": random.choice(USER_AGENTS),
                     "Accept": "image/*,*/*;q=0.8",
                     "Accept-Encoding": "gzip, deflate"},
        )
        # Per-host circuit breakers (lazy-created)
        self._breakers: dict[str, CircuitBreaker] = {}
        # Adaptive rate suppressions: host → monotonic resume time
        self._host_cool: dict[str, float] = {}
        self._host_cool_lock = asyncio.Lock()
        # Per-host concurrency cap — prevents 16 workers hammering one CDN and
        # tripping anti-DDoS. Workers block on the semaphore, not the host cooldown.
        self._host_sems: dict[str, asyncio.Semaphore] = {}
        self.max_per_host = 4
        # Upper bound on cooldown to avoid cumulative backoff starving workers
        self.max_cool_s = 120.0
        # Proactive per-host minimum inter-request interval (seconds).
        # Hosts that 429 aggressively get a pre-emptive spacing so we never trigger it.
        # pixabay CDN throttles well below 1 req/sec; 3s spacing (~20/min) lands safely.
        self._host_min_interval: dict[str, float] = {
            "pixabay.com": 3.0,
            "cdn.pixabay.com": 3.0,
        }
        self._host_last_fetch: dict[str, float] = {}
        self._host_interval_lock = asyncio.Lock()
        # Per-host User-Agent override. Wikimedia (API + upload.wikimedia.org)
        # returns 403 to browser-like UAs — they require a contact-info UA per
        # https://w.wiki/4wJS. Keyed by host SUFFIX.
        self._host_ua_suffix: list[tuple[str, str]] = [
            ("wikimedia.org",
             "ninja-fence-scraper/0.1 "
             "(https://github.com/technotau/ninja-fence; send@technotau.com) httpx/python"),
        ]

    async def close(self) -> None:
        await self.client.aclose()

    def _breaker_for(self, key: str) -> CircuitBreaker:
        if key not in self._breakers:
            self._breakers[key] = CircuitBreaker(key, failure_threshold=5, cool_down_s=60.0)
        return self._breakers[key]

    def _host_of(self, url: str) -> str:
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc or "unknown"
        except Exception:
            return "unknown"

    def _sem_for(self, host: str) -> asyncio.Semaphore:
        if host not in self._host_sems:
            self._host_sems[host] = asyncio.Semaphore(self.max_per_host)
        return self._host_sems[host]

    async def _respect_min_interval(self, host: str) -> None:
        """Enforce a pre-emptive minimum interval between consecutive fetches
        to the same host. Prevents triggering 429 on fragile CDNs (e.g. pixabay)."""
        interval = self._host_min_interval.get(host)
        if interval is None:
            return
        import time
        async with self._host_interval_lock:
            last = self._host_last_fetch.get(host, 0.0)
            now = time.monotonic()
            wait = (last + interval) - now
            if wait > 0:
                # Sleep INSIDE the lock so concurrent callers serialize naturally.
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._host_last_fetch[host] = now

    async def _wait_for_host_cooldown(self, host: str) -> None:
        """If a host is in adaptive cool-down, wait it out."""
        async with self._host_cool_lock:
            import time
            resume_at = self._host_cool.get(host, 0)
            now = time.monotonic()
            if resume_at > now:
                delay = resume_at - now
                log.debug("host_cooling", host=host, delay_s=round(delay, 1))
        if resume_at > now:
            await asyncio.sleep(delay)

    async def _mark_host_cool(self, host: str, seconds: float) -> None:
        """Set cooldown to `seconds` from now. Clamped to max_cool_s so repeated
        429s don't accumulate into unbounded backoff."""
        import time
        seconds = min(seconds, self.max_cool_s)
        async with self._host_cool_lock:
            self._host_cool[host] = time.monotonic() + seconds

    def _parse_retry_after(self, resp) -> Optional[float]:
        """Parse Retry-After (seconds or HTTP date)."""
        ra = resp.headers.get("retry-after") if hasattr(resp, "headers") else None
        if not ra:
            return None
        try:
            return float(ra)
        except ValueError:
            try:
                from email.utils import parsedate_to_datetime
                import time
                dt = parsedate_to_datetime(ra)
                return max(0, dt.timestamp() - time.time())
            except Exception:
                return None

    def _parse_rate_limit_reset(self, resp) -> Optional[float]:
        """Parse X-RateLimit-Reset (unix epoch or seconds)."""
        headers = resp.headers if hasattr(resp, "headers") else {}
        for k in ("x-ratelimit-reset", "x-rate-limit-reset", "ratelimit-reset"):
            val = headers.get(k)
            if val:
                try:
                    import time
                    v = float(val)
                    # Heuristic: if > 10^9, treat as absolute unix epoch
                    if v > 1_000_000_000:
                        return max(0, v - time.time())
                    return max(0, v)
                except ValueError:
                    continue
        return None

    async def fetch_image(self, url: str) -> Optional[bytes]:
        """Download image bytes with retry. Returns bytes or None."""
        host = self._host_of(url)
        breaker = self._breaker_for(host) if self.use_circuit_breakers else None
        sem = self._sem_for(host)

        async def _do_fetch():
            # Acquire per-host concurrency slot BEFORE checking cooldown.
            # If cooldown > 10s, bail quickly and let the worker try another URL
            # rather than sitting idle.
            import time
            async with self._host_cool_lock:
                resume_at = self._host_cool.get(host, 0)
            remaining = resume_at - time.monotonic()
            if remaining > 10:
                return None
            async with sem:
                await self._wait_for_host_cooldown(host)
                await self._respect_min_interval(host)
                proxy = self.proxy_rotator.next_proxy() if self.proxy_rotator else None
                # Per-host UA override beats the random browser UA when a site's
                # robot policy requires a specific contact-info UA (e.g. Wikimedia).
                ua_override = next((ua for suffix, ua in self._host_ua_suffix
                                     if host.endswith(suffix)), None)
                headers = {"User-Agent": ua_override or random.choice(USER_AGENTS)}
                # Auto-Referer from URL's own host — many CDNs 403 without it.
                # (Skip for hosts that require a specific UA; they usually don't need Referer.)
                if ua_override is None:
                    headers["Referer"] = f"https://{host}/"
                async with self.client.stream("GET", url, headers=headers) as resp:
                    if resp.status_code == 429 or resp.status_code == 503:
                        retry_after = self._parse_retry_after(resp) or self._parse_rate_limit_reset(resp) or 30
                        log.warn("rate_limited", host=host, retry_after_s=round(retry_after, 1),
                                 status=resp.status_code)
                        await self._mark_host_cool(host, retry_after)
                        raise RuntimeError(f"HTTP {resp.status_code}")
                    if resp.status_code != 200:
                        if resp.status_code in (403, 404, 410):
                            if proxy:
                                self.proxy_rotator.mark_failure(proxy) if self.proxy_rotator else None
                            return None
                        raise RuntimeError(f"HTTP {resp.status_code}")
                    ctype = resp.headers.get("content-type", "").lower()
                    if ctype and "image" not in ctype and "octet-stream" not in ctype:
                        return None
                    body = bytearray()
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        body.extend(chunk)
                        if len(body) > self.max_bytes:
                            return None
                    if proxy:
                        self.proxy_rotator.mark_success(proxy) if self.proxy_rotator else None
                    return bytes(body)

        for attempt in range(self.max_retries):
            try:
                if breaker:
                    return await breaker.call(_do_fetch)
                return await _do_fetch()
            except CircuitOpenError:
                log.debug("circuit_open_skip", host=host)
                return None
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as e:
                if attempt == self.max_retries - 1:
                    log.debug("fetch_give_up", url=url[:120], error=str(e)[:80])
                    return None
                await asyncio.sleep(min(2 ** attempt, 10) + random.uniform(0, 0.5))
            except Exception:
                return None
        return None

    async def fetch_json(self, url: str, params: Optional[dict] = None,
                         headers: Optional[dict] = None) -> Optional[dict]:
        host = self._host_of(url)
        breaker = self._breaker_for(host) if self.use_circuit_breakers else None
        sem = self._sem_for(host)

        async def _do():
            async with sem:
                await self._wait_for_host_cooldown(host)
                resp = await self.client.get(url, params=params, headers=headers or {})
                if resp.status_code == 429:
                    ra = self._parse_retry_after(resp) or self._parse_rate_limit_reset(resp) or 30
                    log.warn("api_rate_limited", host=host, retry_after_s=round(ra, 1))
                    await self._mark_host_cool(host, ra)
                    raise RuntimeError("HTTP 429")
                if resp.status_code >= 500:
                    raise RuntimeError(f"HTTP {resp.status_code}")
                if resp.status_code >= 400:
                    return None
                return resp.json()

        for attempt in range(self.max_retries):
            try:
                if breaker:
                    return await breaker.call(_do)
                return await _do()
            except CircuitOpenError:
                return None
            except Exception:
                if attempt == self.max_retries - 1:
                    return None
                await asyncio.sleep(min(2 ** attempt, 10))
        return None

    async def fetch_text(self, url: str, headers: Optional[dict] = None) -> Optional[str]:
        """GET a URL and return the response body as text (for HTML scraping)."""
        host = self._host_of(url)
        breaker = self._breaker_for(host) if self.use_circuit_breakers else None
        sem = self._sem_for(host)

        async def _do():
            async with sem:
                await self._wait_for_host_cooldown(host)
                h = {"User-Agent": random.choice(USER_AGENTS),
                     "Accept": "text/html,application/xhtml+xml,*/*;q=0.8"}
                if headers:
                    h.update(headers)
                resp = await self.client.get(url, headers=h)
                if resp.status_code == 429:
                    ra = self._parse_retry_after(resp) or self._parse_rate_limit_reset(resp) or 30
                    await self._mark_host_cool(host, ra)
                    raise RuntimeError("HTTP 429")
                if resp.status_code >= 500:
                    raise RuntimeError(f"HTTP {resp.status_code}")
                if resp.status_code >= 400:
                    return None
                ctype = resp.headers.get("content-type", "").lower()
                if ctype and "html" not in ctype and "text" not in ctype:
                    return None
                return resp.text

        for attempt in range(self.max_retries):
            try:
                if breaker:
                    return await breaker.call(_do)
                return await _do()
            except CircuitOpenError:
                return None
            except Exception:
                if attempt == self.max_retries - 1:
                    return None
                await asyncio.sleep(min(2 ** attempt, 10))
        return None
