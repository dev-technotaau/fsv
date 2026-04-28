"""Reddit image scraping via JSON endpoints.

Two modes:
  1. Unauthenticated (no client_id set) — hits www.reddit.com/*.json.
     Rate: ~10 requests/minute per IP. Frequent 429s expected on large runs.

  2. OAuth2 authenticated (client_id + client_secret set) — hits oauth.reddit.com.
     Rate: 60 requests/minute per account. Token auto-refreshes on expiry.

To create OAuth credentials:
  1. Visit https://www.reddit.com/prefs/apps
  2. Click "Create another app...", choose "script" type
  3. Set redirect_uri to http://localhost:8080 (unused for script apps)
  4. After creation: client_id is under the app name, client_secret is the "secret" field
  5. Set user_agent to something identifiable like "ninja-fence-scraper/0.1 by /u/yourusername"
"""
from __future__ import annotations

import base64
import time
from typing import AsyncIterator, Optional

from ..logger import get_logger
from .base import Source, URLCandidate

log = get_logger("source.reddit")


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _url_is_image(url: str) -> bool:
    u = url.lower().split("?", 1)[0]
    return u.endswith(IMAGE_EXTS) or "i.redd.it" in u or "i.imgur.com" in u


class _OAuthToken:
    """Manages an OAuth2 access token with auto-refresh."""
    def __init__(self, downloader, client_id: str, client_secret: str,
                 user_agent: str, username: Optional[str] = None,
                 password: Optional[str] = None):
        self.downloader = downloader
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.username = username
        self.password = password
        self._token: Optional[str] = None
        self._expires_at: float = 0.0

    def _basic_auth(self) -> str:
        raw = f"{self.client_id}:{self.client_secret}".encode()
        return "Basic " + base64.b64encode(raw).decode()

    async def get(self) -> Optional[str]:
        """Return a valid bearer token, refreshing if needed."""
        now = time.monotonic()
        # Refresh 60s before expiry
        if self._token and now < self._expires_at - 60:
            return self._token
        try:
            import httpx
        except ImportError:
            return None

        # Choose grant flow
        if self.username and self.password:
            data = {
                "grant_type": "password",
                "username": self.username,
                "password": self.password,
            }
        else:
            data = {"grant_type": "client_credentials"}

        headers = {
            "Authorization": self._basic_auth(),
            "User-Agent": self.user_agent,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://www.reddit.com/api/v1/access_token",
                    data=data, headers=headers,
                )
            if resp.status_code != 200:
                log.warn("oauth_token_fetch_failed",
                         status=resp.status_code, body=resp.text[:200])
                self._token = None
                return None
            j = resp.json()
            self._token = j.get("access_token")
            expires_in = int(j.get("expires_in", 3600))
            self._expires_at = now + expires_in
            log.info("oauth_token_obtained", expires_in_s=expires_in)
            return self._token
        except Exception as e:
            log.warn("oauth_token_error", error=str(e))
            return None


class RedditSource(Source):
    name = "reddit"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._oauth: Optional[_OAuthToken] = None
        if self.cfg.client_id and self.cfg.client_secret:
            self._oauth = _OAuthToken(
                downloader=self.downloader,
                client_id=self.cfg.client_id,
                client_secret=self.cfg.client_secret,
                user_agent=self.cfg.user_agent,
                username=self.cfg.username,
                password=self.cfg.password,
            )

    @property
    def _host(self) -> str:
        return "oauth.reddit.com" if self._oauth else "www.reddit.com"

    async def _auth_headers(self) -> dict:
        headers = {"User-Agent": self.cfg.user_agent}
        if self._oauth:
            token = await self._oauth.get()
            if token:
                headers["Authorization"] = f"Bearer {token}"
        return headers

    async def discover(self) -> AsyncIterator[URLCandidate]:
        if not self.enabled:
            return
        if self._oauth:
            log.info("using_oauth_endpoint")
        else:
            log.info("using_unauthenticated_endpoint",
                     hint="Set reddit.client_id/client_secret for 6× higher rate limit")

        # 1. Query-driven search across each subreddit
        for sub in self.cfg.subreddits:
            for q in self.queries:
                after: Optional[str] = None
                for _ in range(4):
                    await self.throttle()
                    headers = await self._auth_headers()
                    base = f"https://{self._host}/r/{sub}/search.json"
                    params = {"q": q, "restrict_sr": "1", "limit": 100,
                              "sort": "relevance", "t": "all"}
                    if after:
                        params["after"] = after
                    data = await self.downloader.fetch_json(base, params=params, headers=headers)
                    if not data or "data" not in data:
                        break
                    children = data["data"].get("children", [])
                    if not children:
                        break
                    async for cand in self._emit_from_posts(children, q):
                        yield cand
                    after = data["data"].get("after")
                    if not after:
                        break

            # 2. Recent /new posts in each sub (query-free)
            await self.throttle()
            headers = await self._auth_headers()
            url = f"https://{self._host}/r/{sub}/new.json"
            data = await self.downloader.fetch_json(url, params={"limit": 100}, headers=headers)
            if data and "data" in data:
                async for cand in self._emit_from_posts(data["data"].get("children", []),
                                                        query=f"r/{sub}:new"):
                    yield cand

    async def _emit_from_posts(self, children: list, query: str) -> AsyncIterator[URLCandidate]:
        for c in children:
            post = c.get("data", {}) if isinstance(c, dict) else {}
            if not post or post.get("over_18"):
                continue
            url = post.get("url_overridden_by_dest") or post.get("url")
            if not url or not _url_is_image(url):
                # Fall back to preview image
                preview = post.get("preview", {}).get("images", [])
                if preview:
                    src = preview[0].get("source", {}).get("url")
                    if src:
                        url = src.replace("&amp;", "&")
            if not url or not _url_is_image(url):
                continue
            if self.dedup.url_seen(url):
                continue
            self.dedup.mark_url(url, self.name)
            yield URLCandidate(
                url=url, source=self.name, query=query,
                title=post.get("title"),
                origin_page=f"https://www.reddit.com{post.get('permalink', '')}",
                extra={
                    "subreddit": post.get("subreddit"),
                    "score": post.get("score"),
                    "authenticated": bool(self._oauth),
                },
            )
