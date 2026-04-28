"""Content moderation — blocked keywords in title/URL + optional NSFW-URL heuristics.

This is a lightweight filter for obviously off-topic / unsafe content. For a
real classifier, enable `google_vision` which uses Google's SafeSearch + label
detection (already integrated via google_vision.py).
"""
from __future__ import annotations

import re
from typing import Optional

from .sources.base import URLCandidate


BLOCK_KEYWORDS = [
    # Explicit content
    "nsfw", "porn", "xxx", "nude", "nudes", "erotic", "sexy", "bikini",
    # Violence
    "blood", "gore", "violence", "weapon",
    # Irrelevant category drift
    "meme", "cartoon", "drawing", "illustration", "vector", "clipart",
    "logo", "icon set", "stock vector", "wallpaper hd download",
]

# These domains often serve galleries that aren't fence-centric even if the
# image happens to include a fence.
OFF_TOPIC_DOMAINS = [
    "deviantart.com", "artstation.com",
]


class ContentFilter:
    def __init__(self, extra_block_keywords: Optional[list[str]] = None,
                 extra_block_domains: Optional[list[str]] = None):
        kw = list(BLOCK_KEYWORDS) + list(extra_block_keywords or [])
        self._keyword_re = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in kw) + r")\b", re.IGNORECASE,
        )
        self._domains = set(d.lower() for d in (list(OFF_TOPIC_DOMAINS) + list(extra_block_domains or [])))

    def check(self, cand: URLCandidate) -> Optional[str]:
        """Return None if OK, else a rejection reason string."""
        url = (cand.url or "").lower()
        title = (cand.title or "").lower()
        page = (cand.origin_page or "").lower()
        q = (cand.query or "").lower()

        # Domain block
        for d in self._domains:
            if d in url or d in page:
                return f"blocked_domain:{d}"

        # Keyword block — but ONLY check title/page, NOT the query we sent ourselves
        combined = f"{title} {page}"
        m = self._keyword_re.search(combined)
        if m:
            return f"blocked_keyword:{m.group(1)}"

        return None
