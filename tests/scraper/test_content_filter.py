from data_scraper.content_filter import ContentFilter
from data_scraper.sources.base import URLCandidate


def test_accepts_clean_candidate():
    cf = ContentFilter()
    c = URLCandidate(url="https://example.com/img.jpg", source="test",
                     query="cedar fence", title="Beautiful cedar fence",
                     origin_page="https://example.com/blog/fences")
    assert cf.check(c) is None


def test_rejects_nsfw_title():
    cf = ContentFilter()
    c = URLCandidate(url="https://x.com/a.jpg", source="test",
                     title="nude fence model")
    assert cf.check(c) is not None
    assert "blocked_keyword" in cf.check(c)


def test_rejects_blocked_domain():
    cf = ContentFilter()
    c = URLCandidate(url="https://deviantart.com/art/fence", source="test",
                     title="fence")
    assert cf.check(c) is not None
    assert "blocked_domain" in cf.check(c)


def test_query_itself_not_checked():
    """Our own query text should never trigger the block list."""
    cf = ContentFilter()
    c = URLCandidate(url="https://x.com/img.jpg", source="test",
                     query="meme reference",   # user wouldn't put this but test it's allowed
                     title="cedar fence at sunset")
    # title is clean, url is clean → accept regardless of query text
    assert cf.check(c) is None


def test_extra_keyword_blocklist():
    cf = ContentFilter(extra_block_keywords=["yellowbrick"])
    c = URLCandidate(url="http://x/a.jpg", source="t", title="yellowbrick fence")
    assert cf.check(c) is not None
