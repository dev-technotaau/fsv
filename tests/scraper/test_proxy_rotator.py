from data_scraper.proxy_rotator import ProxyRotator


def test_empty_rotator_returns_none():
    r = ProxyRotator()
    assert r.next_proxy() is None


def test_rotation():
    r = ProxyRotator(["http://p1", "http://p2", "http://p3"])
    seen = [r.next_proxy() for _ in range(6)]
    # Each proxy used at least once
    assert set(seen) == {"http://p1", "http://p2", "http://p3"}


def test_failure_evicts_temporarily():
    r = ProxyRotator(["http://p1", "http://p2"], failure_cool_down_s=3600)
    r.mark_failure("http://p1")
    # Rotate a bunch — p1 should NOT appear until cool-down expires
    for _ in range(4):
        p = r.next_proxy()
        assert p != "http://p1"


def test_success_clears_failure():
    r = ProxyRotator(["http://p1", "http://p2"], failure_cool_down_s=3600)
    r.mark_failure("http://p1")
    r.mark_success("http://p1")
    # p1 should be eligible again
    seen = set()
    for _ in range(6):
        p = r.next_proxy()
        seen.add(p)
    assert "http://p1" in seen


def test_env_proxies(monkeypatch):
    monkeypatch.setenv("SCRAPER_PROXIES", "http://envA,http://envB")
    r = ProxyRotator()
    assert r.enabled
    seen = set()
    for _ in range(4):
        seen.add(r.next_proxy())
    assert {"http://envA", "http://envB"}.issubset(seen)
