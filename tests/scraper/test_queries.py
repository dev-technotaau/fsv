from __future__ import annotations

from data_scraper.queries import ALL_STATIC_QUERIES, build_queries, round_robin


def test_static_corpus_has_breadth():
    # ≥ 150 queries across all 8 categories
    assert len(ALL_STATIC_QUERIES) >= 150


def test_static_corpus_is_unique():
    assert len(set(ALL_STATIC_QUERIES)) == len(ALL_STATIC_QUERIES)


def test_build_queries_dedup_custom():
    q = build_queries(static=False, custom=["a", "b", "a"])
    assert q == ["a", "b"]


def test_build_queries_combines():
    q = build_queries(static=True, custom=["novel_query_xyz"])
    assert "novel_query_xyz" in q
    assert len(q) >= len(ALL_STATIC_QUERIES)


def test_build_queries_handles_empty():
    q = build_queries(static=False, custom=None, gemini_extra=0, gemini_api_key=None)
    assert q == []


def test_round_robin():
    it = round_robin(["a", "b", "c"])
    seq = [next(it) for _ in range(7)]
    assert seq == ["a", "b", "c", "a", "b", "c", "a"]


def test_contains_occlusion_category():
    """Sanity: the critical occlusion category is present."""
    corpus = " ".join(ALL_STATIC_QUERIES).lower()
    for term in ("tree branches", "bushes", "hedge", "vines", "plants in front"):
        # at least SOME occlusion hints should be present
        if term in corpus:
            return
    assert False, "expected at least one occlusion-related term in static corpus"


def test_contains_wood_variety():
    corpus = " ".join(ALL_STATIC_QUERIES).lower()
    for kw in ("cedar", "picket", "shadowbox", "redwood"):
        assert kw in corpus
