from data_scraper.query_priority import PriorityQueryScheduler


def test_low_saved_wins():
    s = PriorityQueryScheduler(["a", "b", "c"], target_per_query=100)
    s.record_saved("a", 50)
    s.record_saved("b", 10)
    s.record_saved("c", 30)
    assert s.next_query() == "b"


def test_excluded_skipped():
    s = PriorityQueryScheduler(["a", "b"], target_per_query=100)
    s.record_saved("a", 5)
    s.record_saved("b", 20)
    # next would normally be 'a'; exclude it
    assert s.next_query(exclude={"a"}) == "b"


def test_target_met_skipped():
    s = PriorityQueryScheduler(["a", "b"], target_per_query=10)
    s.record_saved("a", 20)   # over target
    s.record_saved("b", 3)
    assert s.next_query() == "b"


def test_reorder_for_shortage():
    s = PriorityQueryScheduler(["a", "b", "c"], target_per_query=100)
    s.record_saved("a", 50)
    s.record_saved("b", 0)
    s.record_saved("c", 25)
    ordered = s.reorder_for_shortage()
    assert ordered == ["b", "c", "a"]


def test_all_over_target_returns_none():
    s = PriorityQueryScheduler(["a"], target_per_query=1)
    s.record_saved("a", 5)
    assert s.next_query() is None
