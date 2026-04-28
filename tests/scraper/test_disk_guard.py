import pytest
from pathlib import Path

from data_scraper.disk_guard import DiskGuard


def test_free_gb_returns_number(tmp_path):
    g = DiskGuard(tmp_path, min_free_gb=0.001)
    assert g.free_gb() > 0


def test_check_ok_when_below_threshold(tmp_path):
    g = DiskGuard(tmp_path, min_free_gb=0.001)   # 1 MB requirement
    assert g.check() is True


def test_preflight_raises_on_impossible(tmp_path):
    # 1 EB requirement → guaranteed to fail
    g = DiskGuard(tmp_path, min_free_gb=1_000_000_000)
    with pytest.raises(RuntimeError):
        g.preflight()
