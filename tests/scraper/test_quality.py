"""Tests for quality filter + PIL bomb protection."""
from __future__ import annotations

import io

import pytest

from data_scraper.config import QualityConfig
from data_scraper.quality import check_bytes, check_url, harden_pil


def test_url_blocked_domain(default_quality_cfg):
    cfg = QualityConfig(blocked_domains=["bad.com"])
    assert check_url("https://bad.com/x.jpg", cfg) is not None
    assert check_url("https://good.com/x.jpg", cfg) is None


def test_url_bad_extension(default_quality_cfg):
    assert check_url("http://x/foo.svg", default_quality_cfg) == "bad_extension"
    assert check_url("http://x/foo.gif", default_quality_cfg) == "bad_extension"
    assert check_url("http://x/foo.jpg", default_quality_cfg) is None


def test_check_bytes_ok(default_quality_cfg, jpeg_bytes):
    qc = check_bytes(jpeg_bytes(1024, 768), default_quality_cfg)
    assert qc.ok
    assert qc.width == 1024 and qc.height == 768
    assert qc.format == "jpeg"


def test_too_small_resolution(default_quality_cfg, jpeg_bytes):
    qc = check_bytes(jpeg_bytes(200, 200), default_quality_cfg)
    assert not qc.ok
    assert "too_small" in qc.reason


def test_bad_aspect(default_quality_cfg, jpeg_bytes):
    qc = check_bytes(jpeg_bytes(3000, 600), default_quality_cfg)
    assert not qc.ok
    assert "aspect" in qc.reason


def test_corrupt_bytes(default_quality_cfg):
    qc = check_bytes(b"this is not an image" * 5000, default_quality_cfg)
    assert not qc.ok


def test_too_small_bytes(default_quality_cfg):
    qc = check_bytes(b"x" * 100, default_quality_cfg)
    assert not qc.ok
    assert qc.reason == "too_small_bytes"


def test_pil_bomb_cap():
    """harden_pil sets MAX_IMAGE_PIXELS — verify it took effect."""
    from PIL import Image
    harden_pil(max_megapixels=10)    # 10 Mpx cap
    assert Image.MAX_IMAGE_PIXELS == 10 * 1_000_000
