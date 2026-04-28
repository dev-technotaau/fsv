"""Shared pytest fixtures for scraper tests."""
from __future__ import annotations

import io
from pathlib import Path

import pytest


@pytest.fixture
def tmp_images_dir(tmp_path: Path) -> Path:
    d = tmp_path / "images"
    d.mkdir()
    return d


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "dedup.sqlite"


def make_jpeg_bytes(w: int = 1024, h: int = 768, color=(100, 150, 80)) -> bytes:
    """Generate deterministic JPEG bytes. Use distinct colors for near-dup tests."""
    from PIL import Image
    img = Image.new("RGB", (w, h), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
def jpeg_bytes():
    return make_jpeg_bytes


@pytest.fixture
def default_quality_cfg():
    from data_scraper.config import QualityConfig
    return QualityConfig()
