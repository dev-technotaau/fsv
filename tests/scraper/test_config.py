"""Config loading + env expansion + CLI overrides."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from data_scraper.config import load_config


def test_env_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_KEY", "abc123")
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent("""
        pexels:
          enabled: true
          api_key: ${MY_KEY}
    """))
    cfg = load_config(p)
    assert cfg.pexels.api_key == "abc123"


def test_override_with_set(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("runtime:\n  target_total_images: 100\n")
    cfg = load_config(p, overrides=["runtime.target_total_images=500",
                                     "runtime.dry_run=true"])
    assert cfg.runtime.target_total_images == 500
    assert cfg.runtime.dry_run is True


def test_default_config_works_without_file():
    cfg = load_config(None)
    assert cfg.runtime.target_total_images > 0
    assert cfg.dedup.phash_hamming_threshold > 0


def test_invalid_override_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("")
    with pytest.raises(Exception):
        load_config(p, overrides=["no_equals_sign"])
