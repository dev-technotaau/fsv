"""Tests for structured logger."""
from __future__ import annotations

import json
from pathlib import Path

from data_scraper.logger import StructuredLogger, get_logger, set_correlation_id


def test_writes_json_to_file(tmp_path):
    f = tmp_path / "log.jsonl"
    StructuredLogger.configure(level="DEBUG", file_path=f, console_format="plain")
    log = get_logger("test")
    log.info("hello", n=42, src="pexels")
    log.warn("partial_failure", code=503)
    lines = f.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    assert rec1["msg"] == "hello"
    assert rec1["level"] == "INFO"
    assert rec1["n"] == 42
    assert rec1["src"] == "pexels"


def test_respects_level(tmp_path):
    f = tmp_path / "log.jsonl"
    StructuredLogger.configure(level="WARN", file_path=f, console_format="plain")
    log = get_logger("test2")
    log.info("should_not_appear")
    log.warn("should_appear")
    lines = f.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["msg"] == "should_appear"


def test_correlation_id_propagates(tmp_path):
    f = tmp_path / "log.jsonl"
    StructuredLogger.configure(level="DEBUG", file_path=f, console_format="plain")
    log = get_logger("test3")
    cid = set_correlation_id("abc123xyz")
    log.info("tagged")
    rec = json.loads(f.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["cid"] == "abc123xyz"
