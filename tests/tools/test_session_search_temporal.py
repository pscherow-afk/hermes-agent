"""Tests for temporal filters in session_search (P1 feature)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.session_search_tool import _parse_temporal_filters  # noqa: E402


def test_parse_last_n_days():
    since, until = _parse_temporal_filters(None, None, 7)
    assert since is not None
    assert until is None


def test_parse_iso_date():
    since, until = _parse_temporal_filters("2026-07-01", "2026-07-10", None)
    assert since is not None and until is not None
    assert since < until


def test_parse_relative_keyword():
    since, until = _parse_temporal_filters("last_week", None, None)
    assert since is not None
    assert until is None


def test_parse_none():
    since, until = _parse_temporal_filters(None, None, None)
    assert since is None and until is None


def test_parse_invalid_iso_falls_through():
    # Invalid ISO should not raise; returns (None, None)
    since, until = _parse_temporal_filters("not-a-date", None, None)
    assert since is None
