#!/usr/bin/env python3
"""Tests for agent.auto_memory (post-turn auto-memory extraction hook)."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from agent.auto_memory import (
    AutoMemoryExtractor,
    CandidateFact,
    ExtractResult,
    PROMPT_VERSION,
    _parse_candidates,
    _build_user_prompt,
    route_candidate,
)


# ---------------------------------------------------------------------------
# Candidate parsing
# ---------------------------------------------------------------------------

def test_parse_empty_returns_empty():
    assert _parse_candidates("") == []
    assert _parse_candidates("   ") == []
    assert _parse_candidates("no array here") == []


def test_parse_plain_array():
    raw = json.dumps([
        {"target": "user", "content": "Prefers terse replies", "confidence": 0.9},
        {"target": "memory", "content": "Uses pytest -x", "confidence": 0.7},
    ])
    out = _parse_candidates(raw)
    assert len(out) == 2
    assert out[0].target == "user"
    assert out[0].confidence == 0.9
    assert out[1].target == "memory"


def test_parse_strips_fences_and_prose():
    raw = (
        "Here are the facts:\n```json\n"
        + json.dumps([{"target": "user", "content": "Likes tea", "confidence": 0.8}])
        + "\n```\nHope that helps!"
    )
    out = _parse_candidates(raw)
    assert len(out) == 1
    assert out[0].content == "Likes tea"


def test_parse_drops_bad_entries():
    raw = json.dumps([
        {"target": "memory", "content": "Good one", "confidence": 0.9},
        {"target": "bogus", "content": "bad target", "confidence": 0.9},
        {"target": "user", "confidence": 0.9},  # missing content
        "not a dict",
        {"target": "user", "content": "ok", "confidence": "high"},  # bad conf -> 0.0
    ])
    out = _parse_candidates(raw)
    assert [c.content for c in out] == ["Good one", "ok"]
    assert out[1].confidence == 0.0


def test_parse_clamps_confidence():
    raw = json.dumps([
        {"target": "user", "content": "a", "confidence": 1.7},
        {"target": "user", "content": "b", "confidence": -3},
    ])
    out = _parse_candidates(raw)
    assert out[0].confidence == 1.0
    assert out[1].confidence == 0.0


def test_parse_respects_max_candidates():
    raw = json.dumps([
        {"target": "user", "content": f"fact {i}", "confidence": 0.9}
        for i in range(20)
    ])
    out = _parse_candidates(raw)
    assert len(out) == 5  # MAX_CANDIDATES_PER_TURN


# ---------------------------------------------------------------------------
# Routing (pure)
# ---------------------------------------------------------------------------

def test_route_disabled_skips():
    c = CandidateFact("user", "x", 0.99)
    assert route_candidate(c, auto_extract=False, threshold=0.85, approval_required=False) == "skip"


def test_route_high_conf_auto_saves():
    c = CandidateFact("user", "x", 0.9)
    assert route_candidate(c, auto_extract=True, threshold=0.85, approval_required=False) == "auto_save"


def test_route_low_conf_without_gate_skips():
    c = CandidateFact("user", "x", 0.5)
    assert route_candidate(c, auto_extract=True, threshold=0.85, approval_required=False) == "skip"


def test_route_approval_gate_overrides():
    # Even high confidence stages when the approval gate is on.
    c = CandidateFact("user", "x", 0.99)
    assert route_candidate(c, auto_extract=True, threshold=0.85, approval_required=True) == "stage"


def test_route_low_conf_with_gate_stages():
    c = CandidateFact("user", "x", 0.4)
    assert route_candidate(c, auto_extract=True, threshold=0.85, approval_required=True) == "stage"


# ---------------------------------------------------------------------------
# End-to-end run (mocked aux extraction)
# ---------------------------------------------------------------------------

RAW_HIGH = json.dumps([
    {"target": "user", "content": "Prefers Russian, terse", "confidence": 0.95, "rationale": "stated"},
])
RAW_LOW = json.dumps([
    {"target": "memory", "content": "Maybe uses nginx", "confidence": 0.4},
])


class _FakeStore:
    def __init__(self):
        self.adds = []
        self.fail_next = False

    def add(self, target, content):
        if self.fail_next:
            return {"success": False, "error": "would exceed limit"}
        self.adds.append((target, content))
        return {"success": True}


def _make_extractor(
    *,
    auto_extract: bool = True,
    threshold: float = 0.85,
    approval_required: bool = False,
) -> AutoMemoryExtractor:
    return AutoMemoryExtractor(
        auto_extract=auto_extract,
        threshold=threshold,
        approval_required=approval_required,
    )


def test_run_auto_saves_high_confidence(monkeypatch):
    monkeypatch.setattr(
        "agent.auto_memory._call_auxiliary_extract", lambda u, r: RAW_HIGH
    )
    store = _FakeStore()
    ex = _make_extractor()
    res = ex.run(user_text="hi", response_text="ok", memory_store=store)
    assert len(res.auto_saved) == 1
    assert res.auto_saved[0].target == "user"
    assert store.adds == [("user", "Prefers Russian, terse")]
    assert res.staged == []


def test_run_skips_low_confidence(monkeypatch):
    monkeypatch.setattr(
        "agent.auto_memory._call_auxiliary_extract", lambda u, r: RAW_LOW
    )
    store = _FakeStore()
    ex = _make_extractor()
    res = ex.run(user_text="hi", response_text="ok", memory_store=store)
    # No gate -> low conf dropped, nothing auto-saved, nothing staged.
    assert res.auto_saved == []
    assert res.staged == []
    assert store.adds == []


def test_run_stages_when_approval_required(monkeypatch):
    monkeypatch.setattr(
        "agent.auto_memory._call_auxiliary_extract", lambda u, r: RAW_HIGH
    )
    store = _FakeStore()
    ex = _make_extractor(approval_required=True)
    with mock.patch("tools.write_approval.stage_write") as m_stage:
        res = ex.run(user_text="hi", response_text="ok", memory_store=store)
    # High conf but gate on -> staged, NOT auto-saved.
    assert res.auto_saved == []
    assert len(res.staged) == 1
    assert m_stage.called
    assert store.adds == []


def test_run_falls_back_to_stage_on_store_reject(monkeypatch):
    monkeypatch.setattr(
        "agent.auto_memory._call_auxiliary_extract", lambda u, r: RAW_HIGH
    )
    store = _FakeStore()
    store.fail_next = True  # add() reports failure
    ex = _make_extractor()
    with mock.patch("tools.write_approval.stage_write") as m_stage:
        res = ex.run(user_text="hi", response_text="ok", memory_store=store)
    # Auto-save rejected -> fact staged instead of lost.
    assert res.auto_saved == []
    assert len(res.staged) == 1
    assert m_stage.called


def test_run_disabled_short_circuits(monkeypatch):
    spy = mock.Mock(return_value=RAW_HIGH)
    monkeypatch.setattr("agent.auto_memory._call_auxiliary_extract", spy)
    store = _FakeStore()
    ex = _make_extractor(auto_extract=False)
    res = ex.run(user_text="hi", response_text="ok", memory_store=store)
    assert res.candidates == []
    assert store.adds == []
    spy.assert_not_called()


def test_run_no_store_errors(monkeypatch):
    monkeypatch.setattr(
        "agent.auto_memory._call_auxiliary_extract", lambda u, r: RAW_HIGH
    )
    ex = _make_extractor()
    res = ex.run(user_text="hi", response_text="ok", memory_store=None)
    assert res.errors and "no memory_store" in res.errors[0]


def test_run_aux_unavailable_is_safe(monkeypatch):
    monkeypatch.setattr(
        "agent.auto_memory._call_auxiliary_extract", lambda u, r: None
    )
    store = _FakeStore()
    ex = _make_extractor()
    res = ex.run(user_text="hi", response_text="ok", memory_store=store)
    assert res.errors and "aux extraction unavailable" in res.errors[0]
    assert store.adds == []


def test_build_user_prompt_contains_markers():
    p = _build_user_prompt("user says hi", "agent replies ok")
    assert "USER MESSAGE" in p
    assert "AGENT RESPONSE" in p
    assert "user says hi" in p
    assert "agent replies ok" in p


def test_prompt_version_constant():
    assert PROMPT_VERSION and isinstance(PROMPT_VERSION, str)
