"""Unit tests for TurnRetryCounters (per-turn retry/fallback counters).

These tests pin the dataclass shape, defaults, and the subgroup-reset
semantics used by the conversation loop. Behavioral guarantees for the
loop itself live in tests/run_agent/ which now exercise the fields via
``agent._retry_counters``.
"""

from __future__ import annotations

from dataclasses import fields

from agent.turn_retry_counters import TurnRetryCounters


EXPECTED_FIELDS = {
    "invalid_tool_retries",
    "invalid_json_retries",
    "empty_content_retries",
    "thinking_prefill_retries",
    "incomplete_scratchpad_retries",
    "codex_incomplete_retries",
    "post_tool_empty_retried",
    "last_content_with_tools",
    "last_content_tools_all_housekeeping",
    "mute_post_response",
    "unicode_sanitization_passes",
}


def test_field_set_matches_contract():
    names = {f.name for f in fields(TurnRetryCounters)}
    assert names == EXPECTED_FIELDS, (
        f"unexpected drift: missing={EXPECTED_FIELDS - names} extra={names - EXPECTED_FIELDS}"
    )


def test_defaults_zero_or_false():
    c = TurnRetryCounters()
    # Integer counters default to 0.
    for name in (
        "invalid_tool_retries",
        "invalid_json_retries",
        "empty_content_retries",
        "thinking_prefill_retries",
        "incomplete_scratchpad_retries",
        "codex_incomplete_retries",
        "unicode_sanitization_passes",
    ):
        assert getattr(c, name) == 0, f"{name} should default to 0"
    # Booleans default to False.
    assert c.post_tool_empty_retried is False
    assert c.last_content_tools_all_housekeeping is False
    assert c.mute_post_response is False
    # Scratch field defaults to None.
    assert c.last_content_with_tools is None


def test_reset_all_clears_everything():
    c = TurnRetryCounters()
    # Mutate every field.
    c.invalid_tool_retries = 3
    c.invalid_json_retries = 2
    c.empty_content_retries = 1
    c.thinking_prefill_retries = 2
    c.incomplete_scratchpad_retries = 1
    c.codex_incomplete_retries = 3
    c.post_tool_empty_retried = True
    c.last_content_with_tools = "x"
    c.last_content_tools_all_housekeeping = True
    c.mute_post_response = True
    c.unicode_sanitization_passes = 2

    c.reset_all()

    assert c.invalid_tool_retries == 0
    assert c.invalid_json_retries == 0
    assert c.empty_content_retries == 0
    assert c.thinking_prefill_retries == 0
    assert c.incomplete_scratchpad_retries == 0
    assert c.codex_incomplete_retries == 0
    assert c.post_tool_empty_retried is False
    assert c.last_content_with_tools is None
    assert c.last_content_tools_all_housekeeping is False
    assert c.mute_post_response is False
    assert c.unicode_sanitization_passes == 0


def test_reset_empty_response_group_scope():
    c = TurnRetryCounters()
    c.invalid_tool_retries = 3          # tool group — must survive
    c.incomplete_scratchpad_retries = 2  # scratchpad group — must survive
    c.unicode_sanitization_passes = 1    # must survive

    c.empty_content_retries = 3
    c.thinking_prefill_retries = 2
    c.post_tool_empty_retried = True
    c.last_content_with_tools = "x"
    c.last_content_tools_all_housekeeping = True
    c.mute_post_response = True

    c.reset_empty_response_group()

    # Empty-response subgroup cleared.
    assert c.empty_content_retries == 0
    assert c.thinking_prefill_retries == 0
    assert c.post_tool_empty_retried is False
    assert c.last_content_with_tools is None
    assert c.last_content_tools_all_housekeeping is False
    assert c.mute_post_response is False
    # Other subgroups untouched.
    assert c.invalid_tool_retries == 3
    assert c.incomplete_scratchpad_retries == 2
    assert c.unicode_sanitization_passes == 1


def test_reset_tool_group_scope():
    c = TurnRetryCounters()
    c.empty_content_retries = 3          # must survive
    c.incomplete_scratchpad_retries = 2  # must survive

    c.invalid_tool_retries = 2
    c.invalid_json_retries = 1

    c.reset_tool_group()

    assert c.invalid_tool_retries == 0
    assert c.invalid_json_retries == 0
    # Others untouched.
    assert c.empty_content_retries == 3
    assert c.incomplete_scratchpad_retries == 2


def test_reset_scratchpad_group_scope():
    c = TurnRetryCounters()
    c.invalid_tool_retries = 3          # must survive
    c.empty_content_retries = 2          # must survive

    c.incomplete_scratchpad_retries = 2
    c.codex_incomplete_retries = 1

    c.reset_scratchpad_group()

    assert c.incomplete_scratchpad_retries == 0
    assert c.codex_incomplete_retries == 0
    # Others untouched.
    assert c.invalid_tool_retries == 3
    assert c.empty_content_retries == 2
