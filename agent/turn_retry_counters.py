"""Unified retry counters for a single conversation turn.

Consolidates all per-turn retry/fallback counters that were previously
scattered as loose attributes on the agent instance. A fresh instance is
created at the start of each turn via ``TurnContext`` and mutated in-place
during the turn loop.

Loop-control variables (``retry_count``, ``max_retries``,
``compression_attempts``, ``fallback_index``) intentionally stay as plain
locals / agent attributes — they are the ``while`` mechanics, not per-turn
recovery bookkeeping, and putting them here would add indirection without
clarifying anything. ``TurnRetryState`` (the inner-attempt recovery guards)
is a separate object created once per API-call attempt.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurnRetryCounters:
    """All per-turn retry/fallback counters in one place.

    A fresh instance is created in ``build_turn_context`` and passed through
    the turn loop. Mutated in-place; no indirection for hot paths.
    """

    # ── Tool-call validation retries ──────────────────────────────────────
    invalid_tool_retries: int = 0
    invalid_json_retries: int = 0

    # ── Empty / incomplete content retries ────────────────────────────────
    empty_content_retries: int = 0
    thinking_prefill_retries: int = 0
    incomplete_scratchpad_retries: int = 0
    codex_incomplete_retries: int = 0
    post_tool_empty_retried: bool = False

    # ── Fallback / content recovery scratch ───────────────────────────────
    last_content_with_tools: str | None = None
    last_content_tools_all_housekeeping: bool = False
    mute_post_response: bool = False

    # ── Unicode sanitisation passes (bounded to 2) ────────────────────────
    unicode_sanitization_passes: int = 0

    def reset_all(self) -> None:
        """Reset all counters to initial state (used at turn start)."""
        self.invalid_tool_retries = 0
        self.invalid_json_retries = 0
        self.empty_content_retries = 0
        self.thinking_prefill_retries = 0
        self.incomplete_scratchpad_retries = 0
        self.codex_incomplete_retries = 0
        self.post_tool_empty_retried = False
        self.last_content_with_tools = None
        self.last_content_tools_all_housekeeping = False
        self.mute_post_response = False
        self.unicode_sanitization_passes = 0

    def reset_empty_response_group(self) -> None:
        """Reset the empty-response subgroup after mid-turn compression."""
        self.empty_content_retries = 0
        self.thinking_prefill_retries = 0
        self.post_tool_empty_retried = False
        self.last_content_with_tools = None
        self.last_content_tools_all_housekeeping = False
        self.mute_post_response = False

    def reset_tool_group(self) -> None:
        """Reset tool-validation subgroup."""
        self.invalid_tool_retries = 0
        self.invalid_json_retries = 0

    def reset_scratchpad_group(self) -> None:
        """Reset scratchpad/continuation subgroup."""
        self.incomplete_scratchpad_retries = 0
        self.codex_incomplete_retries = 0
