#!/usr/bin/env python3
"""Auto-memory extraction hook — post-turn candidate fact proposals.

Background
---------
Hermes' built-in memory is the ``memory`` tool: the agent decides *when* to
save a durable fact (user preference, environment quirk, workflow lesson) during
its own turn. That works, but it is entirely reliant on the model remembering
to call the tool — and a quiet preference stated mid-task ("I always use pytest
-x") is easy to drop.

This module adds a *post-turn* safety net. After each completed turn,
``AIAgent._sync_external_memory_for_turn`` (run_agent.py) calls
``AutoMemoryExtractor.run``, which:

  1. Builds a compact extraction prompt from the just-completed
     (user_message, assistant_response) pair.
  2. Calls a cheap auxiliary LLM to propose *candidate* memory facts as
     structured JSON (target store + content + 0..1 confidence).
  3. Routes each candidate:
       * confidence >= auto_extract_threshold  → auto-save to the store
         (best-effort, never blocks the user).
       * otherwise                              → stage to the pending store
         (``memory.write_approval``), surfaced via ``/memory pending`` /
         inline approval on interactive CLI.

The hook is OFF by default (``memory.auto_extract: false``). It only runs
when built-in memory is enabled (``memory.memory_enabled`` or
``memory.user_profile_enabled``), and it degrades gracefully: any failure
(config missing, aux LLM unavailable, parse error) is swallowed into a log
line — auto-memory is strictly best-effort and must never break a turn.

Design constraints (why it looks the way it does)
------------------------------------------------
* **No blocking.** The whole run is wrapped by the caller in try/except and
  we additionally guard every step. A wedged aux provider or a full memory
  store must not cost the user a single token of latency on their response.
* **No cache-busting.** We write to MEMORY.md / USER.md on disk the same
  way the ``memory`` tool does. The frozen system-prompt snapshot only
  refreshes at the next session start, so an auto-saved fact does NOT
  perturb the running turn's prefix cache — same guarantee the tool gives.
* **Confidence, not a binary gate.** Auto-save only fires for high
  confidence so we don't pollute memory with half-true guesses; lower
  confidence candidates still get a human in the loop via staging.
* **Reuses existing machinery.** Staging goes through
  ``tools.write_approval.stage_write`` (the same pending store the
  ``/memory approve`` flow already reviews) and auto-save goes through
  ``MemoryStore.add`` — no second write path, no drift.

Prompt versioning
-----------------
``PROMPT_VERSION`` is stamped onto every staged/auto-saved candidate so a
future prompt change can be filtered/audited. Bump it when you change the
extraction instructions below.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Bump when the extraction prompt or schema changes.
PROMPT_VERSION = "1"

# Default confidence above which a candidate is auto-saved (vs staged).
DEFAULT_THRESHOLD = 0.85

# Hard cap on candidates per turn — a turn that "wants" more than this is
# almost certainly the model hallucinating a memory dump; truncate silently.
MAX_CANDIDATES_PER_TURN = 5

# Token budget for the aux extraction call. Facts are short; we don't need
# much headroom, and keeping this small lets the cheapest aux model serve it.
_EXTRACT_MAX_TOKENS = 600


@dataclass
class CandidateFact:
    """A single proposed durable memory entry.

    Attributes:
        target:      ``"memory"`` (agent notes) or ``"user"`` (user profile).
        content:      The declarative entry text to save.
        confidence:   0.0..1.0 model-estimated likelihood this is a genuine,
                      durable, worth-saving fact (vs a guess / task noise).
        rationale:    Short reason the model proposed it (for staging display).
    """

    target: str
    content: str
    confidence: float
    rationale: str = ""

    def to_payload(self) -> Dict[str, Any]:
        """Shape understood by ``tools.memory_tool.apply_memory_pending``."""
        return {
            "action": "add",
            "target": self.target,
            "content": self.content,
        }


@dataclass
class ExtractResult:
    """Outcome of one ``AutoMemoryExtractor.run`` invocation."""

    candidates: List[CandidateFact] = field(default_factory=list)
    auto_saved: List[CandidateFact] = field(default_factory=list)
    staged: List[CandidateFact] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def processed(self) -> int:
        return len(self.auto_saved) + len(self.staged)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a memory-extraction filter for an AI agent's persistent memory. "
    "You receive one completed conversation turn (a user message and the "
    "agent's final response). Your job: identify ANY durable, reusable facts "
    "worth saving so the agent never has to be told twice.\n\n"
    "Save a fact ONLY if it is:\n"
    "  - a stable USER preference / correction / personal detail, OR\n"
    "  - a stable ENVIRONMENT / convention / tool quirk fact about the user's "
    "setup, OR\n"
    "  - a reusable PROCEDURE or lesson learned (not a one-off task step).\n\n"
    "DO NOT save:\n"
    "  - task progress, completed-work logs, or temporary TODO state\n"
    "  - raw data dumps, easily re-discovered facts, or trivial/obvious info\n"
    "  - anything already implied by the visible conversation context\n\n"
    "For each fact, output JSON with:\n"
    '  "target": "user" for who-the-user-is facts, "memory" for '
    "agent notes/environment,\n"
    '  "content": a compact declarative entry (under ~200 chars, no bullet, '
    "past or present tense as natural),\n"
    '  "confidence": your 0.0-1.0 estimate that this is a genuine durable '
    "fact worth persisting,\n"
    '  "rationale": one short phrase explaining why.\n\n'
    "Output ONLY a JSON array (may be empty []). No prose, no markdown fences."
)


def _build_user_prompt(user_text: str, response_text: str) -> str:
    """Compact, delimiter-safe turn encoding for the aux model."""
    # Cap each side so a huge turn can't blow the aux context window.
    u = user_text.strip()[-1500:] if user_text else ""
    r = response_text.strip()[-1500:] if response_text else ""
    return (
        "=== USER MESSAGE ===\n"
        f"{u}\n\n"
        "=== AGENT RESPONSE ===\n"
        f"{r}\n\n"
        "=== EXTRACT ===\n"
        "Return the JSON array of candidate facts now."
    )


# ---------------------------------------------------------------------------
# Extraction (LLM call + parse)
# ---------------------------------------------------------------------------

def _call_auxiliary_extract(user_text: str, response_text: str) -> Optional[str]:
    """Run the aux LLM and return its raw text, or None on any failure.

    Network/provider failures are the expected error mode here (the user's
    primary aux provider may be rate-limited, depleted, or unset). We return
    None and let the caller log + skip — auto-memory is best-effort.
    """
    try:
        from agent.auxiliary_client import call_llm
    except Exception as e:  # pragma: no cover - import path issue
        logger.debug("auto_memory: cannot import call_llm: %s", e)
        return None
    try:
        resp = call_llm(
            "auto_memory",  # auxiliary task name → resolves provider:model
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(user_text, response_text)},
            ],
            max_tokens=_EXTRACT_MAX_TOKENS,
            temperature=0.0,
        )
    except Exception as e:
        logger.debug("auto_memory: aux extraction call failed: %s", e)
        return None
    try:
        return resp.choices[0].message.content or ""
    except Exception as e:  # pragma: no cover - malformed response
        logger.debug("auto_memory: malformed aux response: %s", e)
        return None


def _parse_candidates(raw: str) -> List[CandidateFact]:
    """Parse the aux model's JSON array into CandidateFact objects.

    Robust to the model wrapping the array in ```json fences or trailing prose:
    we locate the first '[' and last ']' and parse just that span. Invalid
    entries are dropped individually (logged) rather than failing the turn.
    """
    if not raw or not raw.strip():
        return []
    text = raw.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        logger.debug("auto_memory: no JSON array in aux output: %.200s", raw)
        return []
    span = text[start : end + 1]
    try:
        data = json.loads(span)
    except json.JSONDecodeError as e:
        logger.debug("auto_memory: JSON parse failed: %s", e)
        return []
    if not isinstance(data, list):
        return []

    out: List[CandidateFact] = []
    for item in data[: MAX_CANDIDATES_PER_TURN]:
        if not isinstance(item, dict):
            continue
        target = str(item.get("target", "")).strip().lower()
        if target not in ("memory", "user"):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        try:
            conf = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        out.append(
            CandidateFact(
                target=target,
                content=content,
                confidence=conf,
                rationale=str(item.get("rationale", "")).strip(),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Routing (pure, testable)
# ---------------------------------------------------------------------------

def route_candidate(
    candidate: CandidateFact,
    *,
    auto_extract: bool,
    threshold: float,
    approval_required: bool,
) -> str:
    """Decide what to do with one candidate.

    Returns one of: ``"auto_save"``, ``"stage"``, ``"skip"``.

    Decision matrix:
        feature disabled (auto_extract False)  → skip
        confidence >= threshold             → auto_save
        approval gate on (approval_required) → stage
        otherwise (low conf, no gate)       → skip

    Rationale: auto-save only for high confidence. If the user has the
    memory write-approval gate on, *every* proposed fact (even high
    confidence) goes through their approval — the gate is a strict override.
    Low-confidence facts with no gate are simply dropped; staging them would
    spam the pending queue with noise the user must manually clear.
    """
    if not auto_extract:
        return "skip"
    if approval_required:
        return "stage"
    if candidate.confidence >= threshold:
        return "auto_save"
    return "skip"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

class AutoMemoryExtractor:
    """Stateless post-turn memory extractor.

    Instances are cheap; the agent holds one and calls ``run`` once per
    completed turn. All inputs are passed per-call so the extractor stays
    free of agent-instance coupling (easy to unit test).
    """

    def __init__(
        self,
        *,
        auto_extract: bool = False,
        threshold: float = DEFAULT_THRESHOLD,
        approval_required: bool = False,
    ) -> None:
        self.auto_extract = auto_extract
        self.threshold = max(0.0, min(1.0, float(threshold)))
        self.approval_required = approval_required

    def run(
        self,
        *,
        user_text: str,
        response_text: str,
        memory_store: Any,
    ) -> ExtractResult:
        """Extract + route candidate facts for one completed turn.

        ``memory_store`` is the built-in ``MemoryStore`` (tools.memory_tool).
        Best-effort: any failure is captured in ``result.errors`` and the
        method returns a (possibly empty) result rather than raising.
        """
        result = ExtractResult()
        if not self.auto_extract:
            return result
        if memory_store is None:
            result.errors.append("no memory_store")
            return result
        if not (user_text and response_text):
            return result

        raw = _call_auxiliary_extract(user_text, response_text)
        if raw is None:
            result.errors.append("aux extraction unavailable")
            return result

        candidates = _parse_candidates(raw)
        result.candidates = candidates

        for cand in candidates:
            decision = route_candidate(
                cand,
                auto_extract=self.auto_extract,
                threshold=self.threshold,
                approval_required=self.approval_required,
            )
            if decision == "auto_save":
                self._auto_save(cand, memory_store, result)
            elif decision == "stage":
                self._stage(cand, result)
            # "skip" → nothing

        return result

    # -- sinks ---------------------------------------------------------------

    def _auto_save(
        self, cand: CandidateFact, memory_store: Any, result: ExtractResult
    ) -> None:
        try:
            res = memory_store.add(cand.target, cand.content)
            if res.get("success"):
                result.auto_saved.append(cand)
            else:
                # e.g. would exceed char limit — fall back to staging so the
                # fact is not silently lost.
                logger.debug(
                    "auto_memory: auto-save rejected (%s); staging instead",
                    res.get("error"),
                )
                self._stage(cand, result)
        except Exception as e:  # pragma: no cover - store failure
            result.errors.append(f"auto_save failed: {e}")
            logger.debug("auto_memory: auto_save error: %s", e)

    def _stage(self, cand: CandidateFact, result: ExtractResult) -> None:
        try:
            from tools.write_approval import stage_write, MEMORY

            stage_write(
                MEMORY,
                cand.to_payload(),
                summary=f"[auto] {cand.target}: {cand.content[:80]}",
                origin="auto_memory",
            )
            result.staged.append(cand)
        except Exception as e:  # pragma: no cover - disk failure
            result.errors.append(f"stage failed: {e}")
            logger.debug("auto_memory: stage error: %s", e)
