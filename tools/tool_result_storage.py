"""Tool result persistence -- preserves large outputs instead of truncating.

Defense against context-window overflow operates at three levels:

1. **Per-tool output cap** (inside each tool): Tools like search_files
   pre-truncate their own output before returning. This is the first line
   of defense and the only one the tool author controls.

2. **Per-result persistence** (maybe_persist_tool_result): After a tool
   returns, if its output exceeds the tool's registered threshold
   (registry.get_max_result_size), the full output is written INTO THE
   SANDBOX temp dir (for example /tmp/hermes-results/{tool_use_id}.txt on
   standard Linux, or $TMPDIR/hermes-results/{tool_use_id}.txt on Termux)
   via env.execute(). The in-context content is replaced with a preview +
   file path reference. The model can read_file to access the full output
   on any backend.

3. **Per-turn aggregate budget** (enforce_turn_budget): After all tool
   results in a single assistant turn are collected, if the total exceeds
   MAX_TURN_BUDGET_CHARS (200K), the largest non-persisted results are
   spilled to disk until the aggregate is under budget. This catches cases
   where many medium-sized results combine to overflow context.

4. **Tool-output compaction** (compact_tool_result): Optional feature-flag
   layer for paid execution profiles. Stores full output as an artifact
   on disk and injects a compact summary + excerpt + artifact_id instead
   of the full raw result. Controlled by `agent.tool_output_compaction`
   in config.yaml. Aims to reduce fresh input tokens.
"""

import hashlib
import json
import logging
import os
import re
import shlex
import time
import uuid

from tools.budget_config import (
    DEFAULT_PREVIEW_SIZE_CHARS,
    BudgetConfig,
    DEFAULT_BUDGET,
)

logger = logging.getLogger(__name__)
PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
STORAGE_DIR = "/tmp/hermes-results"
HEREDOC_MARKER = "HERMES_PERSIST_EOF"
_BUDGET_TOOL_NAME = "__budget_enforcement__"
_UNSAFE_RESULT_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_RESULT_FILENAME_STEM = 120


def _resolve_storage_dir(env) -> str:
    """Return the best temp-backed storage dir for this environment."""
    if env is not None:
        get_temp_dir = getattr(env, "get_temp_dir", None)
        if callable(get_temp_dir):
            try:
                temp_dir = get_temp_dir()
            except Exception as exc:
                logger.debug("Could not resolve env temp dir: %s", exc)
            else:
                if temp_dir:
                    temp_dir = temp_dir.rstrip("/") or "/"
                    return f"{temp_dir}/hermes-results"
    return STORAGE_DIR


def _safe_result_filename(tool_use_id: str) -> str:
    """Return a single safe filename for a tool result id."""
    raw_id = str(tool_use_id or "tool_result")
    safe_stem = _UNSAFE_RESULT_FILENAME_CHARS.sub("_", raw_id).strip("._-")
    changed = safe_stem != raw_id

    if not safe_stem:
        safe_stem = "tool_result"
        changed = True

    if changed or len(safe_stem) > _MAX_RESULT_FILENAME_STEM:
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:12]
        safe_stem = safe_stem[:_MAX_RESULT_FILENAME_STEM].rstrip("._-") or "tool_result"
        safe_stem = f"{safe_stem}_{digest}"

    return f"{safe_stem}.txt"


def generate_preview(content: str, max_chars: int = DEFAULT_PREVIEW_SIZE_CHARS) -> tuple[str, bool]:
    """Truncate at last newline within max_chars. Returns (preview, has_more)."""
    if len(content) <= max_chars:
        return content, False
    truncated = content[:max_chars]
    last_nl = truncated.rfind("\n")
    if last_nl > max_chars // 2:
        truncated = truncated[:last_nl + 1]
    return truncated, True


def _heredoc_marker(content: str) -> str:
    """Return a heredoc delimiter that doesn't collide with content."""
    if HEREDOC_MARKER not in content:
        return HEREDOC_MARKER
    return f"HERMES_PERSIST_{uuid.uuid4().hex[:8]}"


def _write_to_sandbox(content: str, remote_path: str, env) -> bool:
    """Write content into the sandbox via env.execute(). Returns True on success.

    Pushes ``content`` through stdin rather than embedding it in the command
    string. Linux's ``MAX_ARG_STRLEN`` caps any single argv element at 128 KB
    (32 * PAGE_SIZE), so the previous heredoc-in-the-command-string approach
    silently failed with ``OSError: [Errno 7] Argument list too long`` for any
    tool result over ~128 KB — exactly the case persistence exists to handle.
    Routing through stdin removes that ceiling on local + ssh (``_stdin_mode
    == "pipe"``); remote backends with ``_stdin_mode == "heredoc"`` keep their
    existing API-body sized limit, which is orders of magnitude larger than
    the exec-arg ceiling.
    """
    storage_dir = os.path.dirname(remote_path)
    cmd = f"mkdir -p {shlex.quote(storage_dir)} && cat > {shlex.quote(remote_path)}"
    result = env.execute(cmd, timeout=30, stdin_data=content)
    return result.get("returncode", 1) == 0


def _build_persisted_message(
    preview: str,
    has_more: bool,
    original_size: int,
    file_path: str,
) -> str:
    """Build the <persisted-output> replacement block."""
    size_kb = original_size / 1024
    if size_kb >= 1024:
        size_str = f"{size_kb / 1024:.1f} MB"
    else:
        size_str = f"{size_kb:.1f} KB"

    msg = f"{PERSISTED_OUTPUT_TAG}\n"
    msg += f"This tool result was too large ({original_size:,} characters, {size_str}).\n"
    msg += f"Full output saved to: {file_path}\n"
    msg += "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
    msg += f"Preview (first {len(preview)} chars):\n"
    msg += preview
    if has_more:
        msg += "\n..."
    msg += f"\n{PERSISTED_OUTPUT_CLOSING_TAG}"
    return msg


def maybe_persist_tool_result(
    content: str,
    tool_name: str,
    tool_use_id: str,
    env=None,
    config: BudgetConfig = DEFAULT_BUDGET,
    threshold: int | float | None = None,
) -> str:
    """Layer 2: persist oversized result into the sandbox, return preview + path.

    Writes via env.execute() so the file is accessible from any backend
    (local, Docker, SSH, Modal, Daytona). Falls back to inline truncation
    if write fails or no env is available.

    Args:
        content: Raw tool result string.
        tool_name: Name of the tool (used for threshold lookup).
        tool_use_id: Unique ID for this tool call (used as filename).
        env: The active BaseEnvironment instance, or None.
        config: BudgetConfig controlling thresholds and preview size.
        threshold: Explicit override; takes precedence over config resolution.

    Returns:
        Original content if small, or <persisted-output> replacement.
    """
    effective_threshold = threshold if threshold is not None else config.resolve_threshold(tool_name)

    if effective_threshold == float("inf"):
        return content

    if len(content) <= effective_threshold:
        return content

    storage_dir = _resolve_storage_dir(env)
    remote_path = f"{storage_dir}/{_safe_result_filename(tool_use_id)}"
    preview, has_more = generate_preview(content, max_chars=config.preview_size)

    if env is not None:
        try:
            if _write_to_sandbox(content, remote_path, env):
                logger.info(
                    "Persisted large tool result: %s (%s, %d chars -> %s)",
                    tool_name, tool_use_id, len(content), remote_path,
                )
                return _build_persisted_message(preview, has_more, len(content), remote_path)
        except Exception as exc:
            logger.warning("Sandbox write failed for %s: %s", tool_use_id, exc)

    logger.info(
        "Inline-truncating large tool result: %s (%d chars, no sandbox write)",
        tool_name, len(content),
    )
    return (
        f"{preview}\n\n"
        f"[Truncated: tool response was {len(content):,} chars. "
        f"Full output could not be saved to sandbox.]"
    )


def enforce_turn_budget(
    tool_messages: list[dict],
    env=None,
    config: BudgetConfig = DEFAULT_BUDGET,
) -> list[dict]:
    """Layer 3: enforce aggregate budget across all tool results in a turn.

    If total chars exceed budget, persist the largest non-persisted results
    first (via sandbox write) until under budget. Already-persisted results
    are skipped.

    Mutates the list in-place and returns it.
    """
    candidates = []
    total_size = 0
    for i, msg in enumerate(tool_messages):
        content = msg.get("content", "")
        size = len(content)
        total_size += size
        if PERSISTED_OUTPUT_TAG not in content:
            candidates.append((i, size))

    if total_size <= config.turn_budget:
        return tool_messages

    candidates.sort(key=lambda x: x[1], reverse=True)

    for idx, size in candidates:
        if total_size <= config.turn_budget:
            break
        msg = tool_messages[idx]
        content = msg["content"]
        tool_use_id = msg.get("tool_call_id", f"budget_{idx}")

        replacement = maybe_persist_tool_result(
            content=content,
            tool_name=_BUDGET_TOOL_NAME,
            tool_use_id=tool_use_id,
            env=env,
            config=config,
            threshold=0,
        )
        if replacement != content:
            total_size -= size
            total_size += len(replacement)
            tool_messages[idx]["content"] = replacement
            logger.info(
                "Budget enforcement: persisted tool result %s (%d chars)",
                tool_use_id, size,
            )

    return tool_messages


# ── Layer 4: Tool-output compaction (feature-flag) ──────────────

COMPACT_OUTPUT_TAG = "<compacted-output>"
COMPACT_OUTPUT_CLOSING_TAG = "</compacted-output>"


_COMPACTION_DEFAULT = {"enabled": False, "threshold": 5000, "excerpt_size": 500}
_COMPACTION_CACHE_TTL_S = 30.0
_compaction_cache: list = []  # [monotonic_ts, config_dict]


def _read_compaction_config() -> dict:
    """Read the tool_output_compaction config section.

    Returns a dict with keys: enabled, threshold, excerpt_size.
    Defaults to disabled when config is inaccessible.

    Result is cached for a short TTL so the per-tool-call overhead of
    re-reading config.yaml is amortized across a turn.
    """
    now = time.monotonic()
    if _compaction_cache and now - _compaction_cache[0] < _COMPACTION_CACHE_TTL_S:
        return _compaction_cache[1]
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        agent_cfg = cfg.get("agent", {})
        compaction = agent_cfg.get("tool_output_compaction", {})
        if not isinstance(compaction, dict):
            compaction = _COMPACTION_DEFAULT
    except Exception:
        compaction = _COMPACTION_DEFAULT
    _compaction_cache[:] = [now, compaction]
    return compaction


def _compact_summary(content: str) -> str:
    """Generate a one-line summary of the tool output content.

    For JSON content, describes the structure (keys for dict, count for list).
    Otherwise uses the first meaningful line of content.
    """
    if not content:
        return "Empty result"

    first_line = content.split("\n", 1)[0].strip()
    if first_line:
        # Try to parse JSON and extract a meaningful summary
        if first_line.startswith("{"):
            try:
                data = json.loads(content[:2000])
                if isinstance(data, dict):
                    keys = list(data.keys())
                    return f"JSON object with keys: {', '.join(keys[:5])}{'...' if len(keys) > 5 else ''}"
                if isinstance(data, list):
                    return f"JSON array of {len(data)} items"
            except (json.JSONDecodeError, ValueError):
                pass
        if first_line.startswith("["):
            try:
                data = json.loads(content[:2000])
                if isinstance(data, list):
                    return f"JSON array of {len(data)} items"
            except (json.JSONDecodeError, ValueError):
                pass
        # Use the first meaningful line
        if len(first_line) > 120:
            return first_line[:117] + "..."
        if first_line:
            return first_line
    size_kb = len(content) / 1024
    return f"Tool result ({size_kb:.1f} KB)"


def _format_compact_block(
    content: str,
    excerpt: str,
    artifact_id: str,
    original_size: int,
    remote_path: str,
) -> str:
    """Build the <compacted-output> replacement block."""
    size_kb = original_size / 1024
    if size_kb >= 1024:
        size_str = f"{size_kb / 1024:.1f} MB"
    else:
        size_str = f"{size_kb:.1f} KB"

    compact_summary = _compact_summary(content)

    msg = f"{COMPACT_OUTPUT_TAG}\n"
    msg += f"Tool result: {compact_summary}\n"
    msg += f"Size: {original_size:,} chars ({size_str})\n"
    msg += f"Artifact: {artifact_id}\n"
    msg += f"Full output saved to: {remote_path}\n"
    msg += "Use read_file with the path above to access the full output.\n"
    msg += "\n"
    msg += f"Excerpt:\n{excerpt}\n"
    msg += f"{COMPACT_OUTPUT_CLOSING_TAG}"
    return msg


def compact_tool_result(
    content: str,
    tool_name: str,
    tool_use_id: str,
    env=None,
) -> str:
    """Layer 4: compact tool output for paid execution profiles.

    When the feature flag ``agent.tool_output_compaction.enabled`` is True
    in config.yaml, outputs larger than the configured threshold are saved
    to disk as artifacts and replaced with a compact summary + excerpt +
    artifact_id. This reduces fresh input tokens for paid models.

    The full output is written to the same sandbox storage dir as the
    existing persistence layer (``/tmp/hermes-results/``), so read_file
    can access it regardless of terminal backend.

    Args:
        content: Raw tool result string.
        tool_name: Name of the tool (for logging).
        tool_use_id: Unique ID for this tool call (used as filename).
        env: The active BaseEnvironment instance, or None.

    Returns:
        Original content if compaction is disabled or content is small,
        or <compacted-output> replacement block.
    """
    compaction = _read_compaction_config()
    if not compaction.get("enabled", False):
        return content

    threshold = int(compaction.get("threshold", 5000))
    excerpt_size = int(compaction.get("excerpt_size", 500))

    if len(content) <= threshold:
        return content

    # Skip multimodal results (they are already compact representations)
    from agent.tool_dispatch_helpers import _is_multimodal_tool_result

    if _is_multimodal_tool_result(content):
        return content

    # Skip already-persisted/compacted content (don't double-wrap)
    if COMPACT_OUTPUT_TAG in content or PERSISTED_OUTPUT_TAG in content:
        return content

    storage_dir = _resolve_storage_dir(env)
    filename = _safe_result_filename(tool_use_id)
    remote_path = f"{storage_dir}/{filename}"
    artifact_id = f"tool:{tool_name}/{tool_use_id}"

    excerpt, has_more = generate_preview(content, max_chars=excerpt_size)

    # Write full content to sandbox
    write_ok = False
    if env is not None:
        try:
            write_ok = _write_to_sandbox(content, remote_path, env)
        except Exception as exc:
            logger.warning(
                "Compaction sandbox write failed for %s: %s", tool_use_id, exc
            )

    if write_ok:
        logger.info(
            "Compacted tool result: %s (%s, %d chars -> %s)",
            tool_name, tool_use_id, len(content), remote_path,
        )
        return _format_compact_block(
            content=content,
            excerpt=excerpt + ("\n..." if has_more else ""),
            artifact_id=artifact_id,
            original_size=len(content),
            remote_path=remote_path,
        )

    # If sandbox write failed, still compact inline (no artifact reference)
    logger.info(
        "Inline-compacting tool result: %s (%d chars, no sandbox write)",
        tool_name, len(content),
    )
    continuation = "\n..." if has_more else ""
    return (
        f"{COMPACT_OUTPUT_TAG}\n"
        f"Tool result: {_compact_summary(content)}\n"
        f"Size: {len(content):,} chars\n"
        f"Excerpt:\n{excerpt}\n"
        f"{continuation}\n"
        f"{COMPACT_OUTPUT_CLOSING_TAG}"
    )