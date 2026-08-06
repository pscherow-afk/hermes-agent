"""Tests for tools/tool_result_storage.py -- 3-layer tool result persistence."""

import pytest
from unittest.mock import MagicMock, patch

from tools.budget_config import (
    DEFAULT_RESULT_SIZE_CHARS,
    DEFAULT_PREVIEW_SIZE_CHARS,
    BudgetConfig,
)
from tools.tool_result_storage import (
    HEREDOC_MARKER,
    PERSISTED_OUTPUT_TAG,
    PERSISTED_OUTPUT_CLOSING_TAG,
    STORAGE_DIR,
    _build_persisted_message,
    _heredoc_marker,
    _resolve_storage_dir,
    _safe_result_filename,
    _write_to_sandbox,
    compact_tool_result,
    COMPACT_OUTPUT_TAG,
    COMPACT_OUTPUT_CLOSING_TAG,
    enforce_turn_budget,
    generate_preview,
    maybe_persist_tool_result,
)


# ── generate_preview ──────────────────────────────────────────────────

class TestGeneratePreview:
    def test_short_content_unchanged(self):
        text = "short result"
        preview, has_more = generate_preview(text)
        assert preview == text
        assert has_more is False


    def test_exact_boundary(self):
        text = "x" * DEFAULT_PREVIEW_SIZE_CHARS
        preview, has_more = generate_preview(text)
        assert preview == text
        assert has_more is False


# ── _heredoc_marker ───────────────────────────────────────────────────

class TestHeredocMarker:
    def test_default_marker_when_no_collision(self):
        assert _heredoc_marker("normal content") == HEREDOC_MARKER

    def test_uuid_marker_on_collision(self):
        content = f"some text with {HEREDOC_MARKER} embedded"
        marker = _heredoc_marker(content)
        assert marker != HEREDOC_MARKER
        assert marker.startswith("HERMES_PERSIST_")
        assert marker not in content


# ── _write_to_sandbox ─────────────────────────────────────────────────

class TestWriteToSandbox:
    def test_success(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        result = _write_to_sandbox("hello world", "/tmp/hermes-results/abc.txt", env)
        assert result is True
        env.execute.assert_called_once()
        cmd = env.execute.call_args[0][0]
        assert "mkdir -p" in cmd
        # Content travels through stdin, NOT inside the command string —
        # otherwise large content would hit Linux's 128 KB MAX_ARG_STRLEN
        # ceiling on `bash -c <cmd>` (#22906).
        assert "hello world" not in cmd
        assert env.execute.call_args[1]["stdin_data"] == "hello world"


    def test_large_content_via_stdin(self):
        """Regression: 200 KB content exceeds Linux MAX_ARG_STRLEN (128 KB).
        It must travel via stdin, never inside the command string."""
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        big = "x" * 200_000
        _write_to_sandbox(big, "/tmp/hermes-results/big.txt", env)
        cmd = env.execute.call_args[0][0]
        assert len(cmd) < 1_000  # cmd is just `mkdir -p X && cat > Y`
        assert env.execute.call_args[1]["stdin_data"] == big


    def test_path_with_spaces_is_quoted(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        remote_path = "/tmp/hermes results/abc file.txt"
        _write_to_sandbox("content", remote_path, env)
        cmd = env.execute.call_args[0][0]
        assert "'/tmp/hermes results'" in cmd
        assert "'/tmp/hermes results/abc file.txt'" in cmd

    def test_shell_metacharacters_neutralized(self):
        """Paths with shell metacharacters must be quoted to prevent injection."""
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        malicious_path = "/tmp/hermes-results/$(whoami).txt"
        _write_to_sandbox("content", malicious_path, env)
        cmd = env.execute.call_args[0][0]
        # The $() must not appear unquoted — shlex.quote wraps it
        assert "'/tmp/hermes-results/$(whoami).txt'" in cmd

    def test_semicolon_injection_neutralized(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        malicious_path = "/tmp/x; rm -rf /; echo .txt"
        _write_to_sandbox("content", malicious_path, env)
        cmd = env.execute.call_args[0][0]
        # The semicolons must be inside quotes, not acting as command separators
        assert "'/tmp/x; rm -rf /; echo .txt'" in cmd


class TestResolveStorageDir:
    def test_defaults_to_storage_dir_without_env(self):
        assert _resolve_storage_dir(None) == STORAGE_DIR

    def test_uses_env_temp_dir_when_available(self):
        env = MagicMock()
        env.get_temp_dir.return_value = "/data/data/com.termux/files/usr/tmp"
        assert _resolve_storage_dir(env) == "/data/data/com.termux/files/usr/tmp/hermes-results"


class TestSafeResultFilename:
    def test_preserves_normal_tool_call_id(self):
        assert _safe_result_filename("tc_456") == "tc_456.txt"

    def test_replaces_path_and_shell_metacharacters(self):
        filename = _safe_result_filename("../outside/$(whoami);x")
        assert filename.startswith("outside_whoami_x_")
        assert filename.endswith(".txt")
        assert "/" not in filename
        assert "$" not in filename
        assert ";" not in filename


# ── _build_persisted_message ──────────────────────────────────────────

class TestBuildPersistedMessage:
    def test_structure(self):
        msg = _build_persisted_message(
            preview="first 100 chars...",
            has_more=True,
            original_size=50_000,
            file_path="/tmp/hermes-results/test123.txt",
        )
        assert msg.startswith(PERSISTED_OUTPUT_TAG)
        assert msg.endswith(PERSISTED_OUTPUT_CLOSING_TAG)
        assert "50,000 characters" in msg
        assert "/tmp/hermes-results/test123.txt" in msg
        assert "read_file" in msg
        assert "first 100 chars..." in msg
        assert "..." in msg  # has_more indicator


    def test_large_size_shows_mb(self):
        msg = _build_persisted_message(
            preview="x",
            has_more=True,
            original_size=2_000_000,
            file_path="/tmp/hermes-results/big.txt",
        )
        assert "MB" in msg


# ── maybe_persist_tool_result ─────────────────────────────────────────

class TestMaybePersistToolResult:
    def test_below_threshold_returns_unchanged(self):
        content = "small result"
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_123",
            env=None,
            threshold=50_000,
        )
        assert result == content

    def test_above_threshold_with_env_persists(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        content = "x" * 60_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_456",
            env=env,
            threshold=30_000,
        )
        assert PERSISTED_OUTPUT_TAG in result
        assert "tc_456.txt" in result
        assert len(result) < len(content)
        env.execute.assert_called_once()

    def test_persists_full_content_as_is(self):
        """Content is persisted verbatim — no JSON extraction."""
        import json
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        raw = "line1\nline2\n" * 5_000
        content = json.dumps({"output": raw, "exit_code": 0, "error": None})
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_json",
            env=env,
            threshold=30_000,
        )
        assert PERSISTED_OUTPUT_TAG in result
        # Content is delivered through stdin (no longer embedded in the
        # command string — see test_large_content_via_stdin for why).
        assert env.execute.call_args[1]["stdin_data"] == content


    def test_tool_use_id_cannot_escape_storage_dir(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        env.get_temp_dir.return_value = ""
        content = "x" * 60_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="../outside/$(whoami);x",
            env=env,
            threshold=30_000,
        )
        cmd = env.execute.call_args[0][0]
        target = cmd.split("cat > ", 1)[1].split(" <<", 1)[0]

        assert "Full output saved to: /tmp/hermes-results/outside_whoami_x_" in result
        assert "/tmp/hermes-results/../" not in result
        assert target.startswith("/tmp/hermes-results/outside_whoami_x_")
        assert "/../" not in target
        assert "$(whoami)" not in target
        assert ";" not in target


    def test_threshold_zero_forces_persist(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        content = "even short content"
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_zero",
            env=env,
            threshold=0,
        )
        # Any non-empty content with threshold=0 should be persisted
        assert PERSISTED_OUTPUT_TAG in result


# ── enforce_turn_budget ───────────────────────────────────────────────

class TestEnforceTurnBudget:
    def test_under_budget_no_changes(self):
        msgs = [
            {"role": "tool", "tool_call_id": "t1", "content": "small"},
            {"role": "tool", "tool_call_id": "t2", "content": "also small"},
        ]
        result = enforce_turn_budget(msgs, env=None, config=BudgetConfig(turn_budget=200_000))
        assert result[0]["content"] == "small"
        assert result[1]["content"] == "also small"


    def test_medium_result_regression(self):
        """6 results of 42K chars each (252K total) — each under 100K default
        threshold but aggregate exceeds 200K budget. L3 should persist."""
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        msgs = [
            {"role": "tool", "tool_call_id": f"t{i}", "content": "x" * 42_000}
            for i in range(6)
        ]
        enforce_turn_budget(msgs, env=env, config=BudgetConfig(turn_budget=200_000))
        # At least some results should be persisted to get under 200K
        persisted_count = sum(
            1 for m in msgs if PERSISTED_OUTPUT_TAG in m["content"]
        )
        assert persisted_count >= 2  # Need to shed at least ~52K


    def test_empty_messages(self):
        result = enforce_turn_budget([], env=None, config=BudgetConfig(turn_budget=200_000))
        assert result == []


# ── Per-tool threshold integration ────────────────────────────────────

class TestPerToolThresholds:
    """Verify registry wiring for per-tool thresholds."""

    def test_registry_has_get_max_result_size(self):
        from tools.registry import registry
        assert hasattr(registry, "get_max_result_size")


    def test_read_file_registry_cap_is_100k(self):
        """Regression test: read_file must have a 100_000 char registry cap (Layer 2 safety net)."""
        from tools.registry import registry
        try:
            import tools.file_tools  # noqa: F401
            val = registry.get_max_result_size("read_file")
            assert val == 100_000, (
                f"read_file registry cap must be 100_000, got {val!r}. "
                "float('inf') is not allowed — it disables the Layer 2 result-size guard."
            )
        except ImportError:
            pytest.skip("file_tools not importable in test env")

    def test_search_files_threshold(self):
        from tools.registry import registry
        try:
            import tools.file_tools  # noqa: F401
            val = registry.get_max_result_size("search_files")
            assert val == 100_000
        except ImportError:
            pytest.skip("file_tools not importable in test env")


# ── Layer 4: Tool-output compaction ──────────────────────────────────

_COMPACTION_ENABLED = {
    "enabled": True,
    "threshold": 5_000,
    "excerpt_size": 500,
}


def _patch_compaction_config(monkeypatch, cfg: dict):
    """Point the compaction config cache at ``cfg`` and clear the TTL cache."""
    import time as _time
    import tools.tool_result_storage as trs

    monkeypatch.setattr(trs, "_compaction_cache", [_time.monotonic(), cfg])


class TestCompactToolResult:
    """Feature-flag compaction: summary + excerpt + artifact_id."""

    def test_disabled_by_default_returns_unchanged(self, monkeypatch):
        _patch_compaction_config(monkeypatch, {"enabled": False})
        content = "x" * 10_000
        result = compact_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_off",
            env=None,
        )
        assert result == content

    def test_small_content_returns_unchanged(self, monkeypatch):
        _patch_compaction_config(monkeypatch, _COMPACTION_ENABLED)
        content = "small"
        result = compact_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_small",
            env=None,
        )
        assert result == content

    def test_large_content_compacted_with_artifact(self, monkeypatch):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        _patch_compaction_config(monkeypatch, _COMPACTION_ENABLED)
        content = "\n".join(f"line {i} - " + "x" * 80 for i in range(200))
        result = compact_tool_result(
            content=content,
            tool_name="search_files",
            tool_use_id="tc_big",
            env=env,
        )
        assert result.startswith(COMPACT_OUTPUT_TAG)
        assert result.endswith(COMPACT_OUTPUT_CLOSING_TAG)
        assert "Artifact: tool:search_files/tc_big" in result
        assert "Full output saved to:" in result
        assert len(result) < len(content)
        # Excerpt contains the first meaningful line
        assert "line 0" in result
        # Full content persisted to sandbox via stdin
        assert env.execute.call_args[1]["stdin_data"] == content

    def test_excerpt_respects_excerpt_size(self, monkeypatch):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        _patch_compaction_config(
            monkeypatch,
            {"enabled": True, "threshold": 100, "excerpt_size": 300},
        )
        content = "a" * 5_000
        result = compact_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_excerpt",
            env=env,
        )
        excerpt_part = result.split("Excerpt:")[1].split("</compacted-output>")[0]
        # Excerpt capped at ~300 chars + continuation marker
        assert len(excerpt_part.strip()) <= 305
        assert "..." in excerpt_part

    def test_json_summary_describes_structure(self, monkeypatch):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        _patch_compaction_config(monkeypatch, _COMPACTION_ENABLED)
        content = '{"status": "ok", "items": [1, 2, 3]}' + " " * 6_000
        result = compact_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_json",
            env=env,
        )
        assert "JSON object with keys: status, items" in result

    def test_already_persisted_not_double_wrapped(self, monkeypatch):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        _patch_compaction_config(monkeypatch, _COMPACTION_ENABLED)
        content = (
            f"{PERSISTED_OUTPUT_TAG}\nFull output saved to: /tmp/x.txt\n"
            f"{PERSISTED_OUTPUT_CLOSING_TAG}"
        )
        result = compact_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_persisted",
            env=env,
        )
        assert result == content
        env.execute.assert_not_called()

    def test_sandbox_write_failure_falls_back_to_inline(self, monkeypatch):
        env = MagicMock()
        env.execute.side_effect = Exception("sandbox down")
        _patch_compaction_config(monkeypatch, _COMPACTION_ENABLED)
        content = "x" * 10_000
        result = compact_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_nosandbox",
            env=env,
        )
        assert result.startswith(COMPACT_OUTPUT_TAG)
        assert result.endswith(COMPACT_OUTPUT_CLOSING_TAG)
        # No artifact reference when write failed — still inline excerpt
        assert "Artifact:" not in result
        assert "Excerpt:" in result
        assert len(result) < len(content)

    def test_compact_summary_empty(self):
        from tools.tool_result_storage import _compact_summary

        assert _compact_summary("") == "Empty result"

    def test_config_cache_ttl(self, monkeypatch):
        """Config cache respects TTL so per-call config reads are amortized."""
        import time as _time
        import tools.tool_result_storage as trs

        # Seed cache with a config, verify it's served without re-read
        trs._compaction_cache[:] = [_time.monotonic(), {"enabled": True}]
        cfg = trs._read_compaction_config()
        assert cfg["enabled"] is True
