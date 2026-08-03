"""Tests for MCP Vector Backend."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from plugins.memory.vector.backends.mcp import MCPVectorBackend, validate_mcp_config


class TestMCPVectorBackendConfig:
    """Test MCP backend config validation."""

    def test_valid_stdio_config(self):
        """Test valid stdio server config."""
        config = {
            "servers": [
                {
                    "name": "qdrant-mcp",
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-qdrant"],
                    "tools": {
                        "upsert": "qdrant_upsert",
                        "search": "qdrant_search",
                        "delete": "qdrant_delete",
                        "stats": "qdrant_stats",
                    },
                }
            ],
            "embedding_tool": "embed_text",
        }
        validated = validate_mcp_config(config)
        assert len(validated["servers"]) == 1
        assert validated["servers"][0]["name"] == "qdrant-mcp"
        assert validated["servers"][0]["transport"] == "stdio"

    def test_valid_http_config(self):
        """Test valid HTTP server config."""
        config = {
            "servers": [
                {
                    "name": "remote-qdrant",
                    "transport": "http",
                    "url": "http://localhost:8000/mcp",
                    "headers": {"Authorization": "Bearer token"},
                }
            ]
        }
        validated = validate_mcp_config(config)
        assert len(validated["servers"]) == 1
        assert validated["servers"][0]["transport"] == "http"
        assert validated["servers"][0]["url"] == "http://localhost:8000/mcp"

    def test_missing_servers_raises(self):
        """Test that missing servers raises error."""
        with pytest.raises(ValueError, match="At least one MCP server must be configured"):
            validate_mcp_config({"servers": []})

    def test_auto_generated_name(self):
        """Test that missing name is auto-generated."""
        config = {
            "servers": [
                {"transport": "stdio", "command": "npx"}
            ]
        }
        validated = validate_mcp_config(config)
        assert validated["servers"][0]["name"] == "mcp-server-0"

    def test_invalid_transport_requires_url(self):
        """Test that non-stdio transport requires URL."""
        config = {
            "servers": [
                {"name": "test", "transport": "invalid"}
            ]
        }
        with pytest.raises(ValueError, match="HTTP transport requires 'url'"):
            validate_mcp_config(config)

    def test_stdio_missing_command_raises(self):
        """Test that stdio transport without command raises error."""
        config = {
            "servers": [
                {"name": "test", "transport": "stdio"}
            ]
        }
        with pytest.raises(ValueError, match="stdio transport requires 'command'"):
            validate_mcp_config(config)

    def test_http_missing_url_raises(self):
        """Test that HTTP transport without URL raises error."""
        config = {
            "servers": [
                {"name": "test", "transport": "http"}
            ]
        }
        with pytest.raises(ValueError, match="HTTP transport requires 'url'"):
            validate_mcp_config(config)

    def test_default_tools_applied(self):
        """Test that default tool names are applied."""
        config = {
            "servers": [
                {"name": "test", "transport": "stdio", "command": "npx"}
            ]
        }
        validated = validate_mcp_config(config)
        server = validated["servers"][0]
        assert server["tools"]["upsert"] == "qdrant_upsert"
        assert server["tools"]["search"] == "qdrant_search"
        assert server["tools"]["delete"] == "qdrant_delete"
        assert server["tools"]["stats"] == "qdrant_stats"
        assert server["tools"]["embedding"] == "embed_text"

    def test_duplicate_names_raises(self):
        """Test that duplicate server names raise error."""
        config = {
            "servers": [
                {"name": "test", "transport": "stdio", "command": "npx"},
                {"name": "test", "transport": "stdio", "command": "npx"},
            ]
        }
        with pytest.raises(ValueError, match="Duplicate server name"):
            validate_mcp_config(config)


class TestMCPVectorBackend:
    """Test MCP Vector Backend behavior."""

    def test_init_with_valid_config(self):
        """Test backend initialization with valid config."""
        config = {
            "servers": [
                {
                    "name": "test-server",
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "test-server"],
                }
            ]
        }
        backend = MCPVectorBackend(config)
        assert len(backend.servers) == 1
        assert backend.servers[0]["name"] == "test-server"

    def test_upsert_calls_mcp_tool(self):
        """Test upsert calls MCP tool with correct format."""
        config = {
            "servers": [
                {
                    "name": "test-server",
                    "transport": "stdio",
                    "command": "npx",
                    "tools": {"upsert": "custom_upsert"},
                }
            ]
        }
        backend = MCPVectorBackend(config)

        # Mock the internal MCP call
        with patch.object(backend, '_verify_connections'):
            with patch.object(backend, '_call_mcp_tool') as mock_call:
                backend.upsert(
                    ids=["id1", "id2"],
                    vectors=[[0.1, 0.2], [0.3, 0.4]],
                    metadata=[{"content": "text1"}, {"content": "text2"}]
                )
                mock_call.assert_called_once()
                call_args = mock_call.call_args
                assert call_args[0][0] == "test-server"  # server_name
                assert call_args[0][1] == "custom_upsert"  # tool_name
                assert "points" in call_args[0][2]  # arguments

    def test_search_calls_mcp_tool(self):
        """Test search calls MCP tool with correct format."""
        config = {
            "servers": [
                {
                    "name": "test-server",
                    "transport": "stdio",
                    "command": "npx",
                    "tools": {"search": "custom_search"},
                }
            ]
        }
        backend = MCPVectorBackend(config)

        with patch.object(backend, '_verify_connections'):
            with patch.object(backend, '_call_mcp_tool') as mock_call:
                mock_call.return_value = [
                    {"id": "id1", "score": 0.9, "payload": {"content": "result1", "meta": "value"}}
                ]
                results = backend.search([0.1, 0.2], top_k=10, threshold=0.7)
                mock_call.assert_called_once()
                call_args = mock_call.call_args
                assert call_args[0][0] == "test-server"
                assert call_args[0][1] == "custom_search"
                args = call_args[0][2]
                assert args["vector"] == [0.1, 0.2]
                assert args["limit"] == 10
                assert args["score_threshold"] == 0.7
                assert len(results) == 1
                assert results[0]["id"] == "id1"
                assert results[0]["content"] == "result1"

    def test_search_handles_wrapped_response(self):
        """Test search handles wrapped response format."""
        config = {
            "servers": [
                {"name": "test-server", "transport": "stdio", "command": "npx"}
            ]
        }
        backend = MCPVectorBackend(config)

        with patch.object(backend, '_verify_connections'):
            with patch.object(backend, '_call_mcp_tool') as mock_call:
                mock_call.return_value = {
                    "results": [
                        {"id": "id1", "score": 0.9, "payload": {"content": "result1"}}
                    ]
                }
                results = backend.search([0.1, 0.2], top_k=10, threshold=0.7)
                assert len(results) == 1
                assert results[0]["id"] == "id1"

    def test_delete_calls_mcp_tool(self):
        """Test delete calls MCP tool."""
        config = {
            "servers": [
                {
                    "name": "test-server",
                    "transport": "stdio",
                    "command": "npx",
                    "tools": {"delete": "custom_delete"},
                }
            ]
        }
        backend = MCPVectorBackend(config)

        with patch.object(backend, '_verify_connections'):
            with patch.object(backend, '_call_mcp_tool') as mock_call:
                backend.delete(["id1", "id2"])
                mock_call.assert_called_once()
                call_args = mock_call.call_args
                assert call_args[0][0] == "test-server"
                assert call_args[0][1] == "custom_delete"
                assert call_args[0][2] == {"ids": ["id1", "id2"]}

    def test_get_stats_returns_error_on_failure(self):
        """Test get_stats returns error info on failure."""
        config = {
            "servers": [
                {"name": "test-server", "transport": "stdio", "command": "npx"}
            ]
        }
        backend = MCPVectorBackend(config)

        with patch.object(backend, '_verify_connections'):
            with patch.object(backend, '_call_mcp_tool') as mock_call:
                mock_call.side_effect = RuntimeError("Connection failed")
                stats = backend.get_stats()
                assert stats["backend"] == "mcp"
                assert "error" in stats
                assert "Connection failed" in stats["error"]


class TestMCPBackendRetryLogic:
    """Test retry logic in MCP backend."""

    @patch("time.sleep")
    def test_retry_on_failure(self, mock_sleep):
        """Test that failed calls are retried with exponential backoff."""
        config = {
            "servers": [{"name": "test", "transport": "stdio", "command": "npx"}],
            "max_retries": 3,
            "retry_base_delay": 0.1,
            "retry_max_delay": 1.0,
        }
        backend = MCPVectorBackend(config)

        call_count = 0

        def mock_call_fail_twice(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Temporary failure")
            return {"id": "id1", "score": 0.9, "payload": {"content": "ok"}}

        with patch.object(backend, '_verify_connections'):
            with patch.object(backend, '_call_mcp_tool', side_effect=mock_call_fail_twice):
                # Test the retry logic works by calling search
                results = backend.search([0.1, 0.2], top_k=10, threshold=0.7)
                assert call_count == 3
                assert results[0]["content"] == "ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])