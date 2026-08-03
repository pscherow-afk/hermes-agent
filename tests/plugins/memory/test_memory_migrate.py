"""Unit tests for the JSON -> vector store migration tool.

Covers:
1. Document collection from MEMORY.md / USER.md (§ delimiter + dedup).
2. JSON memory ingestion (dict form, list form, object form, garbage skipped).
3. Deterministic UUID5 IDs (idempotent re-runs).
4. Embedding + batched upsert into a backend (mocked), incl. dry-run.
5. Embedding/upsert failure isolation (one bad batch doesn't abort the run).
6. CLI arg parsing + --mcp-servers @file resolution.
7. Unknown-backend / invalid-source validation.

Run from the hermes-agent repo root:
    pytest tests/plugins/memory/test_memory_migrate.py
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from plugins.memory.vector import EmbeddingProvider
from plugins.memory.vector.cli import (
    MigrationReport,
    MemoryMigrator,
    _resolve_mcp_servers,
    build_config,
    parse_args,
)


class _FakeEmbed(EmbeddingProvider):
    """Deterministic embedder: returns a fixed-dim vector per text length."""

    def __init__(self, dim: int = 4):
        super().__init__("fake", dim)

    def embed(self, texts):
        return [[float(len(t)) % 7, 0.1, 0.2, 0.3][: self.dim] for t in texts]


def _make_home(tmp_path: Path, memory: str = "", user: str = "") -> Path:
    mem = tmp_path / "memories"
    mem.mkdir(parents=True, exist_ok=True)
    (mem / "MEMORY.md").write_text(memory, encoding="utf-8")
    (mem / "USER.md").write_text(user, encoding="utf-8")
    return tmp_path


def _mock_backend(monkeypatch, stats=None):
    backend = MagicMock()
    backend.get_stats.return_value = stats or {
        "backend": "chroma",
        "collection": "hermes_memory",
        "count": 0,
    }
    monkeypatch.setattr(
        "plugins.memory.vector.cli.create_backend", lambda t, c: backend
    )
    return backend


# ===========================================================================
# 1. Markdown collection
# ===========================================================================


class TestMarkdownCollection:
    def test_reads_both_stores(self, tmp_path, monkeypatch):
        home = _make_home(
            tmp_path,
            memory="note one\n§\nnote two",
            user="user note",
        )
        _mock_backend(monkeypatch)
        m = MemoryMigrator(home, {"backend": "chroma"}, _FakeEmbed())
        docs = m.collect_documents()
        assert len(docs) == 3
        assert [d.metadata["source"] for d in docs] == ["memory", "memory", "user"]

    def test_ignores_missing_files(self, tmp_path, monkeypatch):
        home = tmp_path / "memories"
        home.mkdir()
        _mock_backend(monkeypatch)
        m = MemoryMigrator(home, {"backend": "chroma"}, _FakeEmbed())
        assert m.collect_documents() == []

    def test_deduplicates_entries(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path, memory="dup\n§\ndup\n§\nunique")
        _mock_backend(monkeypatch)
        m = MemoryMigrator(home, {"backend": "chroma"}, _FakeEmbed())
        docs = m.collect_documents()
        assert len(docs) == 2
        assert [d.content for d in docs] == ["dup", "unique"]

    def test_multiline_entries_preserved(self, tmp_path, monkeypatch):
        # A lone "§" inside an entry must NOT split it.
        home = _make_home(tmp_path, memory="line1 § inside\nstill same entry\n§\nentry two")
        _mock_backend(monkeypatch)
        m = MemoryMigrator(home, {"backend": "chroma"}, _FakeEmbed())
        docs = m.collect_documents()
        assert len(docs) == 2
        assert "§ inside" in docs[0].content

    def test_source_filter_memory_only(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path, memory="m", user="u")
        _mock_backend(monkeypatch)
        m = MemoryMigrator(
            home, {"backend": "chroma"}, _FakeEmbed(), source="memory"
        )
        docs = m.collect_documents()
        assert [d.metadata["source"] for d in docs] == ["memory"]


# ===========================================================================
# 2. JSON collection
# ===========================================================================


class TestJsonCollection:
    def test_dict_form(self, tmp_path, monkeypatch):
        home = tmp_path
        jp = home / "m.json"
        jp.write_text(
            json.dumps(
                {
                    "memory": ["json mem a", "json mem b"],
                    "user": ["json user a"],
                    "garbage": "ignored-scalar",
                }
            ),
            encoding="utf-8",
        )
        _mock_backend(monkeypatch)
        m = MemoryMigrator(
            home, {"backend": "chroma"}, _FakeEmbed(), json_path=str(jp)
        )
        docs = m.collect_documents()
        assert len(docs) == 3
        assert sorted(d.metadata["source"] for d in docs) == ["memory", "memory", "user"]

    def test_list_of_strings(self, tmp_path, monkeypatch):
        jp = tmp_path / "m.json"
        jp.write_text(json.dumps(["a", "b", "c"]), encoding="utf-8")
        _mock_backend(monkeypatch)
        m = MemoryMigrator(
            tmp_path, {"backend": "chroma"}, _FakeEmbed(), json_path=str(jp)
        )
        docs = m.collect_documents()
        assert [d.content for d in docs] == ["a", "b", "c"]
        assert all(d.metadata["source"] == "json" for d in docs)

    def test_list_of_objects_with_meta(self, tmp_path, monkeypatch):
        jp = tmp_path / "m.json"
        jp.write_text(
            json.dumps([{"content": "obj", "tag": "x"}, "plain"]),
            encoding="utf-8",
        )
        _mock_backend(monkeypatch)
        m = MemoryMigrator(
            tmp_path, {"backend": "chroma"}, _FakeEmbed(), json_path=str(jp)
        )
        docs = m.collect_documents()
        assert docs[0].content == "obj"
        assert docs[0].metadata["tag"] == "x"
        assert docs[1].content == "plain"

    def test_missing_json_raises(self, tmp_path, monkeypatch):
        _mock_backend(monkeypatch)
        m = MemoryMigrator(
            tmp_path,
            {"backend": "chroma"},
            _FakeEmbed(),
            json_path=str(tmp_path / "nope.json"),
        )
        with pytest.raises(FileNotFoundError):
            m.collect_documents()

    def test_bad_json_raises(self, tmp_path, monkeypatch):
        jp = tmp_path / "m.json"
        jp.write_text("{not valid json", encoding="utf-8")
        _mock_backend(monkeypatch)
        m = MemoryMigrator(
            tmp_path, {"backend": "chroma"}, _FakeEmbed(), json_path=str(jp)
        )
        with pytest.raises(ValueError):
            m.collect_documents()


# ===========================================================================
# 3. Deterministic IDs
# ===========================================================================


class TestDeterministicIds:
    def test_same_input_same_id(self):
        a = MemoryMigrator._make_id("memory", "hello")
        b = MemoryMigrator._make_id("memory", "hello")
        assert a == b

    def test_different_content_differs(self):
        a = MemoryMigrator._make_id("memory", "hello")
        b = MemoryMigrator._make_id("memory", "world")
        assert a != b

    def test_source_affects_id(self):
        a = MemoryMigrator._make_id("memory", "same")
        b = MemoryMigrator._make_id("user", "same")
        assert a != b

    def test_ids_are_uuid_format(self):
        import uuid

        for s in ("memory", "user", "json"):
            assert uuid.UUID(MemoryMigrator._make_id(s, "x"))


# ===========================================================================
# 4. Execution + embedding + upsert
# ===========================================================================


class TestExecution:
    def test_full_run_upserts(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path, memory="a\n§\nb", user="c")
        backend = _mock_backend(monkeypatch)
        m = MemoryMigrator(home, {"backend": "chroma"}, _FakeEmbed(), source="both")
        rep = m.run()
        assert rep.total == 3
        assert rep.embedded == 3
        assert rep.upserted == 3
        assert backend.upsert.call_count == 1
        # metadatas carry content + source
        ids, vecs, metas = backend.upsert.call_args[0]
        assert all("content" in mt for mt in metas)
        backend.close.assert_called_once()

    def test_batching(self, tmp_path, monkeypatch):
        memory = "\n§\n".join(f"entry-{i}" for i in range(5))
        home = _make_home(tmp_path, memory=memory)
        backend = _mock_backend(monkeypatch)
        m = MemoryMigrator(
            home, {"backend": "chroma"}, _FakeEmbed(), batch_size=2
        )
        rep = m.run()
        assert rep.total == 5
        assert backend.upsert.call_count == 3  # ceil(5/2)
        assert rep.batches == 3

    def test_dry_run_does_not_write(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path, memory="a\n§\nb", user="c")
        backend = _mock_backend(monkeypatch)
        m = MemoryMigrator(
            home, {"backend": "chroma"}, _FakeEmbed(), source="both", dry_run=True
        )
        rep = m.run()
        assert rep.dry_run is True
        assert rep.embedded == 3
        assert rep.upserted == 0
        backend.upsert.assert_not_called()
        backend.initialize.assert_not_called()

    def test_empty_memory(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path, memory="", user="")
        backend = _mock_backend(monkeypatch)
        rep = MemoryMigrator(home, {"backend": "chroma"}, _FakeEmbed()).run()
        assert rep.total == 0
        backend.upsert.assert_not_called()

    def test_report_to_dict(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path, memory="a")
        _mock_backend(monkeypatch)
        rep = MemoryMigrator(home, {"backend": "chroma"}, _FakeEmbed()).run()
        d = rep.to_dict()
        assert d["total"] == 1
        assert d["backend"] == "chroma"
        assert "errors" in d


# ===========================================================================
# 5. Failure isolation
# ===========================================================================


class TestFailureIsolation:
    def test_embedding_error_skips_batch(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path, memory="a\n§\nb")
        backend = _mock_backend(monkeypatch)

        class _BoomEmbed(EmbeddingProvider):
            def __init__(self):
                super().__init__("boom", 4)

            def embed(self, texts):
                raise RuntimeError("embedding service down")

        rep = MemoryMigrator(home, {"backend": "chroma"}, _BoomEmbed()).run()
        assert rep.embedded == 0
        assert rep.skipped == 2
        assert len(rep.errors) == 1
        backend.upsert.assert_not_called()

    def test_upsert_error_does_not_abort(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path, memory="a\n§\nb\n§\nc", user="d")
        backend = _mock_backend(monkeypatch)

        # First upsert call fails, rest succeed.
        def _upsert(ids, vecs, metas):
            if metas and metas[0].get("source") == "memory":
                raise RuntimeError("transient db error")
            return None

        backend.upsert.side_effect = _upsert
        rep = MemoryMigrator(home, {"backend": "chroma"}, _FakeEmbed()).run()
        # 4 docs total; memory batch(es) fail, user ok.
        assert rep.skipped >= 1
        assert rep.embedded == 4


# ===========================================================================
# 6. Validation + CLI
# ===========================================================================


class TestValidationAndCli:
    def test_unknown_backend(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown backend"):
            MemoryMigrator(tmp_path, {"backend": "redis"}, _FakeEmbed())

    def test_invalid_source(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid source"):
            MemoryMigrator(tmp_path, {"backend": "chroma"}, _FakeEmbed(), source="bogus")

    def test_embedding_dim_injected(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path, memory="a")
        backend = _mock_backend(monkeypatch)
        m = MemoryMigrator(home, {"backend": "chroma"}, _FakeEmbed(dim=7))
        assert m.config["chroma"]["embedding_dim"] == 7
        m.run()
        # chroma backend got the dim via sub-config
        assert m.config["chroma"]["embedding_dim"] == 7

    def test_build_config_chroma(self):
        args = parse_args(
            [
                "--backend",
                "chroma",
                "--chroma-dir",
                "/tmp/c",
                "--chroma-collection",
                "col",
            ]
        )
        cfg = build_config(args, 384)
        assert cfg["backend"] == "chroma"
        assert cfg["chroma"]["persist_directory"] == "/tmp/c"
        assert cfg["chroma"]["embedding_dim"] == 384

    def test_build_config_qdrant_pgvector_mcp(self):
        aq = parse_args(["--backend", "qdrant", "--qdrant-url", "http://q:6333"])
        cq = build_config(aq, 1536)
        assert cq["qdrant"]["url"] == "http://q:6333"
        assert cq["qdrant"]["embedding_dim"] == 1536

        ap = parse_args(
            ["--backend", "pgvector", "--pg-conn-string", "postgresql://x"]
        )
        cp = build_config(ap, 1536)
        assert cp["pgvector"]["connection_string"] == "postgresql://x"

        am = parse_args(
            ["--backend", "mcp", "--mcp-servers", '[{"name":"s1"}]']
        )
        cm = build_config(am, 8)
        assert cm["mcp"]["servers"] == [{"name": "s1"}]

    def test_mcp_servers_at_file(self, tmp_path):
        fp = tmp_path / "srv.json"
        fp.write_text('[{"name": "s1", "transport": "stdio", "command": "x"}]',
                      encoding="utf-8")
        resolved = _resolve_mcp_servers("@" + str(fp))
        assert json.loads(resolved) == [
            {"name": "s1", "transport": "stdio", "command": "x"}
        ]

    def test_mcp_servers_inline_and_none(self):
        assert _resolve_mcp_servers(None) is None
        assert _resolve_mcp_servers("[1,2]") == "[1,2]"


# ===========================================================================
# 7. MigrationReport defaults
# ===========================================================================


class TestMigrationReport:
    def test_defaults(self):
        r = MigrationReport()
        assert r.total == 0
        assert r.errors == []
        assert r.by_source == {}
        d = r.to_dict()
        assert d["total"] == 0
        assert d["errors"] == []
