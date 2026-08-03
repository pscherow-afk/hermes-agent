"""Tests for the /health endpoint (Top-P0 from Harness Audit)."""

import os
import sqlite3

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Redirect state.db path to a temp file so we control integrity.
    _state = tmp_path / "state.db"
    _conn = sqlite3.connect(str(_state))
    _conn.execute("CREATE TABLE IF NOT EXISTS _probe (id INTEGER PRIMARY KEY)")
    _conn.commit()
    _conn.close()
    monkeypatch.setattr(
        web_server, "_cfg_get", lambda key: str(_state) if key == "state_db_path" else None
    )
    return TestClient(web_server.app)


def test_health_ok_when_db_intact(client):
    """GET /health returns 200 + status ok when state.db is healthy."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["db"] == "ok"


def test_health_error_when_db_missing(client, monkeypatch):
    """GET /health returns 503 when state.db path is missing/unreadable."""
    monkeypatch.setattr(
        web_server, "_cfg_get", lambda key: "/nonexistent/path/state.db" if key == "state_db_path" else None
    )
    resp = client.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "error"
    assert body["db"] == "error"
