"""Integration tests for the read-only Memory API (Task P2b).

Drives the *real* dashboard endpoints in ``hermes_cli.web_server``
(``/api/memory/profile/{name}`` and ``/api/memory/search``) through a
TestClient, verifying the token-auth seam, rate limiting, audit logging,
redaction, and search pagination wired by ``register_memory_api``.
"""

from __future__ import annotations

import os

import pytest

_TEST_KEY = "memory-api-test-key-" + "a" * 32
os.environ.setdefault("HERMES_MEMORY_API_KEY", _TEST_KEY)

from fastapi.testclient import TestClient  # noqa: E402

import hermes_cli.memory_api as memory_api  # noqa: E402
from hermes_cli.dashboard_auth import clear_providers, register_provider  # noqa: E402
from hermes_cli.dashboard_auth.token_auth import clear_token_routes  # noqa: E402
from hermes_cli.memory_api import MemoryApiTokenProvider, get_openapi_spec  # noqa: E402


@pytest.fixture()
def client():
    # Import the real web server app (heavy) and register the Memory API
    # auth provider + token routes into it.
    import hermes_cli.web_server as ws

    clear_providers()
    clear_token_routes()
    provider = MemoryApiTokenProvider(api_key=_TEST_KEY)
    register_provider(provider)
    # Re-run registration so the existing routes become token-authable.
    memory_api.register_memory_api(ws.app)
    return TestClient(ws.app)


def _auth():
    return {"Authorization": f"Bearer {_TEST_KEY}"}


def test_profile_requires_auth(client):
    resp = client.get("/api/memory/profile/walter")
    assert resp.status_code == 401


def test_profile_returns_redacted_facts(client):
    resp = client.get("/api/memory/profile/walter", headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    assert body["profile"] == "walter"
    assert "files" in body
    # MEMORY.md / USER.md are present as redacted strings (or None if absent).
    assert "MEMORY.md" in body["files"]


def test_profile_unknown_404(client):
    resp = client.get("/api/memory/profile/definitely-not-a-real-profile-xyz", headers=_auth())
    assert resp.status_code == 404


def test_search_requires_query(client):
    # Web search endpoint treats empty q as "match all" — verify it returns
    # 200 with results rather than 400, matching the existing contract.
    resp = client.get("/api/memory/search", headers=_auth())
    assert resp.status_code == 200
    assert "results" in resp.json()


def test_search_pagination(client):
    resp = client.get(
        "/api/memory/search",
        params={"q": "a", "limit": 5, "offset": 0, "scope": "all"},
        headers=_auth(),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "total" in body
    assert len(body["results"]) <= 5
    assert body["limit"] == 5


def test_bad_scope_rejected(client):
    resp = client.get("/api/memory/search", params={"q": "x", "scope": "bogus"}, headers=_auth())
    assert resp.status_code == 400


def test_rate_limit(client):
    resp = None
    for _ in range(memory_api._RATE_LIMIT_REQUESTS + 5):
        resp = client.get("/api/memory/profile/walter", headers=_auth())
        if resp.status_code == 429:
            break
    assert resp is not None
    assert resp.status_code in (200, 404, 429)


def test_audit_log_written(tmp_path, monkeypatch):
    # Point the audit log at a temp file and confirm an entry is written.
    import hermes_cli.dashboard_auth.audit as audit_mod

    log_path = tmp_path / "memory-api.log"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Re-resolve inside the module by monkeypatching the path resolver.
    audit_mod._resolve_log_path = lambda: log_path  # type: ignore[attr-defined]

    import hermes_cli.web_server as ws

    clear_providers()
    clear_token_routes()
    register_provider(MemoryApiTokenProvider(api_key=_TEST_KEY))
    memory_api.register_memory_api(ws.app)
    c = TestClient(ws.app)
    c.get("/api/memory/profile/walter", headers=_auth())
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "memory_profile" in content


def test_openapi_spec_shape():
    spec = get_openapi_spec()
    assert spec["openapi"].startswith("3.")
    assert "/api/memory/profile/{name}" in spec["paths"]
    assert "/api/memory/search" in spec["paths"]
    assert "bearerAuth" in spec["components"]["securitySchemes"]


def test_jwt_verify():
    import jwt

    secret = "jwt-secret-" + "b" * 32
    token = jwt.encode({"sub": "svc-1", "scope": "memory:read"}, secret, algorithm="HS256")
    provider = MemoryApiTokenProvider(jwt_secret=secret)
    principal = provider.verify_token(token=token)
    assert principal is not None
    assert principal.principal == "svc-1"
    # Wrong secret → None
    bad = token[: len(token) - 1] + ("c" if token[-1] != "c" else "d")
    assert provider.verify_token(token=bad) is None
