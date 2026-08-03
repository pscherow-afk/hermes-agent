"""Read-only Memory API helpers (Task P2b).

Task P2b extends the *existing* dashboard memory endpoints
(``GET /api/memory/profile/{name}`` and ``GET /api/memory/search`` in
``hermes_cli/web_server.py``) with production-grade access control for
external/analytics consumers:

  * **Auth** — a registered ``DashboardAuthProvider`` (``MemoryApiTokenProvider``)
    verifies either a static API key (``Authorization: Bearer <key>``) or a
    signed JWT (HS256). It reuses the dashboard ``token_auth`` seam, so the
    existing routes become token-authable without a rewrite.
  * **Rate limiting** — per-IP in-memory token bucket.
  * **Audit logging** — every request (success/failure/rate-limited) is
    written to ``$HERMES_HOME/logs/memory-api.log``.
  * **Redaction** — facts are redacted with ``force=True`` regardless of the
    global redaction toggle.

This module is *import-only*: the endpoints themselves live in web_server.py.
``register_memory_api(app)`` wires the auth provider + token routes + rate
limiter into the running FastAPI app. Fail-closed — if no credential is set,
the provider is not registered and the token-guarded routes 401.

Configuration (env vars — the secrets-only rule):
  * ``HERMES_MEMORY_API_KEY``        — static shared secret (>= 256-bit).
  * ``HERMES_MEMORY_API_JWT_SECRET`` — HMAC secret to verify signed JWTs
    (``sub`` → principal, optional ``scope`` claim restricts access).
"""

from __future__ import annotations

import hmac
import logging
import math
import os
import threading
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

from hermes_cli.dashboard_auth import (
    DashboardAuthProvider,
    LoginStart,
    Session,
    TokenPrincipal,
)
from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
from hermes_cli.dashboard_auth.token_auth import register_token_route

logger = logging.getLogger(__name__)

# Existing dashboard memory endpoints (web_server.py) we extend with auth.
PROFILE_ROUTE = "/api/memory/profile/{name}"
SEARCH_ROUTE = "/api/memory/search"

# Entropy bar for the static API key (mirrors dashboard-auth-drain).
_DEFAULT_MIN_KEY_CHARS = 32
_MIN_DISTINCT_CHARS = 12
_MIN_SHANNON_BITS = 64.0

# Default rate-limit: 60 requests / 60s per IP (in-memory token bucket).
_RATE_LIMIT_REQUESTS = int(os.environ.get("HERMES_MEMORY_API_RL_REQUESTS", "60"))
_RATE_LIMIT_WINDOW = int(os.environ.get("HERMES_MEMORY_API_RL_WINDOW", "60"))


def _assess_key_strength(secret: str, min_chars: int = _DEFAULT_MIN_KEY_CHARS) -> Optional[str]:
    """Return a rejection reason if ``secret`` is too weak, else ``None``."""
    if not secret:
        return "empty"
    if len(secret) < min_chars:
        return f"too short: {len(secret)} chars (need >= {min_chars})"
    distinct = len(set(secret))
    if distinct < _MIN_DISTINCT_CHARS:
        return f"only {distinct} distinct chars (need >= {_MIN_DISTINCT_CHARS})"
    bits = -sum((c / len(secret)) * math.log2(c / len(secret)) for c in Counter(secret).values()) * len(secret)
    if bits < _MIN_SHANNON_BITS:
        return f"entropy too low: {bits:.0f} bits (need >= {_MIN_SHANNON_BITS})"
    return None


# ---------------------------------------------------------------------------
# Auth provider (API key + JWT) — reuses the token_auth seam
# ---------------------------------------------------------------------------


def _verify_jwt(token: str, secret: str) -> Optional[TokenPrincipal]:
    """Verify an HMAC-signed JWT; return its principal or ``None``."""
    try:
        import jwt

        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except Exception:
        return None
    sub = payload.get("sub") or "memory-api"
    scopes = payload.get("scope")
    if isinstance(scopes, str):
        scopes = tuple(scopes.split())
    elif isinstance(scopes, (list, tuple)):
        scopes = tuple(str(s) for s in scopes)
    else:
        scopes = ()
    return TokenPrincipal(principal=str(sub), provider="memory-api", scopes=scopes)


class MemoryApiTokenProvider(DashboardAuthProvider):
    """Non-interactive API-key + JWT auth provider for the Memory API.

    Implements ONLY the token capability (``supports_token = True`` +
    ``verify_token``); the interactive ABC methods are no-ops so this
    provider stacks harmlessly in the cookie-session loop.
    """

    name = "memory-api"
    display_name = "Memory API (service credential)"
    supports_token = True
    supports_session = False

    def __init__(self, *, api_key: Optional[str] = None, jwt_secret: Optional[str] = None) -> None:
        # Fail-closed: require at least one strong credential.
        if not api_key and not jwt_secret:
            raise ValueError("memory-api: no api_key or jwt_secret provided")
        if api_key is not None:
            reason = _assess_key_strength(api_key)
            if reason is not None:
                raise ValueError(f"memory-api api_key rejected: {reason}")
        self._api_key = api_key
        self._jwt_secret = jwt_secret

    def verify_token(self, *, token: str) -> Optional[TokenPrincipal]:
        if not token:
            return None
        if self._api_key is not None:
            if hmac.compare_digest(token.encode("utf-8"), self._api_key.encode("utf-8")):
                return TokenPrincipal(
                    principal="memory-api", provider=self.name, scopes=("memory:read",)
                )
        if self._jwt_secret is not None:
            principal = _verify_jwt(token, self._jwt_secret)
            if principal is not None:
                return principal
        return None

    # ---- interactive methods: unsupported (service credential only) --------

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        raise NotImplementedError("MemoryApiTokenProvider is non-interactive.")

    def complete_login(self, *, code: str, state: str, code_verifier: str, redirect_uri: str) -> Session:
        raise NotImplementedError("MemoryApiTokenProvider is non-interactive.")

    def verify_session(self, *, access_token: str) -> Optional[Session]:
        return None

    def refresh_session(self, *, refresh_token: str) -> Session:
        raise NotImplementedError("MemoryApiTokenProvider is non-interactive.")

    def revoke_session(self, *, refresh_token: str) -> None:
        return None


# ---------------------------------------------------------------------------
# Rate limiting (in-memory per-IP token bucket)
# ---------------------------------------------------------------------------

_rate_lock = threading.Lock()
_rate_buckets: Dict[str, List[float]] = {}


def rate_limit_allowed(client_ip: str) -> bool:
    """Token-bucket rate limiter. True if the request is allowed."""
    if _RATE_LIMIT_REQUESTS <= 0:
        return True
    now = time.monotonic()
    window_start = now - _RATE_LIMIT_WINDOW
    with _rate_lock:
        hits = _rate_buckets.get(client_ip)
        if hits is None:
            hits = []
            _rate_buckets[client_ip] = hits
        hits[:] = [t for t in hits if t > window_start]
        if len(hits) >= _RATE_LIMIT_REQUESTS:
            return False
        hits.append(now)
        return True


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------


def audit_event(event: AuditEvent, *, ip: str, principal: str, **fields: object) -> None:
    """Append one memory-api audit entry (never raises).

    NOTE: do not pass ``event=`` inside ``**fields`` — ``event`` is the first
    positional argument. Pass event-type tags under a different key (e.g.
    ``event_type=``) if needed.
    """
    try:
        audit_log(event, ip=ip, principal=principal, **fields)
    except Exception as exc:  # noqa: BLE001
        logger.debug("memory-api: audit_log failed: %s", exc)


def client_ip_from(request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else ""


def principal_from(request) -> str:
    principal = getattr(request.state, "token_principal", None)
    if principal is not None:
        return getattr(principal, "principal", "memory-api")
    return "memory-api"


# ---------------------------------------------------------------------------
# Registration (called from web_server)
# ---------------------------------------------------------------------------

_LAST_SKIP_REASON = ""


def _load_credentials() -> Tuple[Optional[str], Optional[str]]:
    api_key = os.environ.get("HERMES_MEMORY_API_KEY", "").strip() or None
    jwt_secret = os.environ.get("HERMES_MEMORY_API_JWT_SECRET", "").strip() or None
    return api_key, jwt_secret


def register_memory_api(app) -> None:
    """Register the Memory API auth provider + token routes.

    Idempotent-ish: safe to call once at web-server startup. Fail-closed —
    if no credential is configured the provider is skipped and the
    token-guarded routes (already defined in web_server.py) reject all
    callers with 401.
    """
    global _LAST_SKIP_REASON
    _LAST_SKIP_REASON = ""

    # Opt the existing memory endpoints into the generic token-auth seam.
    register_token_route(PROFILE_ROUTE)
    register_token_route(SEARCH_ROUTE)

    api_key, jwt_secret = _load_credentials()
    if not api_key and not jwt_secret:
        _LAST_SKIP_REASON = (
            "HERMES_MEMORY_API_KEY / HERMES_MEMORY_API_JWT_SECRET not set. "
            "Memory API endpoints are token-authable but reject all callers "
            "(401) until a credential is configured."
        )
        logger.warning("memory-api: %s", _LAST_SKIP_REASON)
        return

    try:
        provider = MemoryApiTokenProvider(api_key=api_key, jwt_secret=jwt_secret)
    except ValueError as exc:
        _LAST_SKIP_REASON = f"MemoryApiTokenProvider rejected: {exc}"
        logger.warning("memory-api: %s", _LAST_SKIP_REASON)
        return

    try:
        from hermes_cli.dashboard_auth import register_provider

        register_provider(provider)
        logger.info(
            "memory-api: registered token auth provider (routes=%s)",
            [PROFILE_ROUTE, SEARCH_ROUTE],
        )
    except Exception as exc:  # noqa: BLE001
        _LAST_SKIP_REASON = f"provider registration failed: {exc}"
        logger.warning("memory-api: %s", _LAST_SKIP_REASON)


def get_openapi_spec() -> dict:
    """Return an OpenAPI fragment describing the Memory API endpoints."""
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "Hermes Read-only Memory API",
            "version": "1.0.0",
            "description": (
                "Read-only access to per-profile memory facts (MEMORY.md / USER.md) "
                "with redaction, search+pagination, rate limiting and audit logging. "
                "Auth: Bearer API key or JWT (HS256)."
            ),
        },
        "components": {
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "description": "Static API key or HS256 JWT.",
                }
            }
        },
        "security": [{"bearerAuth": []}],
        "paths": {
            "/api/memory/profile/{name}": {
                "get": {
                    "summary": "Get memory facts for a profile",
                    "parameters": [
                        {
                            "name": "name",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "Profile name (e.g. 'walter', 'default').",
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Redacted memory + user facts.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "profile": {"type": "string"},
                                            "files": {"type": "object"},
                                        },
                                    }
                                }
                            },
                        },
                        "401": {"description": "Unauthorized"},
                        "404": {"description": "Profile not found"},
                        "429": {"description": "Rate limited"},
                    },
                }
            },
            "/api/memory/search": {
                "get": {
                    "summary": "Search memory facts across profiles",
                    "parameters": [
                        {"name": "q", "in": "query", "required": True, "schema": {"type": "string"}},
                        {"name": "limit", "in": "query", "required": False, "schema": {"type": "integer", "default": 20}},
                        {"name": "offset", "in": "query", "required": False, "schema": {"type": "integer", "default": 0}},
                        {
                            "name": "scope",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string", "enum": ["all", "memory", "user"], "default": "all"},
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Search results.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "results": {"type": "array"},
                                            "count": {"type": "integer"},
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Bad request"},
                        "401": {"description": "Unauthorized"},
                        "429": {"description": "Rate limited"},
                    },
                }
            },
        },
    }
