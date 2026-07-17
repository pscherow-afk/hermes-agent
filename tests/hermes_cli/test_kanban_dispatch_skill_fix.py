"""Tests for the kanban dispatcher skill-crash fix (classes #1/#2/#3).

Covers the three defect classes documented in ADR-016 / the
`t_71eedffa` task:

* Class #1 — Pre-spawn validation crash: a task created with a skill
  name that resolves nowhere must raise BEFORE the task is enqueued,
  not crash a worker after spawn.
* Class #2 — Retry exhaustion without diagnostics: a task that exhausts
  its failure budget must land in `gave_up` carrying a structured
  `failure_class` + `failure_hint`, not a bare error line.
* Class #3 — Cross-profile skill resolution: a task pinned to a
  specialist profile whose skill lives only in the master/heisenberg
  profile must have that dir injected into the worker's
  `HERMES_SKILLS_EXTRA_DIRS` so the worker can load it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    kb.init_db()
    return home


def _make_profile_skill(monkeypatch, tmp_path, profile: str, skill: str) -> Path:
    """Create a profile skills dir with one SKILL.md and return the dir."""
    skills = tmp_path / f"profiles/{profile}/skills"
    sk = skills / skill
    sk.mkdir(parents=True)
    (sk / "SKILL.md").write_text(
        "---\nname: %s\ndescription: x\n---\nbody\n" % skill, encoding="utf-8"
    )
    return skills


# ---------------------------------------------------------------------------
# Class #1 — Pre-spawn validation crash
# ---------------------------------------------------------------------------

def test_create_task_with_unknown_skill_raises(kanban_home, monkeypatch):
    """An unresolvable skill name must fail fast at create_task."""
    monkeypatch.setattr(kb, "_skill_resolves_somewhere", lambda name, **kw: False)
    conn = kb.connect()
    try:
        with pytest.raises(ValueError, match="Unknown skill"):
            kb.create_task(
                conn,
                title="bad skill",
                assignee="walter",
                skills=["this-skill-does-not-exist-anywhere"],
            )
    finally:
        conn.close()


def test_create_task_with_valid_skill_passes(kanban_home, tmp_path, monkeypatch):
    """A skill that exists on the assignee profile is accepted.

    We stub the cross-profile resolver (already covered by the
    pure-function path) so the test is deterministic regardless of which
    real profile dirs exist on the host.
    """
    monkeypatch.setattr(kb, "_skill_resolves_somewhere", lambda name, **kw: True)
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="good skill", assignee="walter", skills=["real-skill"]
        )
        t = kb.get_task(conn, tid)
        assert t.skills == ["real-skill"]
    finally:
        conn.close()


def test_create_task_unknown_skill_does_not_enqueue(kanban_home, monkeypatch):
    """On validation failure the task must NOT be persisted."""
    monkeypatch.setattr(kb, "_skill_resolves_somewhere", lambda name, **kw: False)
    conn = kb.connect()
    try:
        before = conn.execute("SELECT COUNT(*) AS c FROM tasks").fetchone()["c"]
        with pytest.raises(ValueError):
            kb.create_task(
                conn, title="x", assignee="walter",
                skills=["nope-not-a-skill"],
            )
        after = conn.execute("SELECT COUNT(*) AS c FROM tasks").fetchone()["c"]
        assert before == after, "invalid-skill task must not be enqueued"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Class #2 — Retry exhaustion without diagnostics
# ---------------------------------------------------------------------------

def test_record_failure_classifies_skill_not_found(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="walter")
        kb._record_task_failure(
            conn, tid,
            "Unknown skill(s): 'hermes-agent' requested via `skills=`",
            outcome="crashed", failure_limit=1, release_claim=False,
        )
        # Breaker tripped → status blocked + gave_up event with class.
        t = kb.get_task(conn, tid)
        assert t.status == "blocked"
        ev = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='gave_up'",
            (tid,),
        ).fetchone()
        assert ev is not None
        payload = json.loads(ev["payload"])
        assert payload["failure_class"] == "skill_not_found"
        assert "hermes-agent" in payload["failure_hint"]
    finally:
        conn.close()


def test_record_failure_classifies_timed_out(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="walter")
        kb._record_task_failure(
            conn, tid,
            "Iteration budget exhausted (40/40) — task could not complete",
            outcome="timed_out", failure_limit=1, release_claim=False,
        )
        ev = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='gave_up'",
            (tid,),
        ).fetchone()
        payload = json.loads(ev["payload"])
        assert payload["failure_class"] == "timed_out"
    finally:
        conn.close()


def test_classify_failure_cause_pure(kanban_home):
    """_classify_failure_cause is pure and maps known patterns."""
    assert kb._classify_failure_cause("Unknown skill(s): x", "crashed")[0] == "skill_not_found"
    assert kb._classify_failure_cause("HTTP 429 rate limited", "crashed")[0] == "rate_limited"
    assert kb._classify_failure_cause("worker PID no longer alive", "crashed")[0] == "crashed"
    cls, hint = kb._classify_failure_cause("", "timed_out")
    assert cls == "timed_out"
    assert hint  # non-empty hint even with empty error


# ---------------------------------------------------------------------------
# Class #3 — Cross-profile skill resolution
# ---------------------------------------------------------------------------

def test_default_spawn_injects_master_skills_dir(kanban_home, tmp_path, monkeypatch):
    """Worker env must include the master/heisenberg skills dir.

    We intercept the subprocess.Popen call inside _default_spawn to
    capture the env without actually launching a worker.
    """
    import subprocess

    # The code resolves the master skills dir as
    # <HERMES_HOME>/profiles/heisenberg/skills. The fixture sets
    # HERMES_HOME to tmp/.hermes, so create it there to mirror the
    # production layout (and so the dir actually exists / gets injected).
    master_skills = tmp_path / ".hermes" / "profiles" / "heisenberg" / "skills"
    master_skills.mkdir(parents=True)
    captured = {}

    real_popen = subprocess.Popen

    def fake_popen(*args, **kwargs):
        captured["env"] = dict(kwargs.get("env", {}))
        # Return a dummy completed-ish process object.
        class _Dummy:
            pid = 12345
            def communicate(self, *a, **k):
                return (b"", b"")
        return _Dummy()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    # The skill name is only used to build the `--skills` argv; we stub the
    # cross-profile resolver so create_task accepts it regardless of which
    # real profile dirs exist (the dir existence / injection is what this
    # test actually asserts, not skill resolution).
    monkeypatch.setattr(kb, "_skill_resolves_somewhere", lambda name, **kw: True)
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="cross", assignee="walter",
            skills=["hermes-agent"],
        )
        t = kb.get_task(conn, tid)
        kb._default_spawn(t, str(tmp_path / "ws"), board=None)
    finally:
        conn.close()

    assert "captured" in dir() or captured, "Popen was not called"
    env = captured.get("env", {})
    extra = env.get("HERMES_SKILLS_EXTRA_DIRS", "")
    assert str(master_skills) in extra, (
        f"master skills dir not injected: {extra!r}"
    )


# ---------------------------------------------------------------------------
# Class #1 — Pre-spawn guard in the dispatcher (defense-in-depth)
# ---------------------------------------------------------------------------

def test_dispatch_blocks_unresolved_skill_before_spawn(kanban_home, tmp_path, monkeypatch):
    """The dispatcher must block (not spawn) a ready task whose skills
    resolve nowhere at dispatch time, instead of launching a worker that
    crashes.

    Covers the in-tick guard added to ``_dispatch_once_locked`` so a task
    created before the enqueue-time ``create_task`` guard existed (or whose
    skills dir vanished after creation) can still never reach a subprocess.

    Honest scenario: the skill is VALID at create_task time (so the
    enqueue-time pre-spawn guard in class #1 accepts it), then becomes
    unresolvable by the time the dispatcher ticks — e.g. the skills dir was
    deleted after creation. At that point the dispatcher's defense-in-depth
    guard must catch it. Monkeypatching the resolver to a constant ``False``
    would make ``create_task`` itself raise before any row is inserted, so
    we only flip the resolver to ``False`` *after* the task is created.
    The assignee profile must exist on disk so the dispatcher's
    ``profile_exists`` check lets the task reach the pre-spawn guard
    (otherwise it's bucketed as ``skipped_nonspawnable``).
    """
    # Create the assignee profile dir so profile_exists('walter') is True.
    prof = tmp_path / ".hermes" / "profiles" / "walter"
    prof.mkdir(parents=True)
    # Skill resolves at create time; the enqueue-time guard accepts it.
    monkeypatch.setattr(kb, "_skill_resolves_somewhere", lambda name, **kw: True)
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="x", assignee="walter", skills=["ghost-skill"],
        )
        # Now the skill"vanishes" (skills dir deleted) before dispatch.
        monkeypatch.setattr(kb, "_skill_resolves_somewhere", lambda name, **kw: False)
        # A freshly created task starts in `running` (initial_status default);
        # it only becomes dispatchable (`ready`) after a claim+reclaim cycle
        # (e.g. a worker crashed on the bad skill and release_stale_claims
        # reset it to ready). Mirror that transition so the dispatcher's
        # in-tick guard actually sees the task.
        conn.execute("UPDATE tasks SET status='ready', claim_lock=NULL WHERE id=?", (tid,))
        conn.commit()
        spawned = []
        # A spawn_fn that records whether it was called at all.
        def _spy_spawn(task, workspace, board=None):
            spawned.append(task.id)
            return 99999
        # failure_limit=1 so a single unresolved-skill dispatch trips the
        # breaker and auto-blocks immediately (matches the class #2 tests
        # that pass failure_limit=1 to _record_task_failure directly).
        res = kb.dispatch_once(
            conn, spawn_fn=_spy_spawn, dry_run=False, failure_limit=1,
        )
        # The worker must NOT have been spawned.
        assert spawned == [], "dispatcher must not spawn an unresolved-skill task"
        # The task must be auto-blocked with a skill_not_found diagnosis.
        assert tid in res.auto_blocked, "task should be auto-blocked"
        t = kb.get_task(conn, tid)
        assert t.status == "blocked", t.status
        ev = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='gave_up'",
            (tid,),
        ).fetchone()
        assert ev is not None
        payload = json.loads(ev["payload"])
        assert payload.get("failure_class") == "skill_not_found"
    finally:
        conn.close()
