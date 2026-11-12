"""Regression for consecutive_failures NULL handling.

Two defects from 2026-11-12:

1. ``create_task`` omitted ``consecutive_failures`` from its INSERT column list,
   so every newly created card landed with ``consecutive_failures = NULL`` (the
   live DB column is nullable, and the DEFAULT was not applied by the
   INSERT-on-omit path). The dispatcher then crashed with
   ``TypeError: int() argument must be a string, ... or a real number, not
   'NoneType'`` in ``_record_task_failure``.

2. ``_record_task_failure`` read ``int(row["consecutive_failures"])`` without a
   NULL guard, so any card carrying NULL crashed the whole dispatch loop.

This test pins both: created cards must carry ``consecutive_failures = 0`` (not
NULL), and ``_record_task_failure`` must tolerate a NULL column value without
raising and must stamp it to 1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_db_graph import decompose_triage_task


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.discard(str(kb.kanban_db_path(board="default").resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def test_create_task_sets_consecutive_failures_to_zero_not_null(conn):
    """A task created through the module API must have consecutive_failures = 0,
    never NULL — this is the root-cause fix for the dispatcher crash."""
    tid = kb.create_task(conn, title="t", assignee="w")
    row = conn.execute(
        "SELECT consecutive_failures FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row is not None
    assert row["consecutive_failures"] is not None, "created card must not be NULL"
    assert row["consecutive_failures"] == 0


def test_record_task_failure_tolerates_null_consecutive_failures(conn):
    """Simulate the live schema state (nullable consecutive_failures holding
    NULL) and prove _record_task_failure does not raise and stamps it to 1."""
    tid = kb.create_task(conn, title="t", assignee="w")

    # Reproduce the buggy live state: drop the NOT NULL DEFAULT column and
    # re-add it as a plain nullable column, then force the row to NULL.
    conn.execute("ALTER TABLE tasks DROP COLUMN consecutive_failures")
    conn.execute("ALTER TABLE tasks ADD COLUMN consecutive_failures INTEGER")
    conn.execute("UPDATE tasks SET consecutive_failures = NULL WHERE id = ?", (tid,))
    conn.commit()

    pre = conn.execute(
        "SELECT consecutive_failures FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert pre["consecutive_failures"] is None, "precondition: bug reproduces on NULL"

    # Must not raise TypeError; must stamp consecutive_failures to 1.
    kbd._record_task_failure(conn, tid, "boom", outcome="crashed")

    post = conn.execute(
        "SELECT consecutive_failures FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert post["consecutive_failures"] == 1


def test_decomposed_child_sets_consecutive_failures_to_zero(conn):
    """The decomposition path (dispatcher auto-decompose, ``kanban decompose``,
    orchestrator fan-out) inserts children via ``_insert_decomposed_child`` — a
    SEPARATE INSERT from ``create_task``. It must also write an explicit
    ``consecutive_failures = 0``; the previous fix covered only ``create_task``,
    so decomposed children landed with NULL and crashed the dispatcher on the
    first failure. This pins the uncovered path."""
    root = kb.create_task(conn, title="root", assignee="w", triage=True)
    child_ids = decompose_triage_task(
        conn, root, root_assignee="w",
        children=[{"title": "child-1", "assignee": "w"}], author="tester",
    )
    assert child_ids, "decompose should create at least one child"
    for cid in child_ids:
        row = conn.execute(
            "SELECT consecutive_failures FROM tasks WHERE id = ?", (cid,)
        ).fetchone()
        assert row is not None
        assert row["consecutive_failures"] is not None, "child must not be NULL"
        assert row["consecutive_failures"] == 0
