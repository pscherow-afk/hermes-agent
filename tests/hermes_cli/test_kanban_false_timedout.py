"""Regression for false "timed out; will retry" notifications on finished tasks.

Cards t_6f791d5f / t_2dfe40ca / t_82a634b9 (and the diagnostic card t_eac0ddde)
reported "timed out; dispatcher will retry" AFTER their worker had already
emitted kanban_complete (status=done, artifacts on disk). Two bug classes:

1. A terminal ``timed_out``/``gave_up`` event was appended to an already-done
   task because ``_record_task_failure`` called ``_append_event`` even when
   ``_end_run`` returned None (no active run → task already finalized).
2. The notifier's ``unseen_events_for_sub`` returned stale terminal events for a
   task whose ``completed_at`` is set, so the watcher emitted a false ping.

These tests pin the contract: no terminal event for an already-finalized task,
and the notifier must suppress stale timed_out/crashed/gave_up for done/archived
tasks while still advancing the cursor past them.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_notify as kbn


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _now() -> int:
    return int(time.time())


def _claim_and_complete(conn, tid: str) -> None:
    """Move a created task to running then done (with a closed run)."""
    kb.claim_task(conn, tid)
    run_id = kb._current_run_id(conn, tid)
    # Complete via the real path: closes the run + emits "completed".
    assert kb.complete_task(conn, tid, result="done", summary="finished") is True
    assert kb.get_task(conn, tid).status == "done"
    assert kb._current_run_id(conn, tid) is None
    assert run_id is not None


def test_budget_exhausted_after_complete_emits_no_timed_out_event(kanban_home):
    """_record_kanban_budget_exhausted (turn_finalizer) is a no-op for done tasks."""
    import importlib
    from agent import turn_finalizer as tf

    importlib.reload(tf)

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="already done", assignee="w")
        _claim_and_complete(conn, tid)

        # Call the real recorder; it must not append a timed_out event.
        tf._record_kanban_budget_exhausted(tid, 99, 100, __import__("logging").getLogger("t"))

        events = kb.list_events(conn, tid)
        kinds = [e.kind for e in events]
        assert "timed_out" not in kinds
        # Only the lifecycle events expected from complete_task survive.
        assert "completed" in kinds


def test_record_task_failure_end_run_no_append_when_run_already_closed(kanban_home):
    """_record_task_failure with end_run=True appends nothing if the run is closed."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="done via complete", assignee="w")
        _claim_and_complete(conn, tid)
        before = {e.kind for e in kb.list_events(conn, tid)}

        # Simulate the late budget-exhaustion path (end_run=True, outcome=timed_out).
        blocked = kbd._record_task_failure(
            conn, tid, error="Iteration budget exhausted (10/10)",
            outcome="timed_out", release_claim=False, end_run=True,
        )
        assert blocked is False

        after = {e.kind for e in kb.list_events(conn, tid)}
        # No new terminal event appended for an already-finalized task.
        assert after == before
        assert "timed_out" not in after
        # Status unchanged — the card stays done.
        assert kb.get_task(conn, tid).status == "done"


def test_gave_up_event_still_emitted_on_genuine_breaker_trip(kanban_home):
    """end_run=False (genuine breaker flip) must still emit gave_up even with run_id=None."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="will trip", assignee="w")
        kb.claim_task(conn, tid)
        # Force the breaker over its default limit via repeated crash accounting.
        for _ in range(5):
            kbd._record_task_failure(
                conn, tid, error="worker crashed (pid gone)",
                outcome="crashed", release_claim=False, end_run=False,
            )
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        kinds = [e.kind for e in kb.list_events(conn, tid)]
        # At least one gave_up was emitted when the breaker tripped.
        assert "gave_up" in kinds


def test_notifier_suppresses_stale_timed_out_for_done_task(kanban_home):
    """unseen_events_for_sub drops timed_out for a done/archived task but advances the cursor."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="done with leftover timed_out", assignee="w")
        _claim_and_complete(conn, tid)

        # A stale timed_out event left over from a raced exit path.
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "timed_out",
                            {"error": "leftover", "retry_status": "ready"})

        kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="c1")
        # Cursor starts caught up at the completed event; rewind it past the
        # completed event so the stale timed_out is in range.
        kbn.advance_notify_cursor(conn, task_id=tid, platform="telegram",
                                  chat_id="c1", new_cursor=0)

        new_cursor, events = kbn.unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="c1",
            kinds=["completed", "timed_out", "crashed", "gave_up"],
        )
        seen = [e.kind for e in events]
        assert "timed_out" not in seen
        # The completed event (and the suppressed stale one) are still counted
        # toward the cursor so they are never re-notified.
        assert new_cursor > 0
