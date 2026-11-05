"""``hermes kanban create`` optional notification subscription (#t_681f2590).

The CLI used to create cards with no completion/block subscription, so they
finished silently; the ``kanban_create`` tool auto-subscribes the calling
session. These tests pin the new ``--platform/--chat-id/...`` (and the
``--subscribe-telegram`` shorthand) so a command-created card can notify its
source chat, while staying a strict opt-in: no flags, no subscription.

Idempotency: ``add_notify_sub`` is unique on (task, platform, chat, thread), so
a retry with the same ``--idempotency-key`` + subscribe flags must not accumulate
duplicate rows.
"""

from __future__ import annotations

import argparse

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    kb.init_db()
    return home


def _create(argv):
    """Drive the real ``hermes kanban create`` parser into ``_cmd_create``."""
    root = argparse.ArgumentParser(prog="hermes")
    kc.build_parser(root.add_subparsers())
    args = root.parse_args(["kanban", "create", *argv])
    return kc._cmd_create(args)


def _last_task_id():
    with kbc.connect_closing() as conn:
        tasks = kb.list_tasks(conn)
    assert tasks, "expected _cmd_create to have stored a task"
    return tasks[-1].id


def _subs_for(task_id):
    with kbc.connect_closing() as conn:
        return kbn.list_notify_subs(conn, task_id)


def _all_subs():
    with kbc.connect_closing() as conn:
        return kbn.list_notify_subs(conn, None)


def test_create_with_subscribe_flags_writes_sub(kanban_home, capsys):
    rc = _create(["PROBE", "--platform", "telegram", "--chat-id", "987654",
                  "--user-id", "u1", "--notifier-profile", "explicit-profile"])
    assert rc == 0
    capsys.readouterr()
    subs = _subs_for(_last_task_id())
    assert len(subs) == 1
    s = subs[0]
    assert s["platform"] == "telegram"
    assert s["chat_id"] == "987654"
    assert s["user_id"] == "u1"
    assert s["notifier_profile"] == "explicit-profile"
    # New subscription defaults: chat_type 'dm', delivery_mode 'notify'.
    assert s["chat_type"] == "dm"
    assert s["delivery_mode"] == "notify"


def test_create_subscribe_telegram_shorthand(kanban_home, capsys):
    rc = _create(["PROBE", "--subscribe-telegram", "555", "--notifier-profile", "p2"])
    assert rc == 0
    capsys.readouterr()
    subs = _subs_for(_last_task_id())
    assert len(subs) == 1
    assert subs[0]["platform"] == "telegram"
    assert subs[0]["chat_id"] == "555"
    assert subs[0]["notifier_profile"] == "p2"


def test_create_without_subscribe_flags_writes_no_sub(kanban_home, capsys):
    rc = _create(["PROBE"])
    assert rc == 0
    capsys.readouterr()
    assert _subs_for(_last_task_id()) == []


def test_create_requires_platform_and_chat_id_together(kanban_home, capsys):
    # --platform without --chat-id must be refused (exit 2), not silently skipped.
    assert _create(["PROBE", "--platform", "telegram"]) == 2
    capsys.readouterr()
    assert _all_subs() == []


def test_subscribe_telegram_excludes_explicit_platform_and_chat_id(kanban_home, capsys):
    rc = _create(["PROBE", "--subscribe-telegram", "555", "--platform", "discord", "--chat-id", "x"])
    assert rc == 2
    capsys.readouterr()
    assert _all_subs() == []


def test_create_idempotency_key_does_not_duplicate_subs(kanban_home, capsys):
    argv = ["PROBE", "--idempotency-key", "dup-key",
            "--subscribe-telegram", "777", "--notifier-profile", "p3"]
    assert _create(argv) == 0
    capsys.readouterr()
    assert _create(argv) == 0
    capsys.readouterr()
    subs = _subs_for(_last_task_id())
    assert len(subs) == 1, f"expected exactly 1 subscription, got {len(subs)}"
    assert subs[0]["chat_id"] == "777"


def test_create_defaults_notifier_profile_to_active_profile(kanban_home, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_PROFILE", "env-profile-x")
    rc = _create(["PROBE", "--subscribe-telegram", "888"])
    assert rc == 0
    capsys.readouterr()
    subs = _subs_for(_last_task_id())
    assert len(subs) == 1
    assert subs[0]["notifier_profile"] == "env-profile-x"
