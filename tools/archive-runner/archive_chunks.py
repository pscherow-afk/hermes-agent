"""archive_chunks.py - shared, deterministic chunk builder for holographic backfill.

Used by BOTH the runner (archive-runner3.py) and the progress oracle
(progress_count.py) so that the chunk count used for DONE-detection matches
the chunk count the runner actually processes.

Requirement coverage (task t_b0d6e707):
- Skips cron sessions and sessions with < 4 messages (empty-ish sessions) so
  we don't waste LLM calls on non-substantive transcripts.
- Read-only access to state.db (SELECT only).
- Deterministic chunk numbering so resume aligns.
"""
import sqlite3


def session_skip_map(state_db):
    """Return the set of session_ids whose messages should be skipped.

    Skip rule: source == 'cron' OR real message count < 4.
    Messages with session_id NULL are never skipped (kept).
    """
    con = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
    cur = con.cursor()
    counts = {}
    for sid, c in cur.execute(
        "SELECT session_id, COUNT(*) FROM messages WHERE session_id IS NOT NULL GROUP BY session_id"
    ).fetchall():
        counts[sid] = c
    sources = {}
    try:
        for sid, src in cur.execute("SELECT id, source FROM sessions").fetchall():
            sources[sid] = (src or "").lower()
    except Exception:
        pass
    con.close()
    skip = set()
    for sid, c in counts.items():
        src = sources.get(sid, "")
        if src == "cron" or c < 4:
            skip.add(sid)
    return skip


def build_chunks(state_db, chunk_chars=64000):
    """Yield (chunk_no, transcript) in message-id order, skipping cron/short sessions.

    Read-only on state.db. Deterministic: same inputs -> same chunk numbers.
    """
    skip = session_skip_map(state_db)
    con = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT id, role, content, session_id FROM messages "
        "WHERE content IS NOT NULL AND length(content) > 0 ORDER BY id ASC"
    ).fetchall()
    con.close()

    chunks = []
    buf = []
    size = 0
    no = 0
    skipped_msgs = 0
    for mid, role, content, sid in rows:
        if sid is not None and sid in skip:
            skipped_msgs += 1
            continue
        piece = f"[id={mid}] ({role}): {content}\n"
        if size + len(piece) > chunk_chars and buf:
            chunks.append((no, "".join(buf)))
            no += 1
            buf = []
            size = 0
        buf.append(piece)
        size += len(piece)
    if buf:
        chunks.append((no, "".join(buf)))
    return chunks, skipped_msgs


def count_chunks(state_db, chunk_chars=64000):
    chunks, _ = build_chunks(state_db, chunk_chars)
    return len(chunks)
