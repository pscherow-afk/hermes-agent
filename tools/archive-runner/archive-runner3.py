#!/usr/bin/env python3
"""
archive-runner3.py - holographic memory backfill for ONE Hermes profile.

Generalized over the profile: state.db / memory_store.db / checkpoint / log are
parameters (--state-db, --memory-db, --work-dir), not copy-paste. Drives one
profile to completion; the orchestrator (run_all3b.sh) sequences the three.

Task t_b0d6e707 requirements honored:
- Runs as a SEPARATE process (launched detached by the orchestrator via setsid),
  never inside the agent session (no 50-step cap).
- Checkpoint + resume: DB-backed `archive_progress(chunk_no, status)`; a restarted
  run continues from the last completed chunk.
- Dedup before insert: content is UNIQUE in the schema; store.add_fact returns the
  existing fact_id on conflict (no duplicate rows).
- Insert WITH vector immediately: add_fact computes hrr_vector; we verify it landed
  and recompute if a chunk insert somehow stored None (never ship a fact without a
  vector).
- trust_score 0.55..0.75 by chunk position (later source => higher), tags
  src:<profile>,src:archive.
- Skip cron sessions and sessions with < 4 messages (empty-ish sessions) via the
  shared archive_chunks builder.
- 4h-per-profile soft cap: --limit-seconds (kept < reclaim window) stops cleanly
  with the DB checkpoint; --resume continues.
- Secrets never stored: text is redacted before the LLM and before storage.

Graceful shutdown on SIGTERM/SIGINT: stop pulling new chunks, let in-flight finish,
commit, exit.
"""
import os
import re
import sys
import time
import json
import math
import signal
import argparse
import sqlite3
import threading
import traceback
from pathlib import Path

_SHUTDOWN = threading.Event()


def _on_signal(signum, frame):
    _SHUTDOWN.set()


signal.signal(signal.SIGTERM, _on_signal)
signal.signal(signal.SIGINT, _on_signal)

_HERMES_AGENT = os.environ.get("HERMES_AGENT_REPO") or "/home/pscher/.hermes/hermes-agent"
if _HERMES_AGENT not in sys.path:
    sys.path.insert(0, _HERMES_AGENT)

# Shared, deterministic chunk builder (same counts as the oracle).
sys.path.insert(0, "/var/tmp")
from archive_chunks import build_chunks  # noqa: E402

import openai  # noqa: E402
from openai import OpenAI  # noqa: E402
import httpx  # noqa: E402

SECRET_PATTERNS = [
    re.compile(r"sk-or-v1-\S+"),
    re.compile(r"sk-\S+"),
    re.compile(r"eyJ[A-Za-z0-9._-]{10,}"),
    re.compile(r"\b[A-Fa-f0-9]{32,}\b"),
    re.compile(r"(?i)(password|passwd|pwd|token|secret|api[_-]?key)\s*[:=]\s*\S+"),
    re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._\-+/=]+"),
    re.compile(r"(?i)(wb|moysklad|sber|ozon|wildberries)[_-]?token\s*[:=]\s*\S+"),
    re.compile(r"SUDO_PASSWORD=.*"),
]


def redact(text: str) -> str:
    if not text:
        return text
    for pat in SECRET_PATTERNS:
        text = pat.sub("<REDACTED:secret>", text)
    return text


SYSTEM_PROMPT = (
    "You are a memory-extraction engine for an autonomous coding agent's long-term memory. "
    "Given a raw conversation transcript (role-tagged lines), extract concrete, self-contained "
    "facts worth remembering across future sessions. Focus on: project decisions, architecture, "
    "API endpoints and their quirks, config locations, recurring bugs and their fixes, tool "
    "workflows, credentials-free environment facts, naming conventions, and operational runbooks. "
    "Ignore chit-chat, greetings, and purely ephemeral content. "
    "Return STRICT JSON only: an array of objects, each with keys "
    "`fact` (one concise sentence, no role prefix), `category` (one of: "
    "infra, project, tool, bug, config, convention, person, decision, other), "
    "and `source_ids` (array of integer message ids that support the fact). "
    "If a fact is about a token/key, describe its PURPOSE and LOCATION, never the secret value. "
    "Do not wrap the JSON in markdown. Empty transcript -> empty array []."
)

USER_TEMPLATE = (
    "Extract facts from this transcript chunk (message ids are shown as [id=N]).\n\n{transcript}"
)


def parse_facts_array(content: str) -> list:
    s = (content or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
    s = s.strip()
    attempts = [s]
    if "[" in s and "]" in s:
        attempts.append(s[s.index("["): s.rindex("]") + 1])
    for a in attempts:
        try:
            data = json.loads(a)
            if isinstance(data, list):
                return data
        except Exception:
            continue
    out = []
    depth = 0
    start = None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    blob = s[start: i + 1]
                    try:
                        obj = json.loads(blob)
                        if isinstance(obj, dict):
                            out.append(obj)
                    except Exception:
                        try:
                            import ast
                            obj = ast.literal_eval(blob)
                            if isinstance(obj, dict):
                                out.append(obj)
                        except Exception:
                            pass
                    start = None
    if out:
        return out
    out = []
    for obj in re.findall(r"\{[^{}]*\}", s, flags=re.DOTALL):
        try:
            out.append(json.loads(obj))
        except Exception:
            try:
                import ast
                out.append(ast.literal_eval(obj))
            except Exception:
                continue
    return out


def _log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} {msg}", flush=True)


BACKOFF_429 = [5, 15, 45]
BACKOFF_5XX = [5, 15, 45]
BACKOFF_CONN = [5, 15, 30]


class ModelPicker:
    """Ordered candidate models; rotate to the next alive one on failure."""

    def __init__(self, models):
        self.models = [m for m in models if m]
        self._idx = 0
        self._dead = set()
        self._lock = threading.Lock()

    @property
    def current(self):
        with self._lock:
            if not self.models:
                return None
            return self.models[self._idx]

    def mark_unavailable(self):
        """Advance past the current model to the next alive one.

        Returns the new current model, or None if every model is dead.
        """
        with self._lock:
            n = len(self.models)
            if n == 0:
                return None
            self._dead.add(self._idx)
            for step in range(1, n + 1):
                cand = (self._idx + step) % n
                if cand not in self._dead:
                    self._idx = cand
                    return self.models[self._idx]
            return None


def _backoff(seq, attempt):
    if not seq:
        return 0
    return seq[attempt] if attempt < len(seq) else seq[-1]


def self_test(rate, logfn=_log, timeout_s=60.0, candidates=None):
    """Probe candidate models; return the first one that yields valid JSON.

    Each probe uses a SHORT_TIMEOUT client so a stuck/silent connection (e.g.
    CLOSE-WAIT) raises within ``timeout_s`` instead of blocking the default
    180s read. A 429 or failure rotates to the next candidate. Returns the
    winning model string, or None if no candidate answered (caller aborts).
    """
    if not candidates:
        return None

    probe = "[id=1] (user): The WB API free tier allows 20 requests per minute."
    picker = ModelPicker(candidates)
    short = httpx.Timeout(connect=min(120.0, timeout_s + 2),
                          read=timeout_s + 2, write=30.0, pool=10.0)
    key = os.environ.get("OPENROUTER_API_KEY")
    probe_client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1",
                          timeout=short, max_retries=0)

    def _probe():
        return extract_facts(probe_client, picker, probe, 0, rate, retries=1,
                             max_tokens=300, per_chunk_budget=timeout_s,
                             rotate_on_429=False, selftest=True, logfn=logfn)

    seen = 0
    max_tries = len(picker.models) + 1
    while seen < max_tries:
        model = picker.current
        box = {}
        th = threading.Thread(target=lambda: _store(box, _probe), daemon=True)
        th.start()
        th.join(timeout_s + 2.0)
        if th.is_alive():
            logfn(f"self-test: {model} no response within {timeout_s:.0f}s (hang); trying next")
            seen += 1
            if picker.mark_unavailable() is None:
                break
            continue
        if "err" in box:
            e = box["err"]
            if isinstance(e, openai.RateLimitError):
                logfn(f"self-test: {model} -> 429; trying next candidate")
            else:
                logfn(f"self-test: {model} failed: {e}; trying next")
        else:
            test = box["val"]
            if not isinstance(test, list):
                logfn(f"self-test: {model} returned non-list; trying next")
            else:
                logfn(f"self-test OK on {model}: {len(test)} fact(s)")
                return model
        seen += 1
        if picker.mark_unavailable() is None:
            break
    logfn("self-test: all candidate models unavailable (429/timeout/error)")
    return None


def _store(box, fn):
    try:
        box["val"] = fn()
    except Exception as e:  # noqa: BLE001 - captured for the caller thread
        box["err"] = e


def extract_facts(client, picker, transcript, chunk_no, rate, retries=6,
                  max_tokens=4000, per_chunk_budget=300.0, rotate_on_429=True,
                  selftest=False, logfn=_log):
    """Extract facts with bounded retries, 429/5xx backoff, and model rotation.

    - Per-request timeout is enforced by the client (httpx.Timeout).
    - 429 -> rotate to next model if available, else backoff 5->15->45s.
    - 5xx -> backoff 5->15->45s and retry.
    - Connection/timeout errors -> short backoff and retry.
    - per_chunk_budget caps total wall-clock so a dead connection can't hang.
    """
    start = time.time()
    last_err = None
    for attempt in range(retries):
        if _SHUTDOWN.is_set():
            break
        if time.time() - start > per_chunk_budget:
            last_err = RuntimeError("per-chunk budget exceeded")
            break
        rate.acquire()
        model = picker.current
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(transcript=transcript)},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            content = resp.choices[0].message.content if resp.choices else None
            if not content:
                return []
            data = parse_facts_array(content)
            if not isinstance(data, list):
                return []
            out = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                fact = (item.get("fact") or "").strip()
                if not fact:
                    continue
                cat = (item.get("category") or "other").strip().lower()
                sids = item.get("source_ids") or []
                sids = [int(s) for s in sids if str(s).strip().lstrip("-").isdigit()]
                out.append({"fact": redact(fact), "category": cat, "source_ids": sids})
            return out
        except openai.RateLimitError as e:
            last_err = e
            nxt = picker.mark_unavailable() if rotate_on_429 else None
            if nxt is not None:
                logfn(f"  chunk {chunk_no}: 429 on {model}; rotating -> {nxt}")
                continue
            if selftest:
                raise
            wait = _backoff(BACKOFF_429, attempt)
            logfn(f"  chunk {chunk_no}: 429 on {model}; backoff {wait}s (no fallback)")
            time.sleep(wait)
            continue
        except openai.InternalServerError as e:  # 5xx
            last_err = e
            if not selftest:
                wait = _backoff(BACKOFF_5XX, attempt)
                logfn(f"  chunk {chunk_no}: 5xx {getattr(e, 'status_code', '?')} on {model}; backoff {wait}s")
                time.sleep(wait)
            continue
        except openai.APIStatusError as e:  # other 4xx
            last_err = e
            nxt = picker.mark_unavailable() if rotate_on_429 else None
            if nxt is not None:
                logfn(f"  chunk {chunk_no}: {getattr(e, 'status_code', '?')} on {model}; rotating -> {nxt}")
                continue
            raise
        except (openai.APITimeoutError, openai.APIConnectionError) as e:
            last_err = e
            if not selftest:
                wait = _backoff(BACKOFF_CONN, attempt)
                logfn(f"  chunk {chunk_no}: conn/timeout on {model} ({e}); backoff {wait}s")
                time.sleep(wait)
            continue
        except Exception as e:
            last_err = e
            if not selftest:
                wait = min(2 ** (attempt + 1), 15)
                logfn(f"  chunk {chunk_no}: unexpected {e}; backoff {wait}s")
                time.sleep(wait)
            continue
    raise RuntimeError(f"chunk {chunk_no} failed after {retries} retries: {last_err}")


class RateLimiter:
    def __init__(self, rate=19, period=60.0, min_spacing=None):
        self.rate = rate
        self.period = period
        self.min_spacing = min_spacing if min_spacing is not None else period / rate
        self._lock = threading.Lock()
        self._tokens = rate
        self._last = time.time()
        self._last_call = 0.0

    def acquire(self):
        with self._lock:
            now = time.time()
            self._tokens += (now - self._last) * (self.rate / self.period)
            if self._tokens > self.rate:
                self._tokens = self.rate
            self._last = now
            since = now - self._last_call
            if since < self.min_spacing:
                time.sleep(self.min_spacing - since)
                now = time.time()
            if self._tokens >= 1:
                self._tokens -= 1
                self._last_call = now
                return
            wait = (1 - self._tokens) * (self.period / self.rate)
            time.sleep(wait + 0.1)
            self._tokens = 0
            self._last_call = time.time()


def load_key():
    envp = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(envp):
        for line in open(envp, encoding="utf-8", errors="replace"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == "OPENROUTER_API_KEY":
                    os.environ.setdefault("OPENROUTER_API_KEY", v.strip())
                    return True
    return bool(os.getenv("OPENROUTER_API_KEY"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--state-db", default=None)
    ap.add_argument("--memory-db", default=None)
    ap.add_argument("--work-dir", default="/var/tmp")
    ap.add_argument("--model", default="nvidia/nemotron-3-ultra-550b-a55b:free")
    ap.add_argument("--fallback-models", default=None,
                    help="Comma-separated list of fallback models tried if --model "
                         "errors (429/5xx/unavailable), e.g. "
                         "'deepseek/deepseek-v4-flash-0731,z-ai/glm-5.3-flash'")
    ap.add_argument("--chunk-chars", type=int, default=64000)
    ap.add_argument("--limit-seconds", type=int, default=13800)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--rate", type=int, default=19)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true",
                    help="(accepted for CLI-compat; resume is automatic via archive_progress)")
    ap.add_argument("--force-restart", action="store_true",
                    help="ignore existing archive_progress and rebuild from scratch")
    args = ap.parse_args()

    profile = args.profile
    state_db = args.state_db or os.path.expanduser(f"~/.hermes/profiles/{profile}/state.db")
    memory_db = args.memory_db or os.path.expanduser(f"~/.hermes/profiles/{profile}/memory_store.db")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = work_dir / f"archive-ckpt3-{profile}.json"
    log_path = work_dir / f"archive-runner3-{profile}.log"

    _log_lock = threading.Lock()

    def log(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} {msg}"
        with _log_lock:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass
        try:
            print(line, flush=True)
        except BrokenPipeError:
            pass

    log(f"=== ARCHIVE RUNNER3 START profile={profile} model={args.model} "
        f"chunk_chars={args.chunk_chars} rate={args.rate}/min limit={args.limit_seconds}s ===")

    if not load_key():
        log("ERROR: OPENROUTER_API_KEY not available; abort.")
        sys.exit(2)
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        timeout=httpx.Timeout(connect=120.0, read=180.0, write=60.0, pool=10.0),
        max_retries=0,
    )

    candidates = [args.model]
    if args.fallback_models:
        candidates += [m.strip() for m in args.fallback_models.split(",") if m.strip()]
    log(f"model candidates (in order): {candidates}")

    log("self-test: probing model(s) for valid JSON (60s hard cap per model)...")
    active_model = self_test(RateLimiter(rate=args.rate), logfn=log, timeout_s=60.0,
                             candidates=candidates)
    if not active_model:
        log("ERROR: no usable model passed self-test (all 429/timeout/error); abort.")
        sys.exit(4)
    # Keep the full candidate list (not just the winning model) so that if the
    # active model starts 429'ing mid-run, extract_facts can rotate to the next
    # alive fallback instead of stalling on the same model. self_test already
    # proved at least one candidate answers; the rest are tried on-demand.
    picker = ModelPicker(candidates)
    log(f"self-test OK: active model = {picker.current}")

    try:
        chunks, skipped = build_chunks(state_db, args.chunk_chars)
    except Exception as e:
        log(f"ERROR building chunks from {state_db}: {e}")
        sys.exit(3)
    total = len(chunks)
    log(f"profile {profile}: total chunks={total} (skipped messages in short/cron sessions={skipped})")

    from plugins.memory.holographic.store import MemoryStore
    store = MemoryStore(db_path=memory_db, default_trust=0.6, hrr_dim=1024)
    scon = store._conn
    scon.execute("CREATE TABLE IF NOT EXISTS archive_progress (chunk_no INTEGER PRIMARY KEY, status TEXT, facts INTEGER DEFAULT 0)")
    scon.commit()

    if args.force_restart:
        log("force-restart requested: clearing archive_progress")
        scon.execute("DELETE FROM archive_progress")
        scon.commit()

    done_set = set(r[0] for r in scon.execute("SELECT chunk_no FROM archive_progress WHERE status='done'").fetchall())
    error_set = set(r[0] for r in scon.execute("SELECT chunk_no FROM archive_progress WHERE status='error'").fetchall())
    if error_set:
        log(f"resume: {len(error_set)} prior error chunks will be retried")
    pending = [(no, tx) for (no, tx) in chunks if no not in done_set]
    log(f"resume: done={len(done_set)} pending={len(pending)}")

    # Persist a JSON checkpoint too (belt-and-suspenders alongside DB progress).
    def save_ckpt():
        try:
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({"profile": profile, "done": len(done_set),
                           "pending": len(pending), "total": total,
                           "ts": time.time()}, f)
        except Exception:
            pass

    rate = RateLimiter(rate=args.rate, period=60.0, min_spacing=60.0 / args.rate)
    start = time.time()
    deadline = start + args.limit_seconds
    facts_written = 0
    errors = 0
    processed = 0
    _cnt_lock = threading.Lock()
    _ckpt_every = 10

    def verify_vector(fid, fact_text):
        row = scon.execute("SELECT hrr_vector FROM facts WHERE fact_id=?", (fid,)).fetchone()
        vec = row[0] if row else None
        if vec is None or len(vec) == 0:
            try:
                store._compute_hrr_vector(fid, fact_text)
            except Exception as e:
                log(f"  vector recompute failed for fid={fid}: {e}")

    def handle(no, tx):
        nonlocal facts_written, errors, processed
        try:
            facts = extract_facts(client, picker, tx, no, rate,
                                  max_tokens=args.max_tokens,
                                  per_chunk_budget=args.limit_seconds)
            with _cnt_lock:
                for ft in facts:
                    try:
                        fid = store.add_fact(
                            content=ft["fact"],
                            category=ft.get("category", "other"),
                            tags=f"src:{profile},src:archive",
                        )
                        verify_vector(fid, ft["fact"])
                        target = 0.55 + 0.20 * (no / max(1, total))
                        store.update_fact(fid, trust_delta=target - 0.6)
                        facts_written += 1
                    except Exception as e:
                        errors += 1
                        log(f"  store err chunk {no}: {e}")
            scon.execute("INSERT OR REPLACE INTO archive_progress (chunk_no, status, facts) VALUES (?, 'done', ?)",
                         (no, len(facts)))
            scon.commit()
            with _cnt_lock:
                processed += 1
                if processed % _ckpt_every == 0:
                    save_ckpt()
                    log(f"  progress: {processed}/{len(pending)} done, facts={facts_written}, errors={errors}, "
                        f"elapsed={(time.time()-start)/60:.1f}m")
        except Exception as e:
            with _cnt_lock:
                errors += 1
                processed += 1
            log(f"  chunk {no} FAILED: {e}")
            scon.execute("INSERT OR REPLACE INTO archive_progress (chunk_no, status, facts) VALUES (?, 'error', ?)",
                         (no, str(e)[:200]))
            scon.commit()

    N_WORKERS = args.workers
    _pending_iter = iter(pending)
    _iter_lock = threading.Lock()

    def worker():
        while True:
            with _iter_lock:
                if _SHUTDOWN.is_set() or time.time() > deadline:
                    return
                try:
                    item = next(_pending_iter)
                except StopIteration:
                    return
            handle(*item)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(N_WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed = time.time() - start
    save_ckpt()
    log(f"=== DONE profile={profile} processed={processed}/{len(pending)} "
        f"facts_written={facts_written} errors={errors} elapsed={elapsed/60:.1f}m ===")
    store.close()


if __name__ == "__main__":
    main()
