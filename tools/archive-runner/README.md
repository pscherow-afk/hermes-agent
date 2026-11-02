# archive-runner (holographic memory backfill)

Ad-hoc backfill tool that extracts facts from a Hermes profile's `state.db`
into its holographic `memory_store.db`. Generalized over the profile via CLI
params; driven as a detached process by `memfill_chain.sh`.

## Resilience fixes (task t_3968860d)

The runner previously hung for 25+ minutes when a model returned HTTP 429 or
was unreachable (no per-request timeout, no 429 handling on the self-test
probe). Changes:

1. **Per-request HTTP timeout on every call.** `httpx.Timeout(connect=120,
   read=180, write=60, pool=10)` on the main client; a short `connect=120,
   read=timeout_s+2` client for the self-test probe so a stuck/silent
   connection (CLOSE-WAIT) raises instead of blocking the default 180s read.
2. **Explicit 429/5xx handling.** `extract_facts` retries 429/5xx with a
   bounded backoff (5 -> 15 -> 45 s), then gives up with a logged error
   rather than hanging.
3. **Self-test timeout + rotation.** `self_test` probes each candidate with a
   hard 60 s cap per model via a watchdog thread; a 429 or timeout rotates to
   the next candidate and returns `None` (caller aborts cleanly) instead of
   blocking. Regression-guarded by `test_all_429_returns_none_fast` (< 15 s).
4. **Model fallback in the main loop.** The main `ModelPicker` now keeps the
   full candidate list (not just the winning model), so a model that starts
   429'ing mid-run rotates to the next alive fallback automatically.
   Use `--model` plus `--fallback-models 'a,b'` (see `memfill_chain.sh`).

## Layout

- `archive-runner3.py` — the runner.
- `archive_chunks.py` — shared deterministic chunk builder.
- `test_archive_runner3.py` — pytest unit tests (mocked client; no network).
- `smoke_self_test.py` — real-network self_test smoke driver:
  `python smoke_self_test.py --expect {ok,fail} <model>`.
- `smoke-fail.log` / `smoke-ok.log` — captured smoke runs (see Verification).

## Verification

- Unit: `pytest test_archive_runner3.py` (14 passed).
- Smoke fail: `nvidia/nemotron-3-ultra-550b-a55b:free` -> 429, `self_test`
  returned `None` in 0.41 s (was a 25-minute hang).
- Smoke ok: `deepseek/deepseek-v4-flash-0731` -> valid JSON, model returned
  in 2.51 s.
