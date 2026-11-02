#!/usr/bin/env python3
"""Unit tests for archive-runner3.py resilience changes (task t_3968860d).

Covers:
- ModelPicker rotation
- self_test returns None quickly (<15s) when every candidate 429s (no hang)
- self_test returns the first model that yields valid JSON
- extract_facts rotates to the next alive model on 429 (main loop fallback)
- extract_facts respects per_chunk_budget (never hangs)
- parse_facts_array handles messy LLM output
- BACKOFF sequences match the contract (5->15->45)
"""
import importlib.util
import os
import sys
import time
import unittest
from unittest import mock

# Load the runner as an importable module from /var/tmp. The filename has a
# hyphen (archive-runner3.py), so it cannot be imported by name — load by path.
_VAR_TMP = "/var/tmp"
_RUNNER_PATH = os.path.join(_VAR_TMP, "archive-runner3.py")
if _VAR_TMP not in sys.path:
    sys.path.insert(0, _VAR_TMP)

_spec = importlib.util.spec_from_file_location("archive_runner3", _RUNNER_PATH)
ar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ar)

import openai  # noqa: E402


def _make_rate_limit_error():
    # openai >=1.x: RateLimitError(message, *, response, body); response needs
    # a .request attribute, so build a minimal fake.
    resp = mock.Mock()
    resp.request = mock.Mock()
    return openai.RateLimitError("rate limit", response=resp, body=None)


def _raise_rate_limit(*a, **k):
    raise _make_rate_limit_error()


def _make_timeout_error():
    return openai.APITimeoutError(request=None)


class FakeResp:
    def __init__(self, content):
        self.choices = [mock.Mock(message=mock.Mock(content=content))]


class FakeCompletions:
    def __init__(self, fail_models=None, content=None, sleep=0.0,
                 timeout_raise=False):
        self.fail_models = set(fail_models or [])
        self.content = content or '[{"fact":"x","category":"other","source_ids":[1]}]'
        self.sleep = sleep
        self.timeout_raise = timeout_raise
        self.calls = []

    def create(self, model=None, **kwargs):
        self.calls.append(model)
        if self.timeout_raise:
            raise _make_timeout_error()
        if self.sleep:
            time.sleep(self.sleep)
        if model in self.fail_models:
            raise _make_rate_limit_error()
        return FakeResp(self.content)


class FakeClient:
    def __init__(self, fail_models=None, content=None, sleep=0.0,
                 timeout_raise=False):
        self.chat = mock.Mock()
        self.chat.completions = FakeCompletions(fail_models=fail_models,
                                                content=content, sleep=sleep,
                                                timeout_raise=timeout_raise)


class TestModelPicker(unittest.TestCase):
    def test_rotate_past_dead(self):
        p = ar.ModelPicker(["a", "b", "c"])
        self.assertEqual(p.current, "a")
        self.assertEqual(p.mark_unavailable(), "b")
        self.assertEqual(p.mark_unavailable(), "c")
        self.assertIsNone(p.mark_unavailable())  # all dead

    def test_empty(self):
        p = ar.ModelPicker([])
        self.assertIsNone(p.current)
        self.assertIsNone(p.mark_unavailable())


class TestSelfTest(unittest.TestCase):
    @mock.patch.object(ar, "OpenAI", lambda *a, **k: FakeClient())
    @mock.patch.object(ar.httpx, "Timeout", lambda *a, **k: None)
    @mock.patch.object(ar, "extract_facts", side_effect=_raise_rate_limit)
    def test_all_429_returns_none_fast(self, _m):
        # Every candidate 429s immediately (no network hang) -> must return None
        # well under the 60s cap. This is the regression guard for the original
        # "runner hangs forever on 429" bug.
        start = time.time()
        r = ar.self_test(ar.RateLimiter(rate=1000), logfn=lambda m: None,
                         timeout_s=60.0, candidates=["a", "b", "c"])
        dt = time.time() - start
        self.assertIsNone(r)
        self.assertLess(dt, 15.0, f"self_test took {dt:.1f}s, expected fast exit")

    @mock.patch.object(ar, "OpenAI", lambda *a, **k: FakeClient())
    @mock.patch.object(ar.httpx, "Timeout", lambda *a, **k: None)
    @mock.patch.object(ar, "extract_facts",
                       return_value=[{"fact": "x", "category": "other", "source_ids": [1]}])
    def test_success_returns_first_model(self, _m):
        r = ar.self_test(ar.RateLimiter(rate=1000), logfn=lambda m: None,
                         timeout_s=60.0, candidates=["good", "backup"])
        self.assertEqual(r, "good")

    @mock.patch.object(ar, "OpenAI", lambda *a, **k: FakeClient())
    @mock.patch.object(ar.httpx, "Timeout", lambda *a, **k: None)
    @mock.patch.object(ar, "extract_facts",
                       side_effect=[_make_rate_limit_error(),
                                    [{"fact": "y", "category": "bug", "source_ids": [2]}]])
    def test_rotates_to_next_on_429(self, _m):
        r = ar.self_test(ar.RateLimiter(rate=1000), logfn=lambda m: None,
                         timeout_s=60.0, candidates=["bad", "good"])
        self.assertEqual(r, "good")


class TestExtractFacts(unittest.TestCase):
    def test_rotates_to_next_model_on_429(self):
        picker = ar.ModelPicker(["bad", "good"])
        client = FakeClient(fail_models={"bad"})
        facts = ar.extract_facts(client, picker, "tx", 1,
                                 ar.RateLimiter(rate=1000), retries=6,
                                 rotate_on_429=True, selftest=False)
        self.assertEqual(picker.current, "good")
        self.assertIsInstance(facts, list)
        self.assertEqual(facts[0]["fact"], "x")

    def test_no_hang_on_slow_connection(self):
        # A connection that would block longer than the per-chunk budget must
        # not hang: the real OpenAI client surfaces httpx timeouts as
        # APITimeoutError, and extract_facts must then raise a RuntimeError
        # (budget exceeded) instead of blocking. We emulate the timeout.
        picker = ar.ModelPicker(["slow"])
        client = FakeClient(timeout_raise=True)
        # per_chunk_budget is tiny; even a single (emulated) timeout must not
        # retry forever -> RuntimeError.
        start = time.time()
        with self.assertRaises(RuntimeError):
            ar.extract_facts(client, picker, "tx", 1,
                             ar.RateLimiter(rate=1000), retries=6,
                             per_chunk_budget=0.2, selftest=False)
        dt = time.time() - start
        self.assertLess(dt, 10.0, f"extract_facts hung for {dt:.1f}s")


class TestParseFacts(unittest.TestCase):
    def test_strict_json(self):
        self.assertEqual(len(ar.parse_facts_array('[{"fact":"a"}]')), 1)

    def test_code_fenced(self):
        self.assertEqual(len(ar.parse_facts_array('```json\n[{"fact":"a"}]\n```')), 1)

    def test_bare_array_in_text(self):
        self.assertEqual(len(ar.parse_facts_array('here: [{"fact":"a"}] end')), 1)

    def test_empty(self):
        self.assertEqual(ar.parse_facts_array('[]'), [])


class TestBackoff(unittest.TestCase):
    def test_429_sequence(self):
        self.assertEqual(ar.BACKOFF_429, [5, 15, 45])

    def test_5xx_sequence(self):
        self.assertEqual(ar.BACKOFF_5XX, [5, 15, 45])

    def test_backoff_clamps(self):
        self.assertEqual(ar._backoff(ar.BACKOFF_429, 0), 5)
        self.assertEqual(ar._backoff(ar.BACKOFF_429, 2), 45)
        self.assertEqual(ar._backoff(ar.BACKOFF_429, 9), 45)  # clamp


if __name__ == "__main__":
    unittest.main(verbosity=2)
