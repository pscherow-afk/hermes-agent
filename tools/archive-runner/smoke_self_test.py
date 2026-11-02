#!/usr/bin/env python3
"""Smoke driver for archive-runner3 self_test (task t_3968860d).

Two usage modes:
  --expect fail  <model>   -> assert self_test returns None within 60s (model 429/unavailable)
  --expect ok    <model>   -> assert self_test returns the model within 60s

Uses the real OPENROUTER_API_KEY from ~/.hermes/.env. Only touches the network
(via self_test); it does NOT build chunks or write any DB.
"""
import os
import sys
import time
import argparse

import importlib.util

_VAR_TMP = "/var/tmp"
_RUNNER = os.path.join(_VAR_TMP, "archive-runner3.py")
if _VAR_TMP not in sys.path:
    sys.path.insert(0, _VAR_TMP)

_spec = importlib.util.spec_from_file_location("archive_runner3", _RUNNER)
ar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ar)


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
    ap.add_argument("--expect", choices=["ok", "fail"], required=True)
    ap.add_argument("model")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    if not load_key():
        print("NO_KEY")
        sys.exit(2)

    start = time.time()
    try:
        res = ar.self_test(ar.RateLimiter(rate=1000), logfn=lambda m: print(m, flush=True),
                           timeout_s=args.timeout, candidates=[args.model])
    except Exception as e:
        dt = time.time() - start
        print(f"EXCEPTION after {dt:.2f}s: {e}")
        sys.exit(3)
    dt = time.time() - start
    print(f"RESULT model={res} elapsed={dt:.2f}s")

    if args.expect == "fail":
        ok = (res is None) and (dt < 60.0)
        print("SMOKE_FAIL_EXPECTED_OK" if ok else "SMOKE_FAIL_UNEXPECTED")
        sys.exit(0 if ok else 1)
    else:
        ok = (res == args.model) and (dt < 60.0)
        print("SMOKE_OK_EXPECTED_OK" if ok else "SMOKE_OK_UNEXPECTED")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
