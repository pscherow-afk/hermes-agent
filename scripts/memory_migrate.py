#!/usr/bin/env python3
"""Entry point: migrate Hermes JSON / MEMORY.md memory into a vector store.

Thin launcher that puts the Hermes repo root on ``sys.path`` and delegates to
``plugins.memory.vector.cli.main`` so the script works regardless of the
current working directory.

    python scripts/memory_migrate.py --backend chroma --embedding local
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    from plugins.memory.vector.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
