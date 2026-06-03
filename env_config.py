"""Minimal .env loader (no external dependency).

Import this at the top of any script that needs the key:
    import env_config  # noqa: F401  (loads .env into os.environ)

It reads a `.env` file next to this module and sets any KEY=VALUE pairs that
aren't already in the environment. Lines starting with # are ignored.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(path: Path | None = None) -> None:
    path = path or (Path(__file__).resolve().parent / ".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        # don't override anything already exported in the shell
        os.environ.setdefault(key, val)


load_env()
