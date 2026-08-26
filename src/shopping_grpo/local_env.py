"""Load ignored project-local environment variables without extra dependencies."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_project_env(path: str | Path | None = None, *, override: bool = False) -> Path | None:
    """Load simple ``KEY=VALUE`` entries from the ignored project ``.env`` file."""

    source = Path(path) if path is not None else PROJECT_ROOT / ".env"
    if not source.is_file():
        return None
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not name or not name.replace("_", "a").isalnum() or name[0].isdigit():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if override or name not in os.environ:
            os.environ[name] = value
    return source
