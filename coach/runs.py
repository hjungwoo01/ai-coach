from __future__ import annotations

import datetime as dt
import secrets
from pathlib import Path


def new_run_dir(prefix: str, base_dir: str | Path = "runs") -> tuple[str, Path]:
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(4)
    run_id = f"{prefix}_{timestamp}_{suffix}"
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir
