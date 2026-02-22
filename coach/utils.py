from __future__ import annotations

import datetime as dt
import json
import re
import uuid
from pathlib import Path
from typing import Any, Mapping

RUNS_DIR_ENV = "COACH_RUNS_DIR"


def utc_timestamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{utc_timestamp()}_{uuid.uuid4().hex[:8]}"


def ensure_run_dir(run_id: str | None = None, base_dir: str | Path | None = None) -> Path:
    if base_dir is None:
        base_dir = Path.cwd() / "runs"
    else:
        base_dir = Path(base_dir)

    run_name = run_id or make_run_id()
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: str | Path, payload: Mapping[str, Any] | list[Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clamp(value: float, low: float = 0.01, high: float = 0.99) -> float:
    return max(low, min(high, value))


def sanitize_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    return cleaned.strip("-") or "item"
