from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None or value.strip() == "":
        return None
    lowered = value.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}. Use one of {_TRUE_VALUES | _FALSE_VALUES}.")


@dataclass(frozen=True)
class CoachConfig:
    pat_console_path: Path | None
    pat_use_mono: bool | None
    mono_path: str
    pat_timeout_s: int
    runs_dir: Path

    @classmethod
    def from_env(cls) -> "CoachConfig":
        load_dotenv(override=False)

        pat_console_raw = os.getenv("PAT_CONSOLE_PATH", "").strip()
        pat_console_path = Path(pat_console_raw).expanduser() if pat_console_raw else None

        pat_use_mono = _parse_optional_bool(os.getenv("PAT_USE_MONO"))

        mono_path = os.getenv("MONO_PATH", "mono").strip() or "mono"

        timeout_raw = os.getenv("PAT_TIMEOUT_S", "120").strip()
        try:
            pat_timeout_s = int(timeout_raw)
        except ValueError as exc:
            raise ValueError(f"PAT_TIMEOUT_S must be an integer, got {timeout_raw!r}") from exc
        if pat_timeout_s <= 0:
            raise ValueError("PAT_TIMEOUT_S must be > 0")

        runs_dir_raw = os.getenv("RUNS_DIR", "runs").strip() or "runs"
        runs_dir = Path(runs_dir_raw)

        return cls(
            pat_console_path=pat_console_path,
            pat_use_mono=pat_use_mono,
            mono_path=mono_path,
            pat_timeout_s=pat_timeout_s,
            runs_dir=runs_dir,
        )

    def resolve_use_mono(self, pat_console_path: Path | None = None) -> bool:
        if self.pat_use_mono is not None:
            return self.pat_use_mono

        path = pat_console_path or self.pat_console_path
        if path is None:
            return False
        return path.suffix.lower() == ".exe"
