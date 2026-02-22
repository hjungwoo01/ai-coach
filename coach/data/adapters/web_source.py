from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class WebSourceAdapter:
    """Optional web adapter interface with disk cache support."""

    def __init__(self, cache_dir: str | Path = "coach/data/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, name: str) -> Path:
        safe = "_".join(name.lower().split())
        return self.cache_dir / f"{safe}.json"

    def fetch_player(self, name: str) -> dict[str, Any]:
        cache_file = self._cache_path(name)
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))
        raise RuntimeError(
            "Web adapter is not configured in this project. "
            "Use the local CSV source or provide a project-specific fetch implementation."
        )
