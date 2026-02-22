from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coach.model.params import MatchupParams
from coach.utils import ensure_run_dir, write_json

_PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


@dataclass(frozen=True)
class ModelBuildResult:
    run_dir: Path
    template_path: Path
    matchup_pcsp_path: Path
    params_json_path: Path
    context: dict[str, Any]


def render_template(template_text: str, context: dict[str, Any]) -> str:
    missing: set[str] = set()

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            missing.add(key)
            return match.group(0)
        return str(context[key])

    rendered = _PLACEHOLDER_PATTERN.sub(repl, template_text)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing template parameters: {missing_list}")
    return rendered


def build_matchup_model(
    params: MatchupParams,
    template_name: str = "badminton_rally_template.pcsp",
    out_path: str | Path | None = None,
    run_id: str | None = None,
    run_dir: str | Path | None = None,
) -> ModelBuildResult:
    template_path = Path(__file__).resolve().parent / "templates" / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    context = params.to_template_context()
    template_text = template_path.read_text(encoding="utf-8")
    rendered = render_template(template_text, context)

    if out_path is not None:
        matchup_path = Path(out_path)
        resolved_run_dir = matchup_path.parent
        resolved_run_dir.mkdir(parents=True, exist_ok=True)
    else:
        resolved_run_dir = ensure_run_dir(run_id=run_id, base_dir=run_dir)
        matchup_path = resolved_run_dir / "matchup.pcsp"

    matchup_path.write_text(rendered, encoding="utf-8")

    params_payload = {
        "player_a": params.player_a.model_dump(),
        "player_b": params.player_b.model_dump(),
        "weights": params.weights.model_dump(),
        "effective_probabilities": params.effective_probabilities(),
        "context": context,
    }
    params_path = resolved_run_dir / "params.json"
    write_json(params_path, params_payload)

    return ModelBuildResult(
        run_dir=resolved_run_dir,
        template_path=template_path,
        matchup_pcsp_path=matchup_path,
        params_json_path=params_path,
        context=context,
    )
