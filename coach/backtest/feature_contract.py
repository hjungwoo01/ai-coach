from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from coach.model.params import InfluenceWeights, PlayerParams


@dataclass(frozen=True)
class FeatureSpec:
    scope: str
    field_path: str
    dtype: str
    source_tables: tuple[str, ...]
    raw_columns: tuple[str, ...]
    computation: str
    required_for_inference: bool = True
    note: str | None = None


@dataclass(frozen=True)
class FeatureContract:
    model_name: str
    inference_object: str
    template_context_fields: tuple[str, ...]
    player_features: tuple[FeatureSpec, ...]
    weight_features: tuple[FeatureSpec, ...]
    global_features: tuple[FeatureSpec, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "inference_object": self.inference_object,
            "template_context_fields": list(self.template_context_fields),
            "player_features": [asdict(item) for item in self.player_features],
            "weight_features": [asdict(item) for item in self.weight_features],
            "global_features": [asdict(item) for item in self.global_features],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# {self.model_name} Feature Contract",
            "",
            f"Inference object: `{self.inference_object}`",
            "",
            "## Template Context Outputs",
        ]
        for field_name in self.template_context_fields:
            lines.append(f"- `{field_name}`")

        lines.extend(["", "## Player Features"])
        lines.extend(_specs_to_markdown(self.player_features))

        lines.extend(["", "## Weight Features"])
        lines.extend(_specs_to_markdown(self.weight_features))

        lines.extend(["", "## Global Features"])
        lines.extend(_specs_to_markdown(self.global_features))
        return "\n".join(lines)


def _specs_to_markdown(specs: Iterable[FeatureSpec]) -> list[str]:
    lines: list[str] = []
    for spec in specs:
        lines.append(f"- `{spec.field_path}` ({spec.dtype})")
        lines.append(f"  source: {', '.join(spec.source_tables)}")
        lines.append(f"  raw columns: {', '.join(spec.raw_columns)}")
        lines.append(f"  computation: {spec.computation}")
        if spec.note:
            lines.append(f"  note: {spec.note}")
    return lines


_PLAYER_FEATURE_SOURCE_MAP: dict[str, FeatureSpec] = {
    "player_id": FeatureSpec(
        scope="player",
        field_path="player.player_id",
        dtype="str",
        source_tables=("players",),
        raw_columns=("player_id",),
        computation="Canonical stable player identifier.",
        note="Traceability field. Not used numerically by the PCSP equations.",
    ),
    "name": FeatureSpec(
        scope="player",
        field_path="player.name",
        dtype="str",
        source_tables=("players",),
        raw_columns=("name",),
        computation="Display name for traceability and run artifacts.",
        note="Traceability field. Not used numerically by the PCSP equations.",
    ),
    "base_srv_win": FeatureSpec(
        scope="player",
        field_path="player.base_srv_win",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("a_serve_rallies", "a_serve_wins", "b_serve_rallies", "b_serve_wins"),
        computation="Laplace-smoothed serve rally win rate over the historical window: (wins + alpha) / (trials + 2*alpha).",
        note="Must be computed from matches strictly before the prediction cutoff.",
    ),
    "base_rcv_win": FeatureSpec(
        scope="player",
        field_path="player.base_rcv_win",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("a_serve_rallies", "a_serve_wins", "b_serve_rallies", "b_serve_wins"),
        computation="Laplace-smoothed receive rally win rate, where receive_wins = opponent_serve_rallies - opponent_serve_wins.",
    ),
    "unforced_error_rate": FeatureSpec(
        scope="player",
        field_path="player.unforced_error_rate",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("attack_rate", "safe_rate", "flick_rate", "points_for", "points_against"),
        computation="Proxy = 0.08 + 0.22*attack_rate + 0.08*flick_rate + 0.11*point_loss - 0.09*safe_rate, clipped to [0.01, 0.6].",
    ),
    "return_pressure": FeatureSpec(
        scope="player",
        field_path="player.return_pressure",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("base_rcv_win", "attack_rate", "points_for", "points_against"),
        computation="Clipped composite receive-quality signal: 0.58*base_rcv_win + 0.22*attack_rate + 0.20*point_share.",
    ),
    "clutch_point_win": FeatureSpec(
        scope="player",
        field_path="player.clutch_point_win",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("points_for", "points_against", "winner_id"),
        computation="If close matches exist (absolute point diff <= 6), combine close-point share and close-match win smoothing: 0.65*close_point_share + 0.35*close_win_rate; else use overall point share.",
    ),
    "short_serve_skill": FeatureSpec(
        scope="player",
        field_path="player.short_serve_skill",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("short_serve_win_rate", "short_serve_samples"),
        computation="Laplace-smoothed short-serve rally win rate aggregated over historical matches.",
    ),
    "long_serve_skill": FeatureSpec(
        scope="player",
        field_path="player.long_serve_skill",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("long_serve_win_rate", "long_serve_samples"),
        computation="Laplace-smoothed long-serve rally win rate aggregated over historical matches.",
    ),
    "rally_tolerance": FeatureSpec(
        scope="player",
        field_path="player.rally_tolerance",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("avg_rally_len", "long_rally_share"),
        computation="Clipped long-rally comfort signal: 0.45*(avg_rally_len / 12) + 0.55*long_rally_share.",
    ),
    "net_error_rate": FeatureSpec(
        scope="player",
        field_path="player.net_error_rate",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("net_error_lost_rate",),
        computation="Weighted average of terminal lost-rally net-error rate over historical matches.",
    ),
    "out_error_rate": FeatureSpec(
        scope="player",
        field_path="player.out_error_rate",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("out_error_lost_rate",),
        computation="Weighted average of terminal lost-rally out-error rate over historical matches.",
    ),
    "backhand_rate": FeatureSpec(
        scope="player",
        field_path="player.backhand_rate",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("backhand_rate",),
        computation="Weighted average backhand usage rate over historical matches.",
    ),
    "aroundhead_rate": FeatureSpec(
        scope="player",
        field_path="player.aroundhead_rate",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("aroundhead_rate",),
        computation="Weighted average around-the-head usage rate over historical matches.",
    ),
    "handedness_flag": FeatureSpec(
        scope="player",
        field_path="player.handedness_flag",
        dtype="float",
        source_tables=("players",),
        raw_columns=("handedness",),
        computation="1.0 if the player is left-handed, else 0.0.",
    ),
    "reliability": FeatureSpec(
        scope="player",
        field_path="player.reliability",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("serve_rallies", "receive_rallies"),
        computation="Confidence scale sqrt((serve_trials + receive_trials) / 80), clipped to [0, 1].",
    ),
    "recent_form": FeatureSpec(
        scope="player",
        field_path="player.recent_form",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("winner_id", "points_for", "points_against", "date"),
        computation="Recency-weighted composite of win rate and point share computed only from pre-cutoff matches.",
    ),
    "rest_days": FeatureSpec(
        scope="player",
        field_path="player.rest_days",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("date",),
        computation="Days since the player's previous completed match relative to the prediction cutoff.",
    ),
    "serve_mix.short": FeatureSpec(
        scope="player",
        field_path="player.serve_mix.short",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("short_serve_rate", "serve_rallies"),
        computation="Serve-rally-weighted mean short-serve rate with Dirichlet-style smoothing alpha=0.02.",
    ),
    "serve_mix.flick": FeatureSpec(
        scope="player",
        field_path="player.serve_mix.flick",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("flick_serve_rate", "serve_rallies"),
        computation="Computed as 1 - serve_mix.short.",
    ),
    "rally_style.attack": FeatureSpec(
        scope="player",
        field_path="player.rally_style.attack",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("attack_rate", "points_for", "points_against"),
        computation="Rally-weighted historical attack share, clamped and renormalized with neutral/safe.",
    ),
    "rally_style.neutral": FeatureSpec(
        scope="player",
        field_path="player.rally_style.neutral",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("neutral_rate", "points_for", "points_against"),
        computation="Rally-weighted historical neutral share after attack/safe smoothing and renormalization.",
    ),
    "rally_style.safe": FeatureSpec(
        scope="player",
        field_path="player.rally_style.safe",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("safe_rate", "points_for", "points_against"),
        computation="Rally-weighted historical safe share after attack/safe smoothing and renormalization.",
    ),
    "sample_matches": FeatureSpec(
        scope="player",
        field_path="player.sample_matches",
        dtype="int",
        source_tables=("matches",),
        raw_columns=("playerA_id", "playerB_id"),
        computation="Number of historical matches available for the player within the window.",
        note="Traceability and data-quality field; indirectly influences reliability checks.",
    ),
}

_WEIGHT_FEATURE_SOURCE_MAP: dict[str, FeatureSpec] = {
    "w_short": FeatureSpec(
        scope="matchup",
        field_path="weights.w_short",
        dtype="float",
        source_tables=("matches", "players"),
        raw_columns=("serve_mix.short",),
        computation="Point-in-time ridge regression coefficient magnitude for short-serve differential, clipped to [0.01, 0.2].",
        note="Must be fit only on matches before the prediction cutoff.",
    ),
    "w_attack": FeatureSpec(
        scope="matchup",
        field_path="weights.w_attack",
        dtype="float",
        source_tables=("matches", "players"),
        raw_columns=("rally_style.attack",),
        computation="Point-in-time ridge regression coefficient magnitude for attack-style differential, clipped to [0.01, 0.2].",
    ),
    "w_safe": FeatureSpec(
        scope="matchup",
        field_path="weights.w_safe",
        dtype="float",
        source_tables=("matches", "players"),
        raw_columns=("rally_style.safe",),
        computation="Point-in-time ridge regression coefficient magnitude for safe-style differential, clipped to [0.01, 0.2].",
    ),
    "w_ue": FeatureSpec(
        scope="matchup",
        field_path="weights.w_ue",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("unforced_error_rate",),
        computation="Point-in-time ridge regression coefficient magnitude for unforced-error differential, clipped to [0.01, 0.2].",
    ),
    "w_return_pressure": FeatureSpec(
        scope="matchup",
        field_path="weights.w_return_pressure",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("return_pressure",),
        computation="Point-in-time ridge regression coefficient magnitude for return-pressure differential, clipped to [0.01, 0.2].",
    ),
    "w_clutch": FeatureSpec(
        scope="matchup",
        field_path="weights.w_clutch",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("clutch_point_win",),
        computation="Point-in-time ridge regression coefficient magnitude for clutch differential, clipped to [0.01, 0.12].",
    ),
    "w_serve_type": FeatureSpec(
        scope="matchup",
        field_path="weights.w_serve_type",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("short_serve_skill", "long_serve_skill"),
        computation="Point-in-time ridge regression coefficient magnitude for serve-type effectiveness differential, clipped to [0.0, 0.08].",
    ),
    "w_rally_tolerance": FeatureSpec(
        scope="matchup",
        field_path="weights.w_rally_tolerance",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("avg_rally_len", "long_rally_share"),
        computation="Point-in-time ridge regression coefficient magnitude for rally-tolerance differential, clipped to [0.0, 0.08].",
    ),
    "w_error_profile": FeatureSpec(
        scope="matchup",
        field_path="weights.w_error_profile",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("net_error_rate", "out_error_rate"),
        computation="Point-in-time ridge regression coefficient magnitude for terminal error-profile differential, clipped to [0.0, 0.08].",
    ),
    "w_handedness": FeatureSpec(
        scope="matchup",
        field_path="weights.w_handedness",
        dtype="float",
        source_tables=("players",),
        raw_columns=("handedness",),
        computation="Point-in-time ridge regression coefficient magnitude for handedness differential, clipped to [0.0, 0.08].",
    ),
    "w_backhand": FeatureSpec(
        scope="matchup",
        field_path="weights.w_backhand",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("backhand_rate",),
        computation="Point-in-time ridge regression coefficient magnitude for backhand-usage differential, clipped to [0.0, 0.08].",
    ),
    "w_aroundhead": FeatureSpec(
        scope="matchup",
        field_path="weights.w_aroundhead",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("aroundhead_rate",),
        computation="Point-in-time ridge regression coefficient magnitude for around-head differential, clipped to [0.0, 0.08].",
    ),
    "w_recent_form": FeatureSpec(
        scope="matchup",
        field_path="weights.w_recent_form",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("recent_form",),
        computation="Point-in-time ridge regression coefficient magnitude for recency-weighted form differential, clipped to [0.0, 0.12].",
    ),
    "w_rest": FeatureSpec(
        scope="matchup",
        field_path="weights.w_rest",
        dtype="float",
        source_tables=("matches",),
        raw_columns=("date",),
        computation="Point-in-time ridge regression coefficient magnitude for rest-days differential, clipped to [0.0, 0.1].",
    ),
}

_GLOBAL_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        scope="global",
        field_path="target",
        dtype="int",
        source_tables=("config",),
        raw_columns=(),
        computation="Match target score for each game. Defaults to 21.",
        note="Static rules parameter, not learned from data.",
    ),
    FeatureSpec(
        scope="global",
        field_path="cap",
        dtype="int",
        source_tables=("config",),
        raw_columns=(),
        computation="Game cap score. Defaults to 30.",
        note="Static rules parameter, not learned from data.",
    ),
    FeatureSpec(
        scope="global",
        field_path="best_of",
        dtype="int",
        source_tables=("config",),
        raw_columns=(),
        computation="Match length in games. Defaults to best-of-3.",
        note="Static rules parameter, not learned from data.",
    ),
)

_TEMPLATE_CONTEXT_FIELDS = (
    "aroundhead_rate_A",
    "aroundhead_rate_B",
    "backhand_rate_A",
    "backhand_rate_B",
    "baseA_rcv_win",
    "baseA_srv_win",
    "baseB_rcv_win",
    "baseB_srv_win",
    "best_of",
    "cap",
    "clutch_A",
    "clutch_B",
    "games_to_win",
    "handedness_flag_A",
    "handedness_flag_B",
    "long_serve_skill_A",
    "long_serve_skill_B",
    "net_error_rate_A",
    "net_error_rate_B",
    "out_error_rate_A",
    "out_error_rate_B",
    "pA_rcv_lose_w",
    "pA_rcv_win",
    "pA_rcv_win_w",
    "pA_srv_lose_w",
    "pA_srv_win",
    "pA_srv_win_w",
    "playerA_name",
    "playerB_name",
    "rally_style_A_attack",
    "rally_style_A_neutral",
    "rally_style_A_safe",
    "rally_style_B_attack",
    "rally_style_B_neutral",
    "rally_style_B_safe",
    "rally_tolerance_A",
    "rally_tolerance_B",
    "recent_form_A",
    "recent_form_B",
    "reliability_A",
    "reliability_B",
    "rest_days_A",
    "rest_days_B",
    "return_pressure_A",
    "return_pressure_B",
    "serve_mix_A_flick",
    "serve_mix_A_short",
    "serve_mix_B_flick",
    "serve_mix_B_short",
    "short_serve_skill_A",
    "short_serve_skill_B",
    "target",
    "ue_rate_A",
    "ue_rate_B",
    "unforced_error_A",
    "unforced_error_B",
    "w_aroundhead",
    "w_attack",
    "w_backhand",
    "w_clutch",
    "w_error_profile",
    "w_handedness",
    "w_rally_tolerance",
    "w_recent_form",
    "w_rest",
    "w_return_pressure",
    "w_safe",
    "w_serve_type",
    "w_short",
    "w_ue",
)


def extract_pcsp_feature_contract() -> FeatureContract:
    player_specs = _extract_player_feature_specs()
    weight_specs = _extract_weight_feature_specs()
    return FeatureContract(
        model_name="PCSP Badminton Matchup Model",
        inference_object="coach.model.params.MatchupParams",
        template_context_fields=_TEMPLATE_CONTEXT_FIELDS,
        player_features=player_specs,
        weight_features=weight_specs,
        global_features=_GLOBAL_FEATURES,
    )


def _extract_player_feature_specs() -> tuple[FeatureSpec, ...]:
    specs: list[FeatureSpec] = []

    for field_name in PlayerParams.model_fields:
        if field_name == "serve_mix":
            specs.append(_PLAYER_FEATURE_SOURCE_MAP["serve_mix.short"])
            specs.append(_PLAYER_FEATURE_SOURCE_MAP["serve_mix.flick"])
            continue
        if field_name == "rally_style":
            specs.append(_PLAYER_FEATURE_SOURCE_MAP["rally_style.attack"])
            specs.append(_PLAYER_FEATURE_SOURCE_MAP["rally_style.neutral"])
            specs.append(_PLAYER_FEATURE_SOURCE_MAP["rally_style.safe"])
            continue
        spec = _PLAYER_FEATURE_SOURCE_MAP.get(field_name)
        if spec is None:
            raise KeyError(f"Missing feature-contract mapping for PlayerParams.{field_name}")
        specs.append(spec)

    return tuple(specs)


def _extract_weight_feature_specs() -> tuple[FeatureSpec, ...]:
    specs: list[FeatureSpec] = []
    for field_name in InfluenceWeights.model_fields:
        spec = _WEIGHT_FEATURE_SOURCE_MAP.get(field_name)
        if spec is None:
            raise KeyError(f"Missing feature-contract mapping for InfluenceWeights.{field_name}")
        specs.append(spec)
    return tuple(specs)


def write_feature_contract(path: str | Path, *, format: str = "markdown") -> Path:
    contract = extract_pcsp_feature_contract()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "markdown":
        output_path.write_text(contract.to_markdown(), encoding="utf-8")
    elif format == "json":
        output_path.write_text(json.dumps(contract.to_dict(), indent=2), encoding="utf-8")
    else:
        raise ValueError("format must be 'markdown' or 'json'")

    return output_path
