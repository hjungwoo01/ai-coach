from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from coach.data.adapters.local_csv import LocalCSVAdapter, PlayerRecord
from coach.model.params import InfluenceWeights, MatchupParams, PlayerParams, RallyStyleMix, ServeMix
from coach.utils import clamp


@dataclass(frozen=True)
class MatchupStats:
    player_a: PlayerRecord
    player_b: PlayerRecord
    player_a_stats: dict[str, Any]
    player_b_stats: dict[str, Any]
    head_to_head: dict[str, Any]
    weights: InfluenceWeights


def _resolve_player(adapter: LocalCSVAdapter, player_ref: str) -> PlayerRecord:
    by_id = adapter.players_df[adapter.players_df["player_id"] == player_ref]
    if not by_id.empty:
        row = by_id.iloc[0]
        return PlayerRecord(
            player_id=str(row["player_id"]),
            name=str(row["name"]),
            country=str(row.get("country", "")) or None,
            handedness=str(row.get("handedness", "")) or None,
        )
    return adapter.resolve_player(player_ref)


def estimate_influence_weights(adapter: LocalCSVAdapter) -> InfluenceWeights:
    df = adapter.matches_df.copy()
    if len(df) < 10:
        return InfluenceWeights(w_short=0.04, w_attack=0.06, w_safe=0.05)

    x_short = df["a_short_serve_rate"] - df["b_short_serve_rate"]
    x_attack = df["a_attack_rate"] - df["b_attack_rate"]
    x_safe_term = -(df["b_safe_rate"] - df["a_safe_rate"])

    total_points = df["a_points"] + df["b_points"]
    y = (df["a_points"] / total_points) - 0.5

    X = np.column_stack([
        np.ones(len(df), dtype=float),
        x_short.to_numpy(dtype=float),
        x_attack.to_numpy(dtype=float),
        x_safe_term.to_numpy(dtype=float),
    ])

    beta, *_ = np.linalg.lstsq(X, y.to_numpy(dtype=float), rcond=None)

    w_short = float(np.clip(abs(beta[1]), 0.01, 0.2))
    w_attack = float(np.clip(abs(beta[2]), 0.01, 0.2))
    w_safe = float(np.clip(abs(beta[3]), 0.01, 0.2))

    return InfluenceWeights(w_short=w_short, w_attack=w_attack, w_safe=w_safe)


def _build_player_params(stats: dict[str, Any], sample_matches: int) -> PlayerParams:
    serve_mix = ServeMix(short=float(stats["serve_mix"]["short"]), flick=float(stats["serve_mix"]["flick"]))
    rally_style = RallyStyleMix(
        attack=float(stats["rally_style"]["attack"]),
        neutral=float(stats["rally_style"]["neutral"]),
        safe=float(stats["rally_style"]["safe"]),
    )

    return PlayerParams(
        player_id=str(stats["player_id"]),
        name=str(stats["name"]),
        base_srv_win=clamp(float(stats["base_srv_win"])),
        base_rcv_win=clamp(float(stats["base_rcv_win"])),
        serve_mix=serve_mix,
        rally_style=rally_style,
        sample_matches=sample_matches,
    )


def build_matchup_params(
    adapter: LocalCSVAdapter,
    player_a_ref: str,
    player_b_ref: str,
    window: int = 30,
    as_of_date: str | None = None,
) -> tuple[MatchupParams, MatchupStats]:
    player_a = _resolve_player(adapter, player_a_ref)
    player_b = _resolve_player(adapter, player_b_ref)

    if player_a.player_id == player_b.player_id:
        raise ValueError("Player A and Player B must be different players.")

    a_stats = adapter.get_player_params(player_a.player_id, window=window, as_of_date=as_of_date)
    b_stats = adapter.get_player_params(player_b.player_id, window=window, as_of_date=as_of_date)
    h2h = adapter.get_head_to_head(player_a.player_id, player_b.player_id, window=window, as_of_date=as_of_date)
    weights = estimate_influence_weights(adapter)

    blend = min(0.35, h2h["matches"] / (h2h["matches"] + 12.0)) if h2h["matches"] > 0 else 0.0

    a_stats["base_srv_win"] = (1.0 - blend) * a_stats["base_srv_win"] + blend * h2h["a_srv_win"]
    a_stats["base_rcv_win"] = (1.0 - blend) * a_stats["base_rcv_win"] + blend * h2h["a_rcv_win"]

    player_a_params = _build_player_params(a_stats, sample_matches=int(a_stats["matches"]))
    player_b_params = _build_player_params(b_stats, sample_matches=int(b_stats["matches"]))

    matchup = MatchupParams(player_a=player_a_params, player_b=player_b_params, weights=weights)
    stats = MatchupStats(
        player_a=player_a,
        player_b=player_b,
        player_a_stats=a_stats,
        player_b_stats=b_stats,
        head_to_head=h2h,
        weights=weights,
    )
    return matchup, stats
