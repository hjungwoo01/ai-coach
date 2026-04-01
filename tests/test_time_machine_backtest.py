from __future__ import annotations

from pathlib import Path

import pandas as pd

from coach.backtest.data import SnapshotCSVAdapter, prepare_chronological_matches
from coach.backtest.engine import BacktestConfig, TimeMachineBacktester
from coach.backtest.feature_contract import extract_pcsp_feature_contract
from coach.backtest.models import RollingPlattCalibrator


def _synthetic_match(
    *,
    date: str,
    player_a: str,
    player_b: str,
    winner_id: str,
    round_name: str,
) -> dict[str, object]:
    return {
        "date": date,
        "playerA_id": player_a,
        "playerB_id": player_b,
        "winner_id": winner_id,
        "round": round_name,
        "a_serve_rallies": 24,
        "b_serve_rallies": 22,
        "a_serve_wins": 14,
        "b_serve_wins": 11,
        "a_short_serve_rate": 0.62,
        "b_short_serve_rate": 0.56,
        "a_flick_serve_rate": 0.38,
        "b_flick_serve_rate": 0.44,
        "a_attack_rate": 0.48,
        "b_attack_rate": 0.43,
        "a_neutral_rate": 0.27,
        "b_neutral_rate": 0.31,
        "a_safe_rate": 0.25,
        "b_safe_rate": 0.26,
        "a_points": 42,
        "b_points": 37,
        "avg_rally_len": 8.5,
        "long_rally_share": 0.29,
        "a_short_serve_win_rate": 0.61,
        "b_short_serve_win_rate": 0.53,
        "a_long_serve_win_rate": 0.54,
        "b_long_serve_win_rate": 0.47,
        "a_short_serve_samples": 16,
        "b_short_serve_samples": 15,
        "a_long_serve_samples": 8,
        "b_long_serve_samples": 7,
        "a_backhand_rate": 0.19,
        "b_backhand_rate": 0.24,
        "a_aroundhead_rate": 0.12,
        "b_aroundhead_rate": 0.08,
        "a_net_error_lost_rate": 0.07,
        "b_net_error_lost_rate": 0.09,
        "a_out_error_lost_rate": 0.11,
        "b_out_error_lost_rate": 0.12,
    }


def test_prepare_chronological_matches_uses_previous_day_policy_without_timestamps() -> None:
    matches = pd.DataFrame(
        [
            {
                "date": "2024-01-10",
                "playerA_id": "a",
                "playerB_id": "b",
                "winner_id": "a",
                "round": "Final",
            },
            {
                "date": "2024-01-10",
                "playerA_id": "c",
                "playerB_id": "d",
                "winner_id": "d",
                "round": "Semi-finals",
            },
        ]
    )

    prepared = prepare_chronological_matches(matches)

    assert set(prepared["snapshot_policy"]) == {"previous_day_close"}
    assert set(prepared["cutoff_key"]) == {"2024-01-09"}


def test_snapshot_adapter_filters_matches_strictly_to_cutoff_timestamp() -> None:
    players = pd.DataFrame(
        [
            {"player_id": "a", "name": "A", "country": "X", "handedness": "R"},
            {"player_id": "b", "name": "B", "country": "Y", "handedness": "R"},
        ]
    )
    matches = pd.DataFrame(
        [
            {
                "date": "2024-01-10",
                "match_start_ts": "2024-01-10T09:00:00",
                "playerA_id": "a",
                "playerB_id": "b",
                "winner_id": "a",
            },
            {
                "date": "2024-01-10",
                "match_start_ts": "2024-01-10T12:00:00",
                "playerA_id": "a",
                "playerB_id": "b",
                "winner_id": "b",
            },
        ]
    )

    adapter = SnapshotCSVAdapter(
        players_df=players,
        matches_df=matches,
        cutoff=pd.Timestamp("2024-01-10T10:00:00"),
        timestamp_column="match_start_ts",
    )

    assert len(adapter.matches_df) == 1
    assert adapter.matches_df.iloc[0]["winner_id"] == "a"


def test_feature_contract_covers_model_fields() -> None:
    contract = extract_pcsp_feature_contract()
    player_paths = {spec.field_path for spec in contract.player_features}
    weight_paths = {spec.field_path for spec in contract.weight_features}

    assert "player.base_srv_win" in player_paths
    assert "player.recent_form" in player_paths
    assert "player.rest_days" in player_paths
    assert "player.serve_mix.short" in player_paths
    assert "player.rally_style.attack" in player_paths
    assert "weights.w_short" in weight_paths
    assert "weights.w_recent_form" in weight_paths
    assert "weights.w_rest" in weight_paths
    assert "weights.w_serve_type" in weight_paths
    assert "pA_srv_win" in contract.template_context_fields
    assert "recent_form_A" in contract.template_context_fields
    assert "rest_days_A" in contract.template_context_fields


def test_snapshot_adapter_supports_cold_start_priors() -> None:
    players = pd.DataFrame(
        [
            {"player_id": "a", "name": "A", "country": "X", "handedness": "L"},
            {"player_id": "b", "name": "B", "country": "Y", "handedness": "R"},
        ]
    )
    matches = pd.DataFrame(columns=["date", "playerA_id", "playerB_id", "winner_id"])

    adapter = SnapshotCSVAdapter(
        players_df=players,
        matches_df=matches,
        cutoff=pd.Timestamp("2024-01-10T10:00:00"),
        timestamp_column=None,
    )

    params = adapter.get_player_params("a", allow_cold_start=True)

    assert params["matches"] == 0
    assert params["name"] == "A"
    assert params["handedness_flag"] == 1.0
    assert 0.0 <= params["base_srv_win"] <= 1.0
    assert 0.0 <= params["recent_form"] <= 1.0
    assert params["rest_days"] == 7.0


def test_rolling_platt_calibrator_corrects_systematic_underconfidence() -> None:
    calibrator = RollingPlattCalibrator(min_samples=10, max_iterations=100)

    for _ in range(40):
        calibrator.update(0.2, 1)

    assert calibrator.transform(0.2) > 0.2


def test_time_machine_backtester_runs_on_sample_dataset(tmp_path: Path) -> None:
    backtester = TimeMachineBacktester.from_paths(
        players_path="coach/data/sample_players.csv",
        matches_path="coach/data/sample_matches.csv",
        config=BacktestConfig(
            mode="mock",
            window=30,
            start_date="2020-01-01",
            limit=12,
        ),
        runs_root=tmp_path,
    )

    report = backtester.run()

    assert report.metrics["matches_total"] == 12
    assert report.metrics["matches_scored"] > 0
    assert set(report.metrics["overall"]) == {"pcsp", "elo", "recent_form"}
    assert 0.0 <= report.metrics["overall"]["pcsp"]["calibrated"]["accuracy"] <= 1.0
    assert report.metrics["rankings"]["best_calibrated_log_loss"] in {"pcsp", "elo", "recent_form"}
    assert report.artifacts.predictions_csv.exists()
    assert report.artifacts.model_metrics_csv.exists()
    assert report.artifacts.metrics_json.exists()
    assert report.artifacts.summary_txt.exists()


def test_time_machine_backtester_scores_matches_with_cold_start_players(tmp_path: Path) -> None:
    players = pd.DataFrame(
        [
            {"player_id": "a", "name": "Player A", "country": "X", "handedness": "R"},
            {"player_id": "b", "name": "Player B", "country": "Y", "handedness": "L"},
            {"player_id": "c", "name": "Player C", "country": "Z", "handedness": "R"},
        ]
    )
    matches = pd.DataFrame(
        [
            _synthetic_match(
                date="2024-01-01",
                player_a="a",
                player_b="b",
                winner_id="a",
                round_name="Round 1",
            ),
            _synthetic_match(
                date="2024-01-02",
                player_a="c",
                player_b="a",
                winner_id="a",
                round_name="Quarter-finals",
            ),
        ]
    )

    players_path = tmp_path / "players.csv"
    matches_path = tmp_path / "matches.csv"
    players.to_csv(players_path, index=False)
    matches.to_csv(matches_path, index=False)

    backtester = TimeMachineBacktester.from_paths(
        players_path=players_path,
        matches_path=matches_path,
        config=BacktestConfig(
            mode="mock",
            window=30,
        ),
        runs_root=tmp_path / "runs",
    )

    report = backtester.run()

    assert report.metrics["matches_total"] == 2
    assert report.metrics["matches_scored"] == 2
    assert report.metrics["matches_skipped"] == 0
    assert report.metrics["coverage"] == 1.0
