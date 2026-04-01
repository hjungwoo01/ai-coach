from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

import scripts.build_real_data as brd


@dataclass(frozen=True)
class _RawMatchFixture:
    set_dir: Path
    winner: str
    loser: str


def _base_stroke_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "rally": 1,
        "ball_round": 1.0,
        "time": 0.0,
        "frame_num": 1,
        "roundscore_A": 0,
        "roundscore_B": 0,
        "player": "A",
        "server": "A",
        "type": "發短球",
        "aroundhead": float("nan"),
        "backhand": float("nan"),
        "hit_height": float("nan"),
        "hit_area": float("nan"),
        "hit_x": float("nan"),
        "hit_y": float("nan"),
        "landing_height": float("nan"),
        "landing_area": float("nan"),
        "landing_x": float("nan"),
        "landing_y": float("nan"),
        "lose_reason": float("nan"),
        "win_reason": float("nan"),
        "getpoint_player": float("nan"),
        "flaw": float("nan"),
        "player_location_area": float("nan"),
        "player_location_x": float("nan"),
        "player_location_y": float("nan"),
        "opponent_location_area": float("nan"),
        "opponent_location_x": float("nan"),
        "opponent_location_y": float("nan"),
        "db": float("nan"),
    }
    row.update(overrides)
    return row


def _write_raw_fixture(tmp_path: Path) -> _RawMatchFixture:
    set_dir = tmp_path / "set"
    match_dir = set_dir / "test_match_1"
    match_dir.mkdir(parents=True, exist_ok=True)

    winner = "Kento MOMOTA"
    loser = "CHOU Tien Chen"

    match_df = pd.DataFrame(
        [
            {
                "id": 1,
                "video": "test_match_1",
                "tournament": "Unit Test Open",
                "round": "Finals",
                "year": 2021,
                "month": 1,
                "day": 31,
                "set": 1,
                "duration": 65,
                "winner": winner,
                "loser": loser,
                "downcourt": 0,
                "url": "https://example.com/match",
            }
        ]
    )
    match_df.to_csv(set_dir / "match.csv", index=False)

    set_rows = [
        # Rally 1 (short serve by A, A wins, B out-error)
        _base_stroke_row(
            rally=1,
            ball_round=1.0,
            player="A",
            server="A",
            type="發短球",
            backhand=1.0,
            aroundhead=float("nan"),
            roundscore_A=0,
            roundscore_B=0,
        ),
        _base_stroke_row(
            rally=1,
            ball_round=2.0,
            player="B",
            server="A",
            type="長球",
            backhand=float("nan"),
            aroundhead=1.0,
            getpoint_player="A",
            lose_reason="出界",
            roundscore_A=1,
            roundscore_B=0,
        ),
        # Rally 2 (long serve by B, B wins, A net-error)
        _base_stroke_row(
            rally=2,
            ball_round=1.0,
            player="B",
            server="B",
            type="發長球",
            backhand=float("nan"),
            aroundhead=float("nan"),
            roundscore_A=1,
            roundscore_B=0,
        ),
        _base_stroke_row(
            rally=2,
            ball_round=8.0,
            player="A",
            server="B",
            type="長球",
            backhand=1.0,
            aroundhead=float("nan"),
            getpoint_player="B",
            lose_reason="掛網",
            roundscore_A=1,
            roundscore_B=1,
        ),
        # Rally 3 (short serve by A, A wins, B net-error)
        _base_stroke_row(
            rally=3,
            ball_round=1.0,
            player="A",
            server="A",
            type="發短球",
            backhand=float("nan"),
            aroundhead=1.0,
            roundscore_A=1,
            roundscore_B=1,
        ),
        _base_stroke_row(
            rally=3,
            ball_round=3.0,
            player="B",
            server="A",
            type="長球",
            backhand=float("nan"),
            aroundhead=float("nan"),
            getpoint_player="A",
            lose_reason="未過網",
            roundscore_A=2,
            roundscore_B=1,
        ),
        # Rally 4 (long serve by B, A wins, B out-error)
        _base_stroke_row(
            rally=4,
            ball_round=1.0,
            player="B",
            server="B",
            type="發長球",
            backhand=1.0,
            aroundhead=float("nan"),
            roundscore_A=2,
            roundscore_B=1,
        ),
        _base_stroke_row(
            rally=4,
            ball_round=9.0,
            player="B",
            server="B",
            type="長球",
            backhand=float("nan"),
            aroundhead=float("nan"),
            getpoint_player="A",
            lose_reason="出界",
            roundscore_A=3,
            roundscore_B=1,
        ),
    ]
    pd.DataFrame(set_rows).to_csv(match_dir / "set1.csv", index=False)
    return _RawMatchFixture(set_dir=set_dir, winner=winner, loser=loser)


def test_process_set_csv_rally_length_and_long_rally_threshold(tmp_path: Path) -> None:
    fixture = _write_raw_fixture(tmp_path)
    result = brd.process_set_csv(fixture.set_dir / "test_match_1" / "set1.csv")
    assert result["rallies"]["count"] == 4
    assert result["rallies"]["long_count"] == 2
    assert result["rallies"]["len_sum"] == pytest.approx(22.0)


def test_build_data_enriched_serve_and_error_fields(tmp_path: Path) -> None:
    fixture = _write_raw_fixture(tmp_path)
    _, matches_df = brd.build_data(fixture.set_dir)
    row = matches_df.iloc[0]

    assert row["tournament"] == "Unit Test Open"
    assert row["round"] == "Finals"
    assert int(row["duration_min"]) == 65
    assert int(row["match_sets"]) == 1

    assert float(row["avg_rally_len"]) == pytest.approx(5.5, abs=1e-4)
    assert float(row["long_rally_share"]) == pytest.approx(0.5, abs=1e-4)

    assert int(row["a_short_serve_samples"]) == 2
    assert int(row["a_long_serve_samples"]) == 0
    assert float(row["a_short_serve_win_rate"]) == pytest.approx(1.0, abs=1e-4)
    assert float(row["a_long_serve_win_rate"]) == pytest.approx(0.5, abs=1e-4)

    assert int(row["b_short_serve_samples"]) == 0
    assert int(row["b_long_serve_samples"]) == 2
    assert float(row["b_short_serve_win_rate"]) == pytest.approx(0.5, abs=1e-4)
    assert float(row["b_long_serve_win_rate"]) == pytest.approx(0.5, abs=1e-4)

    assert float(row["a_net_error_lost_rate"]) == pytest.approx(1.0, abs=1e-4)
    assert float(row["a_out_error_lost_rate"]) == pytest.approx(0.0, abs=1e-4)
    assert float(row["b_net_error_lost_rate"]) == pytest.approx(1.0 / 3.0, abs=1e-4)
    assert float(row["b_out_error_lost_rate"]) == pytest.approx(2.0 / 3.0, abs=1e-4)


def test_build_data_backhand_aroundhead_nan_handling(tmp_path: Path) -> None:
    fixture = _write_raw_fixture(tmp_path)
    _, matches_df = brd.build_data(fixture.set_dir)
    row = matches_df.iloc[0]

    assert float(row["a_backhand_rate"]) == pytest.approx(2.0 / 3.0, abs=1e-4)
    assert float(row["a_aroundhead_rate"]) == pytest.approx(1.0 / 3.0, abs=1e-4)
    assert float(row["b_backhand_rate"]) == pytest.approx(1.0 / 5.0, abs=1e-4)
    assert float(row["b_aroundhead_rate"]) == pytest.approx(1.0 / 5.0, abs=1e-4)


def test_build_data_swap_maps_new_side_columns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fixture = _write_raw_fixture(tmp_path)

    class _AlwaysSwapRandom:
        def __init__(self, seed: int) -> None:
            del seed

        def random(self) -> float:
            return 0.0

    monkeypatch.setattr(brd.random, "Random", _AlwaysSwapRandom)
    _, matches_df = brd.build_data(fixture.set_dir)
    row = matches_df.iloc[0]

    assert row["playerA_id"] == brd.make_player_id(fixture.loser)
    assert row["playerB_id"] == brd.make_player_id(fixture.winner)
    assert int(row["a_short_serve_samples"]) == 0
    assert int(row["b_short_serve_samples"]) == 2
    assert float(row["a_backhand_rate"]) == pytest.approx(1.0 / 5.0, abs=1e-4)
    assert float(row["b_backhand_rate"]) == pytest.approx(2.0 / 3.0, abs=1e-4)


def test_validate_rejects_out_of_range_new_rate(tmp_path: Path) -> None:
    fixture = _write_raw_fixture(tmp_path)
    players_df, matches_df = brd.build_data(fixture.set_dir)
    matches_df.loc[0, "a_short_serve_win_rate"] = 1.2

    with pytest.raises(ValueError):
        brd.validate(players_df, matches_df)


def test_validate_rejects_negative_or_non_integer_sample_counts(tmp_path: Path) -> None:
    fixture = _write_raw_fixture(tmp_path)
    players_df, matches_df = brd.build_data(fixture.set_dir)
    matches_df = matches_df.astype({"b_long_serve_samples": "float64"})
    matches_df.loc[0, "a_short_serve_samples"] = -1
    matches_df.loc[0, "b_long_serve_samples"] = 2.5

    with pytest.raises(ValueError):
        brd.validate(players_df, matches_df)
