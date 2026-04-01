from __future__ import annotations

from pathlib import Path

import pytest

from coach.data.adapters.local_csv import LocalCSVAdapter
from coach.model.params import (
    InfluenceWeights,
    MatchupParams,
    PlayerParams,
    RallyStyleMix,
    ServeMix,
)
from coach.service import BadmintonCoachService


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    return tmp_path / "runs"


@pytest.fixture
def csv_adapter() -> LocalCSVAdapter:
    return LocalCSVAdapter()


@pytest.fixture
def coach_service(runs_root: Path, csv_adapter: LocalCSVAdapter) -> BadmintonCoachService:
    return BadmintonCoachService(adapter=csv_adapter, runs_root=runs_root)


@pytest.fixture
def sample_matchup_params() -> MatchupParams:
    return MatchupParams(
        player_a=PlayerParams(
            player_id="player_a",
            name="Player A",
            base_srv_win=0.61,
            base_rcv_win=0.56,
            unforced_error_rate=0.14,
            return_pressure=0.58,
            clutch_point_win=0.55,
            short_serve_skill=0.63,
            long_serve_skill=0.59,
            rally_tolerance=0.57,
            net_error_rate=0.08,
            out_error_rate=0.11,
            backhand_rate=0.22,
            aroundhead_rate=0.18,
            handedness_flag=1.0,
            reliability=0.95,
            serve_mix=ServeMix(short=0.68, flick=0.32),
            rally_style=RallyStyleMix(attack=0.52, neutral=0.29, safe=0.19),
            sample_matches=32,
        ),
        player_b=PlayerParams(
            player_id="player_b",
            name="Player B",
            base_srv_win=0.53,
            base_rcv_win=0.49,
            unforced_error_rate=0.21,
            return_pressure=0.47,
            clutch_point_win=0.48,
            short_serve_skill=0.49,
            long_serve_skill=0.46,
            rally_tolerance=0.45,
            net_error_rate=0.14,
            out_error_rate=0.16,
            backhand_rate=0.37,
            aroundhead_rate=0.07,
            handedness_flag=0.0,
            reliability=0.9,
            serve_mix=ServeMix(short=0.58, flick=0.42),
            rally_style=RallyStyleMix(attack=0.41, neutral=0.31, safe=0.28),
            sample_matches=28,
        ),
        weights=InfluenceWeights(
            w_short=0.04,
            w_attack=0.06,
            w_safe=0.05,
            w_ue=0.08,
            w_return_pressure=0.07,
            w_clutch=0.05,
            w_serve_type=0.03,
            w_rally_tolerance=0.02,
            w_error_profile=0.03,
            w_handedness=0.01,
            w_backhand=0.01,
            w_aroundhead=0.01,
        ),
    )
