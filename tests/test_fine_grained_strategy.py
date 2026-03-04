from __future__ import annotations

from coach.data.adapters.local_csv import LocalCSVAdapter
from coach.data.stats_builder import build_matchup_params
from coach.service import BadmintonCoachService


def test_local_stats_include_fine_grained_metrics() -> None:
    adapter = LocalCSVAdapter()
    stats = adapter.get_player_params("viktor_axelsen", window=30)

    assert 0.01 <= float(stats["unforced_error_rate"]) <= 0.6
    assert 0.01 <= float(stats["return_pressure"]) <= 0.99
    assert 0.01 <= float(stats["clutch_point_win"]) <= 0.99


def test_strategy_candidate_generator_has_micro_steps_and_new_knobs(tmp_path) -> None:
    service = BadmintonCoachService(runs_root=tmp_path)
    baseline, _ = build_matchup_params(
        adapter=service.adapter,
        player_a_ref="Viktor Axelsen",
        player_b_ref="Kento Momota",
        window=30,
    )

    candidates = service._generate_candidates(baseline=baseline, l1_bound=0.35)
    assert candidates
    assert all("serve_short_delta" in c for c in candidates)
    assert all("attack_delta" in c for c in candidates)
    assert all("unforced_error_delta" in c for c in candidates)
    assert all("return_pressure_delta" in c for c in candidates)
    assert all("clutch_delta" in c for c in candidates)

    assert any(abs(c["serve_short_delta"]) == 0.01 for c in candidates)
    assert any(abs(c["attack_delta"]) == 0.01 for c in candidates)
    assert any(abs(c["unforced_error_delta"]) == 0.01 for c in candidates)
    assert any(abs(c["return_pressure_delta"]) == 0.01 for c in candidates)
    assert any(abs(c["clutch_delta"]) == 0.01 for c in candidates)
