#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

from coach.model.params import (
    InfluenceWeights,
    MatchupParams,
    PlayerParams,
    RallyStyleMix,
    ServeMix,
)
from coach.pat.mock_pat import mock_probability
from coach.service import BadmintonCoachService


def _make_reference_params() -> MatchupParams:
    return MatchupParams(
        player_a=PlayerParams(
            player_id="player_a",
            name="Player A",
            base_srv_win=0.62,
            base_rcv_win=0.57,
            unforced_error_rate=0.13,
            return_pressure=0.59,
            clutch_point_win=0.56,
            short_serve_skill=0.64,
            long_serve_skill=0.61,
            rally_tolerance=0.58,
            net_error_rate=0.07,
            out_error_rate=0.1,
            backhand_rate=0.2,
            aroundhead_rate=0.19,
            handedness_flag=1.0,
            reliability=0.95,
            serve_mix=ServeMix(short=0.7, flick=0.3),
            rally_style=RallyStyleMix(attack=0.54, neutral=0.28, safe=0.18),
            sample_matches=30,
        ),
        player_b=PlayerParams(
            player_id="player_b",
            name="Player B",
            base_srv_win=0.51,
            base_rcv_win=0.47,
            unforced_error_rate=0.22,
            return_pressure=0.46,
            clutch_point_win=0.47,
            short_serve_skill=0.48,
            long_serve_skill=0.45,
            rally_tolerance=0.44,
            net_error_rate=0.15,
            out_error_rate=0.18,
            backhand_rate=0.39,
            aroundhead_rate=0.08,
            handedness_flag=0.0,
            reliability=0.9,
            serve_mix=ServeMix(short=0.57, flick=0.43),
            rally_style=RallyStyleMix(attack=0.4, neutral=0.31, safe=0.29),
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


def validate_reference_model_thresholds() -> None:
    base = _make_reference_params()
    improved = base.with_adjustments(
        serve_short_delta=0.03,
        attack_delta=0.04,
        unforced_error_delta=-0.02,
        return_pressure_delta=0.03,
        clutch_delta=0.02,
    )
    weakened = base.model_copy(
        update={
            "player_b": base.player_b.model_copy(update={"base_rcv_win": 0.62, "base_srv_win": 0.63}),
        }
    )

    base_probability = mock_probability(base.to_template_context())
    improved_probability = mock_probability(improved.to_template_context())
    weakened_probability = mock_probability(weakened.to_template_context())

    assert 0.65 <= base_probability <= 0.75, base_probability
    assert improved_probability - base_probability >= 0.02, (improved_probability, base_probability)
    assert base_probability - weakened_probability >= 0.05, (weakened_probability, base_probability)


def validate_mock_service_thresholds() -> None:
    with tempfile.TemporaryDirectory(prefix="coach_inference_") as tmp_dir:
        service = BadmintonCoachService(runs_root=Path(tmp_dir))
        prediction = service.predict(
            player_a="Viktor Axelsen",
            player_b="Kento Momota",
            mode="mock",
            window=30,
        )
        strategy = service.strategy(
            player_a="Viktor Axelsen",
            player_b="Kento Momota",
            mode="mock",
            window=30,
            budget=20,
        )

    assert 0.4 <= prediction.probability <= 0.6, prediction.probability
    assert abs(strategy.baseline_probability - prediction.probability) <= 1e-9
    assert strategy.improved_probability - strategy.baseline_probability >= 0.01
    assert strategy.delta >= 0.01


def main() -> None:
    validate_reference_model_thresholds()
    validate_mock_service_thresholds()
    print("Inference thresholds validated.")


if __name__ == "__main__":
    main()
