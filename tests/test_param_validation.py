from __future__ import annotations

import pytest
from pydantic import ValidationError

from coach.model.params import InfluenceWeights, MatchupParams, PlayerParams, RallyStyleMix, ServeMix


def make_player(player_id: str, name: str) -> PlayerParams:
    return PlayerParams(
        player_id=player_id,
        name=name,
        base_srv_win=0.57,
        base_rcv_win=0.52,
        serve_mix=ServeMix(short=0.65, flick=0.35),
        rally_style=RallyStyleMix(attack=0.5, neutral=0.3, safe=0.2),
        sample_matches=20,
    )


def test_serve_mix_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        ServeMix(short=0.8, flick=0.3)


def test_rally_style_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        RallyStyleMix(attack=0.5, neutral=0.4, safe=0.2)


def test_matchup_constraints_and_adjustments() -> None:
    params = MatchupParams(
        player_a=make_player("a", "Player A"),
        player_b=make_player("b", "Player B"),
        weights=InfluenceWeights(w_short=0.04, w_attack=0.06, w_safe=0.05),
    )
    eff = params.effective_probabilities()
    assert 0.0 < eff["pA_srv_win"] < 1.0
    assert 0.0 < eff["pA_rcv_win"] < 1.0

    adjusted = params.with_adjustments(serve_short_delta=0.05, attack_delta=-0.1)
    assert abs(adjusted.player_a.serve_mix.short + adjusted.player_a.serve_mix.flick - 1.0) < 1e-9
    assert (
        abs(
            adjusted.player_a.rally_style.attack
            + adjusted.player_a.rally_style.neutral
            + adjusted.player_a.rally_style.safe
            - 1.0
        )
        < 1e-9
    )


def test_best_of_must_be_odd() -> None:
    with pytest.raises(ValidationError):
        MatchupParams(
            player_a=make_player("a", "Player A"),
            player_b=make_player("b", "Player B"),
            weights=InfluenceWeights(w_short=0.04, w_attack=0.06, w_safe=0.05),
            best_of=2,
        )
