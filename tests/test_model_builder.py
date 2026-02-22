from __future__ import annotations

from pathlib import Path

from coach.model.builder import build_matchup_model
from coach.model.params import InfluenceWeights, MatchupParams, PlayerParams, RallyStyleMix, ServeMix


def _make_params() -> MatchupParams:
    return MatchupParams(
        player_a=PlayerParams(
            player_id="axelsen",
            name="Viktor Axelsen",
            base_srv_win=0.59,
            base_rcv_win=0.54,
            serve_mix=ServeMix(short=0.67, flick=0.33),
            rally_style=RallyStyleMix(attack=0.56, neutral=0.28, safe=0.16),
            sample_matches=30,
        ),
        player_b=PlayerParams(
            player_id="momota",
            name="Kento Momota",
            base_srv_win=0.55,
            base_rcv_win=0.5,
            serve_mix=ServeMix(short=0.62, flick=0.38),
            rally_style=RallyStyleMix(attack=0.49, neutral=0.29, safe=0.22),
            sample_matches=30,
        ),
        weights=InfluenceWeights(w_short=0.04, w_attack=0.06, w_safe=0.05),
    )


def test_builder_injects_template_values(tmp_path: Path) -> None:
    result = build_matchup_model(
        params=_make_params(),
        template_name="badminton_rally_template.pcsp",
        out_path=tmp_path / "matchup.pcsp",
    )

    text = result.matchup_pcsp_path.read_text(encoding="utf-8")
    assert "{{" not in text
    assert "Viktor Axelsen" in text
    assert "Kento Momota" in text
    assert "#assert BadmintonMatch reaches A_WinsMatch with prob;" in text
    assert result.params_json_path.exists()
