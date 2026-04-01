from __future__ import annotations

from pathlib import Path

import coach.service as service_module
from coach.data.stats_builder import estimate_influence_weights
from coach.service import StrategyAdjustment


def test_get_player_params_returns_detached_cached_copy(csv_adapter) -> None:
    first = csv_adapter.get_player_params("viktor_axelsen", window=30)
    first["base_srv_win"] = 0.01
    first["serve_mix"]["short"] = 0.01

    second = csv_adapter.get_player_params("viktor_axelsen", window=30)

    assert second["base_srv_win"] != 0.01
    assert second["serve_mix"]["short"] != 0.01


def test_estimate_influence_weights_reuses_adapter_cache(csv_adapter, monkeypatch) -> None:
    baseline = estimate_influence_weights(csv_adapter)

    def _fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("influence weights should come from cache")

    monkeypatch.setattr(csv_adapter, "get_player_params", _fail)

    cached = estimate_influence_weights(csv_adapter)
    assert cached == baseline


def test_execute_pat_uses_runner_probability_without_reparsing(
    coach_service,
    monkeypatch,
    tmp_path: Path,
) -> None:
    pcsp_path = tmp_path / "matchup.pcsp"
    pcsp_path.write_text("#assert M reaches X with prob;\n", encoding="utf-8")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    stdout_path = run_dir / "pat_stdout.txt"
    stderr_path = run_dir / "pat_stderr.txt"
    pat_out_path = run_dir / "pat_output.txt"

    def _fake_run_pat(**kwargs):  # type: ignore[no-untyped-def]
        return {
            "ok": True,
            "returncode": 0,
            "cmd": ["mock_pat"],
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "pat_out_path": str(pat_out_path),
            "probability": 0.61,
        }

    monkeypatch.setattr(service_module, "run_pat", _fake_run_pat)
    monkeypatch.setattr(
        service_module,
        "read_pat_output",
        lambda path: (_ for _ in ()).throw(AssertionError("output should not be re-read when probability exists")),
    )

    result = coach_service._execute_pat(
        pcsp_path=pcsp_path,
        run_dir=run_dir,
        mode="mock",
        pat_path=None,
        timeout_s=30,
    )

    assert result.ok is True
    assert result.probability == 0.61


def test_select_candidates_for_budget_prefers_proxy_score_over_smallest_l1(
    coach_service,
    sample_matchup_params,
    monkeypatch,
) -> None:
    low_impact = StrategyAdjustment(serve_short_delta=0.01, l1_change=0.02)
    high_impact = StrategyAdjustment(attack_delta=0.08, l1_change=0.16)

    monkeypatch.setattr(
        coach_service,
        "_estimate_candidate_probability",
        lambda params: params.player_a.rally_style.attack,
    )

    selected = coach_service._select_candidates_for_budget(
        baseline=sample_matchup_params,
        candidates=[low_impact, high_impact],
        budget=1,
    )

    assert len(selected) == 1
    assert selected[0][0] == high_impact


def test_effective_probabilities_account_for_opponent_phase_baselines(sample_matchup_params) -> None:
    stronger_receiver = sample_matchup_params.model_copy(
        update={
            "player_b": sample_matchup_params.player_b.model_copy(update={"base_rcv_win": 0.62}),
        }
    )
    weaker_receiver = sample_matchup_params.model_copy(
        update={
            "player_b": sample_matchup_params.player_b.model_copy(update={"base_rcv_win": 0.40}),
        }
    )
    stronger_server = sample_matchup_params.model_copy(
        update={
            "player_b": sample_matchup_params.player_b.model_copy(update={"base_srv_win": 0.64}),
        }
    )
    weaker_server = sample_matchup_params.model_copy(
        update={
            "player_b": sample_matchup_params.player_b.model_copy(update={"base_srv_win": 0.44}),
        }
    )

    assert weaker_receiver.effective_probabilities()["pA_srv_win"] > stronger_receiver.effective_probabilities()[
        "pA_srv_win"
    ]
    assert weaker_server.effective_probabilities()["pA_rcv_win"] > stronger_server.effective_probabilities()[
        "pA_rcv_win"
    ]
