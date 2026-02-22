from __future__ import annotations

from pathlib import Path

from coach.agent.planner import AgentExecutor
from coach.service import BadmintonCoachService


def test_predict_end_to_end_mock(tmp_path: Path) -> None:
    service = BadmintonCoachService(runs_root=tmp_path)
    result = service.predict(
        player_a="Viktor Axelsen",
        player_b="Kento Momota",
        window=30,
        mode="mock",
    )

    assert 0.0 <= result.probability <= 1.0
    assert (result.run_dir / "matchup.pcsp").exists()
    assert (result.run_dir / "pat_stdout.txt").exists()
    assert (result.run_dir / "prediction_result.json").exists()


def test_strategy_end_to_end_mock(tmp_path: Path) -> None:
    service = BadmintonCoachService(runs_root=tmp_path)
    result = service.strategy(
        player_a="Viktor Axelsen",
        player_b="Kento Momota",
        window=30,
        mode="mock",
        budget=30,
    )

    assert 0.0 <= result.baseline_probability <= 1.0
    assert 0.0 <= result.improved_probability <= 1.0
    assert len(result.top_alternatives) >= 1
    assert (result.run_dir / "strategy_result.json").exists()


def test_agent_query_mock_mode(tmp_path: Path) -> None:
    service = BadmintonCoachService(runs_root=tmp_path)
    executor = AgentExecutor(service=service)
    out = executor.run(
        "What is the expected winning percentage for Viktor Axelsen vs Kento Momota?",
        mode="mock",
        window=30,
    )

    assert out.plan.task_type == "prediction"
    assert "probability" in out.payload
    assert "PAT" in out.answer or "probability" in out.answer


def test_agent_strategy_query_mock_mode(tmp_path: Path) -> None:
    service = BadmintonCoachService(runs_root=tmp_path)
    executor = AgentExecutor(service=service)
    out = executor.run(
        "What adjustment should Viktor Axelsen make to beat Kento Momota?",
        mode="mock",
        window=30,
        budget=25,
    )

    assert out.plan.task_type == "strategy"
    assert "delta" in out.payload
    assert isinstance(out.payload["improved_probability"], float)
    assert isinstance(out.payload["baseline_probability"], float)
