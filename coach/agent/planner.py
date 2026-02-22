from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from coach.agent.llm_client import LLMClient
from coach.agent.schemas import Plan, ToolInstruction
from coach.data.adapters.local_csv import LocalCSVAdapter
from coach.service import BadmintonCoachService


@dataclass(frozen=True)
class ExecutionResult:
    plan: Plan
    tool_trace: list[dict[str, Any]]
    answer: str
    payload: dict[str, Any]


class Planner:
    """Creates tool-call plans from user queries (LLM first, heuristic fallback)."""

    def __init__(self, adapter: LocalCSVAdapter, llm_client: LLMClient | None = None) -> None:
        self.adapter = adapter
        self.llm_client = llm_client

    def create_plan(self, user_query: str, mode: str = "mock", window: int = 30, budget: int = 60) -> Plan:
        if self.llm_client is not None:
            llm_plan = self.llm_client.plan(user_query)
            if llm_plan is not None:
                try:
                    return Plan.model_validate(llm_plan)
                except Exception:
                    pass

        task_type = self._detect_task_type(user_query)
        players = self._extract_players(user_query)
        constraints = self._extract_constraints(user_query)

        if task_type == "prediction":
            tool_calls = [
                ToolInstruction(tool="ResolvePlayers", arguments={"names": players}),
                ToolInstruction(
                    tool="LoadStats",
                    arguments={
                        "playerA_id": "$playerA_id",
                        "playerB_id": "$playerB_id",
                        "window": window,
                        "source": "local",
                    },
                ),
                ToolInstruction(
                    tool="BuildModel",
                    arguments={
                        "params": "$params",
                        "template_name": "badminton_rally_template.pcsp",
                        "out_path": "$run_dir/matchup.pcsp",
                    },
                ),
                ToolInstruction(
                    tool="RunPAT",
                    arguments={
                        "pcsp_path": "$pcsp_path",
                        "pat_path": None,
                        "mode": mode,
                        "timeout_s": 60,
                    },
                ),
                ToolInstruction(
                    tool="SummarizeResults",
                    arguments={
                        "pat_outputs": "$pat_outputs",
                        "question": user_query,
                        "constraints": constraints,
                    },
                ),
            ]
            return Plan(
                task_type="prediction",
                analysis_type="reachability",
                players=players,
                constraints=constraints,
                tool_calls=tool_calls,
            )

        tool_calls = [
            ToolInstruction(tool="ResolvePlayers", arguments={"names": players}),
            ToolInstruction(
                tool="LoadStats",
                arguments={
                    "playerA_id": "$playerA_id",
                    "playerB_id": "$playerB_id",
                    "window": window,
                    "source": "local",
                },
            ),
            ToolInstruction(
                tool="BatchSensitivity",
                arguments={
                    "base_params": "$params",
                    "search_space": {
                        "serve_mix_A.short": [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2],
                        "rally_style_A.attack": [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2],
                        "l1_bound": 0.3,
                    },
                    "budget": budget,
                    "objective": "maximize_A_win",
                },
            ),
            ToolInstruction(
                tool="SummarizeResults",
                arguments={
                    "pat_outputs": "$pat_outputs",
                    "question": user_query,
                    "constraints": constraints,
                },
            ),
        ]

        return Plan(
            task_type="strategy",
            analysis_type="sensitivity",
            players=players,
            constraints=constraints,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _detect_task_type(query: str) -> str:
        q = query.lower()
        strategy_markers = [
            "strategy",
            "adjust",
            "improve",
            "beat",
            "change",
            "optimiz",
            "what should",
            "recommend",
            "tactic",
        ]
        return "strategy" if any(marker in q for marker in strategy_markers) else "prediction"

    def _extract_players(self, query: str) -> list[str]:
        names = self.adapter.players_df["name"].tolist()
        lower = query.lower()

        hits: list[tuple[int, str]] = []
        for name in names:
            idx = lower.find(name.lower())
            if idx >= 0:
                hits.append((idx, name))

        if len(hits) >= 2:
            hits.sort(key=lambda item: item[0])
            return [hits[0][1], hits[1][1]]

        surname_hits: list[tuple[int, str]] = []
        for name in names:
            surname = name.split()[-1].lower()
            idx = lower.find(surname)
            if idx >= 0:
                surname_hits.append((idx, name))

        if len(surname_hits) >= 2:
            surname_hits.sort(key=lambda item: item[0])
            return [surname_hits[0][1], surname_hits[1][1]]

        vs_match = re.search(r"(.+?)\s+vs\.?\s+(.+)", query, flags=re.IGNORECASE)
        if vs_match:
            left = vs_match.group(1).strip(" ?!.,")
            right = vs_match.group(2).strip(" ?!.,")
            try:
                a = self.adapter.resolve_player(left).name
                b = self.adapter.resolve_player(right).name
                return [a, b]
            except Exception:
                pass

        between_match = re.search(r"between\s+(.+?)\s+and\s+(.+)", query, flags=re.IGNORECASE)
        if between_match:
            left = between_match.group(1).strip(" ?!.,")
            right = between_match.group(2).strip(" ?!.,")
            try:
                a = self.adapter.resolve_player(left).name
                b = self.adapter.resolve_player(right).name
                return [a, b]
            except Exception:
                pass

        raise ValueError(
            "Could not identify both players from query. "
            "Try wording like: 'Viktor Axelsen vs Kento Momota'."
        )

    @staticmethod
    def _extract_constraints(query: str) -> list[str]:
        q = query.lower()
        constraints: list[str] = []
        if "aggress" in q:
            constraints.append("emphasize aggression")
        if "safe" in q or "error" in q:
            constraints.append("control unforced errors")
        if "serve" in q:
            constraints.append("serve mix considered")
        return constraints


class AgentExecutor:
    """Executes planned tool calls via deterministic Python tools."""

    def __init__(
        self,
        service: BadmintonCoachService | None = None,
        planner: Planner | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.service = service or BadmintonCoachService()
        self.adapter = self.service.adapter
        self.llm_client = llm_client
        self.planner = planner or Planner(adapter=self.adapter, llm_client=llm_client)

    def run(self, user_query: str, mode: str = "mock", window: int = 30, budget: int = 60) -> ExecutionResult:
        plan = self.planner.create_plan(user_query, mode=mode, window=window, budget=budget)
        player_a, player_b = plan.players

        tool_trace: list[dict[str, Any]] = []
        a_resolved = self.adapter.resolve_player(player_a)
        b_resolved = self.adapter.resolve_player(player_b)
        tool_trace.append(
            {
                "tool": "ResolvePlayers",
                "input": {"names": [player_a, player_b]},
                "output": {
                    "playerA_id": a_resolved.player_id,
                    "playerA_name": a_resolved.name,
                    "playerB_id": b_resolved.player_id,
                    "playerB_name": b_resolved.name,
                },
            }
        )

        if plan.task_type == "prediction":
            prediction = self.service.predict(
                player_a=a_resolved.player_id,
                player_b=b_resolved.player_id,
                window=window,
                mode=mode,
            )

            tool_trace.append(
                {
                    "tool": "LoadStats",
                    "output": {
                        "playerA": prediction.stats.player_a_stats,
                        "playerB": prediction.stats.player_b_stats,
                        "head_to_head": prediction.stats.head_to_head,
                    },
                }
            )
            tool_trace.append(
                {
                    "tool": "BuildModel",
                    "output": {
                        "pcsp_path": str(prediction.model.matchup_pcsp_path),
                        "params_json": str(prediction.model.params_json_path),
                    },
                }
            )
            tool_trace.append({"tool": "RunPAT", "output": prediction.pat.to_dict()})

            payload = {
                "task_type": "prediction",
                "player_a": prediction.player_a,
                "player_b": prediction.player_b,
                "probability": prediction.probability,
                "mode": prediction.mode,
                "run_dir": str(prediction.run_dir),
            }
            answer = self._summarize(user_query, payload)
            return ExecutionResult(plan=plan, tool_trace=tool_trace, answer=answer, payload=payload)

        strategy = self.service.strategy(
            player_a=a_resolved.player_id,
            player_b=b_resolved.player_id,
            window=window,
            mode=mode,
            budget=budget,
        )
        tool_trace.append(
            {
                "tool": "BatchSensitivity",
                "output": {
                    "baseline_probability": strategy.baseline_probability,
                    "improved_probability": strategy.improved_probability,
                    "delta": strategy.delta,
                    "best_candidate": strategy.best_candidate.__dict__,
                    "top_alternatives": [cand.__dict__ for cand in strategy.top_alternatives],
                },
            }
        )

        payload = {
            "task_type": "strategy",
            "player_a": strategy.player_a,
            "player_b": strategy.player_b,
            "baseline_probability": strategy.baseline_probability,
            "improved_probability": strategy.improved_probability,
            "delta": strategy.delta,
            "best_candidate": strategy.best_candidate.__dict__,
            "mode": strategy.mode,
            "run_dir": str(strategy.run_dir),
        }
        answer = self._summarize(user_query, payload)
        return ExecutionResult(plan=plan, tool_trace=tool_trace, answer=answer, payload=payload)

    def _summarize(self, question: str, payload: dict[str, Any]) -> str:
        if self.llm_client is not None:
            llm_text = self.llm_client.summarize(question, payload)
            if llm_text:
                return llm_text

        if payload["task_type"] == "prediction":
            return (
                f"PAT reachability result: {payload['player_a']} has a {payload['probability']:.2%} "
                f"match win probability vs {payload['player_b']} (mode={payload['mode']})."
            )

        best = payload["best_candidate"]
        return (
            f"Baseline P(win)={payload['baseline_probability']:.2%}. "
            f"Best found P(win)={payload['improved_probability']:.2%} "
            f"(delta={payload['delta']:.2%}) by changing A short-serve by {best['serve_short_delta']:+.1%} "
            f"and attack style by {best['attack_delta']:+.1%}."
        )
