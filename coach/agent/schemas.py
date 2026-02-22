from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ResolvePlayers(BaseModel):
    model_config = ConfigDict(extra="forbid")

    names: list[str] = Field(..., min_length=2, max_length=2)


class LoadStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    playerA_id: str
    playerB_id: str
    window: int = Field(default=30, ge=1, le=500)
    source: Literal["local", "web"] = "local"


class BuildModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    params: dict[str, Any]
    template_name: str = "badminton_rally_template.pcsp"
    out_path: str


class RunPAT(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pcsp_path: str
    pat_path: str | None = None
    mode: Literal["real", "mock"] = "mock"
    timeout_s: int = Field(default=60, ge=1, le=1800)


class BatchSensitivity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_params: dict[str, Any]
    search_space: dict[str, Any]
    budget: int = Field(default=60, ge=1, le=2000)
    objective: Literal["maximize_A_win"] = "maximize_A_win"


class SummarizeResults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pat_outputs: list[dict[str, Any]]
    question: str
    constraints: list[str] = Field(default_factory=list)


ToolName = Literal[
    "ResolvePlayers",
    "LoadStats",
    "BuildModel",
    "RunPAT",
    "BatchSensitivity",
    "SummarizeResults",
]


class ToolInstruction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: ToolName
    arguments: dict[str, Any]


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: Literal["prediction", "strategy"]
    analysis_type: Literal["reachability", "sensitivity"]
    players: list[str] = Field(..., min_length=2, max_length=2)
    constraints: list[str] = Field(default_factory=list)
    tool_calls: list[ToolInstruction]
