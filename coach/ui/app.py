from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from coach.service import BadmintonCoachService

app = FastAPI(title="AI Badminton Coach")
service = BadmintonCoachService()


class PredictRequest(BaseModel):
    a: str
    b: str
    mode: str = Field(default="mock", pattern="^(mock|real)$")
    window: int = Field(default=30, ge=1, le=500)


class StrategyRequest(BaseModel):
    a: str
    b: str
    mode: str = Field(default="mock", pattern="^(mock|real)$")
    window: int = Field(default=30, ge=1, le=500)
    budget: int = Field(default=60, ge=1, le=1000)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
    <html>
      <head><title>AI Badminton Coach</title></head>
      <body style="font-family: sans-serif; max-width: 780px; margin: 2rem auto;">
        <h1>AI Badminton Coach</h1>
        <p>Use API endpoints:</p>
        <ul>
          <li><code>POST /predict</code> with JSON {"a":"Viktor Axelsen","b":"Kento Momota","mode":"mock"}</li>
          <li><code>POST /strategy</code> with JSON {"a":"Viktor Axelsen","b":"Kento Momota","budget":60}</li>
        </ul>
      </body>
    </html>
    """


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    result = service.predict(player_a=req.a, player_b=req.b, window=req.window, mode=req.mode)
    return {
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "player_a": result.player_a,
        "player_b": result.player_b,
        "probability": result.probability,
        "mode": result.mode,
    }


@app.post("/strategy")
def strategy(req: StrategyRequest) -> dict:
    result = service.strategy(
        player_a=req.a,
        player_b=req.b,
        window=req.window,
        mode=req.mode,
        budget=req.budget,
    )
    return {
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "player_a": result.player_a,
        "player_b": result.player_b,
        "baseline_probability": result.baseline_probability,
        "improved_probability": result.improved_probability,
        "delta": result.delta,
        "best_candidate": result.best_candidate.__dict__,
    }
