from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from coach.agent.llm_client import LLMClient
from coach.agent.planner import AgentExecutor
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


class ChatRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    mode: str = Field(default="mock", pattern="^(mock|real)$")
    window: int = Field(default=30, ge=1, le=500)
    budget: int = Field(default=60, ge=1, le=1000)
    show_trace: bool = False


def _render_home() -> str:
    example_players = service.adapter.players_df["name"].head(8).tolist()
    example_json = json.dumps(example_players)
    return f"""
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>AI Badminton Coach</title>
        <style>
          :root {{
            --bg: #f6f1e8;
            --panel: rgba(255, 250, 242, 0.84);
            --panel-strong: #fffaf2;
            --ink: #1c2430;
            --muted: #5e6773;
            --line: rgba(28, 36, 48, 0.12);
            --accent: #0d7a5f;
            --accent-2: #e36b2c;
            --accent-3: #1e5aa8;
            --shadow: 0 22px 60px rgba(28, 36, 48, 0.12);
          }}

          * {{
            box-sizing: border-box;
          }}

          body {{
            margin: 0;
            font-family: Georgia, "Times New Roman", serif;
            color: var(--ink);
            background:
              radial-gradient(circle at top left, rgba(227, 107, 44, 0.18), transparent 32%),
              radial-gradient(circle at top right, rgba(13, 122, 95, 0.17), transparent 28%),
              linear-gradient(180deg, #fcfaf6 0%, var(--bg) 100%);
          }}

          .shell {{
            width: min(1180px, calc(100% - 32px));
            margin: 24px auto 40px;
          }}

          .hero {{
            position: relative;
            overflow: hidden;
            padding: 32px;
            border: 1px solid var(--line);
            border-radius: 28px;
            background:
              linear-gradient(135deg, rgba(255, 255, 255, 0.85), rgba(245, 237, 224, 0.95)),
              linear-gradient(160deg, rgba(13, 122, 95, 0.1), rgba(30, 90, 168, 0.04));
            box-shadow: var(--shadow);
          }}

          .hero::after {{
            content: "";
            position: absolute;
            inset: auto -40px -70px auto;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(13, 122, 95, 0.18), transparent 68%);
          }}

          .eyebrow {{
            display: inline-flex;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(13, 122, 95, 0.1);
            color: var(--accent);
            font-size: 12px;
            letter-spacing: 0.14em;
            text-transform: uppercase;
          }}

          h1 {{
            margin: 14px 0 12px;
            font-size: clamp(2.4rem, 5vw, 4.3rem);
            line-height: 0.95;
          }}

          .hero p {{
            max-width: 760px;
            margin: 0;
            font-size: 1.05rem;
            line-height: 1.6;
            color: var(--muted);
          }}

          .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 14px;
            margin-top: 24px;
          }}

          .stat {{
            padding: 16px;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.7);
          }}

          .stat strong {{
            display: block;
            font-size: 1.4rem;
            margin-bottom: 6px;
          }}

          .layout {{
            display: grid;
            grid-template-columns: 1.25fr 0.95fr;
            gap: 18px;
            margin-top: 18px;
          }}

          .panel {{
            border: 1px solid var(--line);
            border-radius: 24px;
            background: var(--panel);
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
          }}

          .panel-head {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            padding: 22px 22px 0;
          }}

          .panel-head h2,
          .panel-head h3 {{
            margin: 0;
          }}

          .panel-body {{
            padding: 22px;
          }}

          .grid {{
            display: grid;
            gap: 12px;
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }}

          .full {{
            grid-column: 1 / -1;
          }}

          label {{
            display: block;
            font-size: 0.88rem;
            margin-bottom: 6px;
            color: var(--muted);
          }}

          input, select, textarea, button {{
            width: 100%;
            font: inherit;
          }}

          input, select, textarea {{
            border: 1px solid rgba(28, 36, 48, 0.12);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.86);
            padding: 12px 14px;
            color: var(--ink);
          }}

          textarea {{
            min-height: 150px;
            resize: vertical;
          }}

          button {{
            border: 0;
            border-radius: 999px;
            padding: 12px 18px;
            cursor: pointer;
            transition: transform 160ms ease, opacity 160ms ease;
          }}

          button:hover {{
            transform: translateY(-1px);
          }}

          button:disabled {{
            opacity: 0.6;
            cursor: wait;
            transform: none;
          }}

          .primary {{
            background: linear-gradient(135deg, var(--accent), #18a17c);
            color: white;
          }}

          .secondary {{
            background: linear-gradient(135deg, var(--accent-2), #ef8e57);
            color: white;
          }}

          .ghost {{
            background: rgba(28, 36, 48, 0.06);
            color: var(--ink);
          }}

          .chat-output {{
            display: flex;
            flex-direction: column;
            gap: 14px;
          }}

          .bubble {{
            padding: 16px 18px;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: var(--panel-strong);
          }}

          .bubble.user {{
            background: rgba(30, 90, 168, 0.08);
            border-color: rgba(30, 90, 168, 0.12);
          }}

          .bubble.system {{
            background: rgba(13, 122, 95, 0.07);
            border-color: rgba(13, 122, 95, 0.12);
          }}

          .muted {{
            color: var(--muted);
          }}

          .result-card {{
            padding: 16px;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.76);
          }}

          .metric-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin-top: 12px;
          }}

          .metric {{
            padding: 12px;
            border-radius: 14px;
            background: rgba(28, 36, 48, 0.04);
          }}

          .metric strong {{
            display: block;
            font-size: 1.1rem;
          }}

          .hint-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
          }}

          .hint {{
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.78);
            cursor: pointer;
          }}

          pre {{
            margin: 0;
            white-space: pre-wrap;
            word-break: break-word;
          }}

          @media (max-width: 960px) {{
            .layout {{
              grid-template-columns: 1fr;
            }}
          }}

          @media (max-width: 700px) {{
            .shell {{
              width: min(100% - 20px, 100%);
              margin-top: 10px;
            }}

            .hero, .panel-body {{
              padding: 18px;
            }}

            .panel-head {{
              padding: 18px 18px 0;
            }}

            .grid {{
              grid-template-columns: 1fr;
            }}
          }}
        </style>
      </head>
      <body>
        <main class="shell">
          <section class="hero">
            <div class="eyebrow">AI Mode Frontend</div>
            <h1>Talk to the badminton coach in your browser.</h1>
            <p>
              Ask natural-language questions in AI mode, run direct win predictions,
              and explore strategy improvements without dropping back to the CLI.
            </p>
            <div class="stats">
              <div class="stat">
                <strong>AI chat</strong>
                Natural-language planning backed by the existing agent executor.
              </div>
              <div class="stat">
                <strong>Prediction</strong>
                Run PAT-based matchup probability checks with mock or real mode.
              </div>
              <div class="stat">
                <strong>Strategy</strong>
                Surface the best tactical knob adjustments and quantified deltas.
              </div>
            </div>
          </section>

          <section class="layout">
            <div class="panel">
              <div class="panel-head">
                <h2>AI Coach</h2>
                <span class="muted">Natural language</span>
              </div>
              <div class="panel-body">
                <form id="chat-form" class="grid">
                  <div class="full">
                    <label for="query">Ask the coach</label>
                    <textarea id="query" name="query" placeholder="Example: What should Viktor Axelsen adjust to improve his chances against Kento Momota?"></textarea>
                  </div>
                  <div>
                    <label for="chat-mode">Execution mode</label>
                    <select id="chat-mode" name="mode">
                      <option value="mock">Mock PAT</option>
                      <option value="real">Real PAT</option>
                    </select>
                  </div>
                  <div>
                    <label for="chat-window">Window</label>
                    <input id="chat-window" name="window" type="number" min="1" max="500" value="30" />
                  </div>
                  <div>
                    <label for="chat-budget">Strategy budget</label>
                    <input id="chat-budget" name="budget" type="number" min="1" max="1000" value="60" />
                  </div>
                  <div>
                    <label for="show-trace">Trace output</label>
                    <select id="show-trace" name="show_trace">
                      <option value="false">Hide trace</option>
                      <option value="true">Show trace</option>
                    </select>
                  </div>
                  <div class="full">
                    <button class="primary" id="chat-submit" type="submit">Run AI coach</button>
                  </div>
                </form>

                <div class="hint-list" id="chat-hints">
                  <button class="hint" type="button">Predict Viktor Axelsen vs Kento Momota</button>
                  <button class="hint" type="button">What strategy should Akane Yamaguchi use against An Se-young?</button>
                  <button class="hint" type="button">How can Tai Tzu Ying improve her odds versus Chen Yufei?</button>
                </div>

                <div class="chat-output" id="chat-output" style="margin-top: 18px;">
                  <div class="bubble system">
                    AI mode is ready. Ask for a prediction or a tactical recommendation and the frontend will call the same coach executor used by the CLI.
                  </div>
                </div>
              </div>
            </div>

            <div style="display: grid; gap: 18px;">
              <div class="panel">
                <div class="panel-head">
                  <h3>Direct Prediction</h3>
                  <span class="muted">Structured API</span>
                </div>
                <div class="panel-body">
                  <form id="predict-form" class="grid">
                    <div>
                      <label for="predict-a">Player A</label>
                      <input id="predict-a" name="a" list="players" placeholder="Viktor Axelsen" />
                    </div>
                    <div>
                      <label for="predict-b">Player B</label>
                      <input id="predict-b" name="b" list="players" placeholder="Kento Momota" />
                    </div>
                    <div>
                      <label for="predict-mode">Mode</label>
                      <select id="predict-mode" name="mode">
                        <option value="mock">Mock</option>
                        <option value="real">Real</option>
                      </select>
                    </div>
                    <div>
                      <label for="predict-window">Window</label>
                      <input id="predict-window" name="window" type="number" min="1" max="500" value="30" />
                    </div>
                    <div class="full">
                      <button class="secondary" id="predict-submit" type="submit">Compute win probability</button>
                    </div>
                  </form>
                  <div id="predict-result" style="margin-top: 14px;" class="muted">No prediction yet.</div>
                </div>
              </div>

              <div class="panel">
                <div class="panel-head">
                  <h3>Strategy Explorer</h3>
                  <span class="muted">Sensitivity search</span>
                </div>
                <div class="panel-body">
                  <form id="strategy-form" class="grid">
                    <div>
                      <label for="strategy-a">Player A</label>
                      <input id="strategy-a" name="a" list="players" placeholder="Akane Yamaguchi" />
                    </div>
                    <div>
                      <label for="strategy-b">Player B</label>
                      <input id="strategy-b" name="b" list="players" placeholder="An Se-young" />
                    </div>
                    <div>
                      <label for="strategy-mode">Mode</label>
                      <select id="strategy-mode" name="mode">
                        <option value="mock">Mock</option>
                        <option value="real">Real</option>
                      </select>
                    </div>
                    <div>
                      <label for="strategy-window">Window</label>
                      <input id="strategy-window" name="window" type="number" min="1" max="500" value="30" />
                    </div>
                    <div>
                      <label for="strategy-budget">Budget</label>
                      <input id="strategy-budget" name="budget" type="number" min="1" max="1000" value="60" />
                    </div>
                    <div class="full">
                      <button class="primary" id="strategy-submit" type="submit">Find best adjustment</button>
                    </div>
                  </form>
                  <div id="strategy-result" style="margin-top: 14px;" class="muted">No strategy run yet.</div>
                </div>
              </div>
            </div>
          </section>
        </main>

        <datalist id="players"></datalist>

        <script>
          const playerSeed = {example_json};
          const playerList = document.getElementById("players");

          function addBubble(kind, content) {{
            const bubble = document.createElement("div");
            bubble.className = `bubble ${{kind}}`;
            bubble.innerHTML = content;
            document.getElementById("chat-output").prepend(bubble);
          }}

          function renderPrediction(result) {{
            return `
              <div class="result-card">
                <strong>${{result.player_a}}</strong> vs <strong>${{result.player_b}}</strong>
                <div class="metric-row">
                  <div class="metric"><span class="muted">Win probability</span><strong>${{(result.probability * 100).toFixed(2)}}%</strong></div>
                  <div class="metric"><span class="muted">Mode</span><strong>${{result.mode}}</strong></div>
                  <div class="metric"><span class="muted">Run</span><strong>${{result.run_id}}</strong></div>
                </div>
                <div class="muted" style="margin-top: 10px;">Artifacts: ${{result.run_dir}}</div>
              </div>
            `;
          }}

          function renderStrategy(result) {{
            const best = result.best_candidate;
            return `
              <div class="result-card">
                <strong>${{result.player_a}}</strong> vs <strong>${{result.player_b}}</strong>
                <div class="metric-row">
                  <div class="metric"><span class="muted">Baseline</span><strong>${{(result.baseline_probability * 100).toFixed(2)}}%</strong></div>
                  <div class="metric"><span class="muted">Improved</span><strong>${{(result.improved_probability * 100).toFixed(2)}}%</strong></div>
                  <div class="metric"><span class="muted">Delta</span><strong>${{(result.delta * 100).toFixed(2)}}%</strong></div>
                </div>
                <div class="muted" style="margin-top: 12px;">
                  Best candidate: short serve ${{(best.serve_short_delta * 100).toFixed(1)}}%, attack ${{(best.attack_delta * 100).toFixed(1)}}%,
                  unforced-error proxy ${{(best.unforced_error_delta * 100).toFixed(1)}}%, return pressure ${{(best.return_pressure_delta * 100).toFixed(1)}}%.
                </div>
              </div>
            `;
          }}

          async function postJson(url, payload) {{
            const response = await fetch(url, {{
              method: "POST",
              headers: {{ "Content-Type": "application/json" }},
              body: JSON.stringify(payload),
            }});
            const data = await response.json();
            if (!response.ok) {{
              throw new Error(data.detail || "Request failed");
            }}
            return data;
          }}

          async function seedPlayers() {{
            playerSeed.forEach((name) => {{
              const option = document.createElement("option");
              option.value = name;
              playerList.appendChild(option);
            }});
            try {{
              const response = await fetch("/players");
              const data = await response.json();
              playerList.innerHTML = "";
              data.players.forEach((name) => {{
                const option = document.createElement("option");
                option.value = name;
                playerList.appendChild(option);
              }});
            }} catch (_err) {{
            }}
          }}

          document.getElementById("chat-form").addEventListener("submit", async (event) => {{
            event.preventDefault();
            const submit = document.getElementById("chat-submit");
            submit.disabled = true;
            const payload = {{
              query: document.getElementById("query").value,
              mode: document.getElementById("chat-mode").value,
              window: Number(document.getElementById("chat-window").value),
              budget: Number(document.getElementById("chat-budget").value),
              show_trace: document.getElementById("show-trace").value === "true",
            }};
            addBubble("user", payload.query.replace(/</g, "&lt;"));
            try {{
              const result = await postJson("/chat", payload);
              let body = `<div><strong>Coach answer</strong></div><div style="margin-top:8px;">${{result.answer.replace(/</g, "&lt;")}}</div>`;
              body += `<div class="muted" style="margin-top: 10px;">Run directory: ${{result.payload.run_dir}}</div>`;
              if (result.show_trace) {{
                body += `<div style="margin-top:12px;"><strong>Trace</strong><pre>${{JSON.stringify(result.tool_trace, null, 2).replace(/</g, "&lt;")}}</pre></div>`;
              }}
              addBubble("system", body);
            }} catch (err) {{
              addBubble("system", `<strong>Request failed</strong><div style="margin-top:8px;">${{String(err.message || err).replace(/</g, "&lt;")}}</div>`);
            }} finally {{
              submit.disabled = false;
            }}
          }});

          document.querySelectorAll("#chat-hints .hint").forEach((button) => {{
            button.addEventListener("click", () => {{
              document.getElementById("query").value = button.textContent;
            }});
          }});

          document.getElementById("predict-form").addEventListener("submit", async (event) => {{
            event.preventDefault();
            const submit = document.getElementById("predict-submit");
            submit.disabled = true;
            document.getElementById("predict-result").textContent = "Running prediction...";
            try {{
              const result = await postJson("/predict", {{
                a: document.getElementById("predict-a").value,
                b: document.getElementById("predict-b").value,
                mode: document.getElementById("predict-mode").value,
                window: Number(document.getElementById("predict-window").value),
              }});
              document.getElementById("predict-result").innerHTML = renderPrediction(result);
            }} catch (err) {{
              document.getElementById("predict-result").textContent = String(err.message || err);
            }} finally {{
              submit.disabled = false;
            }}
          }});

          document.getElementById("strategy-form").addEventListener("submit", async (event) => {{
            event.preventDefault();
            const submit = document.getElementById("strategy-submit");
            submit.disabled = true;
            document.getElementById("strategy-result").textContent = "Running strategy search...";
            try {{
              const result = await postJson("/strategy", {{
                a: document.getElementById("strategy-a").value,
                b: document.getElementById("strategy-b").value,
                mode: document.getElementById("strategy-mode").value,
                window: Number(document.getElementById("strategy-window").value),
                budget: Number(document.getElementById("strategy-budget").value),
              }});
              document.getElementById("strategy-result").innerHTML = renderStrategy(result);
            }} catch (err) {{
              document.getElementById("strategy-result").textContent = String(err.message || err);
            }} finally {{
              submit.disabled = false;
            }}
          }});

          seedPlayers();
        </script>
      </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return _render_home()


@app.get("/players")
def players() -> dict[str, list[str]]:
    names = sorted(service.adapter.players_df["name"].astype(str).tolist())
    return {"players": names}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    try:
        result = service.predict(player_a=req.a, player_b=req.b, window=req.window, mode=req.mode)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "player_a": result.player_a,
        "player_b": result.player_b,
        "probability": result.probability,
        "mode": result.mode,
    }


@app.post("/strategy")
def strategy(req: StrategyRequest) -> dict[str, Any]:
    try:
        result = service.strategy(
            player_a=req.a,
            player_b=req.b,
            window=req.window,
            mode=req.mode,
            budget=req.budget,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "player_a": result.player_a,
        "player_b": result.player_b,
        "baseline_probability": result.baseline_probability,
        "improved_probability": result.improved_probability,
        "delta": result.delta,
        "best_candidate": result.best_candidate.__dict__,
        "top_alternatives": [cand.__dict__ for cand in result.top_alternatives],
        "mode": result.mode,
    }


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    try:
        result = AgentExecutor(service=service, llm_client=LLMClient()).run(
            req.query,
            mode=req.mode,
            window=req.window,
            budget=req.budget,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "answer": result.answer,
        "plan": result.plan.model_dump(),
        "payload": result.payload,
        "tool_trace": result.tool_trace,
        "show_trace": req.show_trace,
    }
