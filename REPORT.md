# REPORT: AI Badminton Coach (LLM + Formal Methods + PAT)

## 1. Executive Summary

This project implements a complete, reproducible **AI Badminton Coach** that:

1. Accepts natural-language prediction/strategy queries.
2. Uses an LLM as a **planner/tool orchestrator only** (no probability guessing).
3. Builds a **matchup-specific PCSP# model** by injecting data-driven parameters into a parametric template.
4. Runs PAT Console (`-pcsp`) in real mode or deterministic mock mode.
5. Returns exact computed win probabilities (reachability) and quantified strategy deltas (sensitivity search).

The system is verified to run on macOS with `PAT3.Console.exe` via Mono, including PAT3 compatibility fallback logic in the runner.

---

## 2. What The System Does

### 2.1 Prediction task
Input:
- "Expected winning percentage Viktor Axelsen vs Kento Momota?"

Output:
- `P(A wins match)` from PAT reachability assertion, not an LLM guess.

### 2.2 Strategy task
Input:
- "What should A change to improve chances vs B?"

Output:
- Baseline probability, best probability after controlled mix perturbations, delta, top alternatives, and explicit knob changes.

---

## 3. End-to-End Architecture

```text
User Query (CLI/Web/Chat)
  |
  v
LLM Planner (JSON plan + tool schema)  ----> only plans, never invents probabilities
  |
  v
Python Tool Executor
  |- ResolvePlayers (local CSV adapter)
  |- LoadStats + estimate parameters
  |- BuildModel (inject into PCSP template)
  |- RunPAT (real or mock)
  |- Strategy mode: BatchSensitivity (multiple PAT runs)
  v
Parser + Result Synthesizer
  |
  v
Final answer with computed probability/delta + run artifacts
```

Primary modules:
- `coach/data/*`: data loading, player resolution, statistical estimation
- `coach/model/*`: typed params + template injection
- `coach/pat/*`: PAT runner/parser/mock integration
- `coach/service.py`: orchestration for predict/strategy
- `coach/agent/*`: LLM prompts/schemas/planner/executor
- `coach/analysis/*`: reproducible batch experiments + plots

---

## 4. Formal Model (PCSP#)

Template file:
- `coach/model/templates/badminton_rally_template.pcsp`

### 4.1 State variables
- `a_games`, `b_games`: games won
- `a_points`, `b_points`: current game score
- `server`: `A` or `B`

### 4.2 Rules
- Best-of-3 (`games_to_win = 2`)
- Rally scoring to 21, win-by-2, cap at 30
- Serve-dependent rally outcome probabilities:
  - `pA_srv_win`: A wins rally when A serves
  - `pA_rcv_win`: A wins rally when B serves

### 4.3 Reachability assertion
```pcsp
#define A_WinsMatch a_games == GAMES_TO_WIN;
#assert BadmintonMatch reaches A_WinsMatch with prob;
```

### 4.4 Parametric injection (no matchup hardcoding)
Parameters are injected via placeholders (e.g. `{{pA_srv_win_w}}`, `{{target}}`) to produce `matchup.pcsp` in run directories.

Model build path:
- `coach/model/builder.py`
- typed parameter source: `coach/model/params.py`

---

## 5. Data Pipeline and Parameter Estimation

Data sources:
- `coach/data/sample_players.csv`
- `coach/data/sample_matches.csv`

Adapter:
- `coach/data/adapters/local_csv.py`

### 5.1 Estimated player quantities
- Base serve win rate (`base_srv_win`)
- Base receive win rate (`base_rcv_win`)
- Serve mix (`short`, `flick`)
- Rally style mix (`attack`, `neutral`, `safe`)

### 5.2 Smoothing and stability
- Laplace smoothing on probabilities to avoid 0/1 extremes.
- Mix smoothing + renormalization to ensure valid distributions.
- Head-to-head blending with shrinkage weight for small-sample robustness.

### 5.3 Effective matchup formula
Style differential is applied as:

`delta = w_short*(A.short - B.short) + w_attack*(A.attack - B.attack) - w_safe*(B.safe - A.safe)`

Then:
- `pA_srv_win = clamp(baseA_srv_win + delta, 0.01, 0.99)`
- `pA_rcv_win = clamp(baseA_rcv_win + delta, 0.01, 0.99)`

---

## 6. PAT Integration (Real + Mock)

Core files:
- `coach/pat/runner.py`
- `coach/pat/parser.py`
- `coach/pat/mock_pat.py`
- `scripts/pat_check.py`

### 6.1 Real PAT command
The runner executes:
- Windows: `PAT*.Console.exe -pcsp input.pcsp output.txt`
- macOS/Linux: `mono PAT*.Console.exe -pcsp input.pcsp output.txt` (auto unless disabled)

### 6.2 macOS + PAT3 compatibility
PAT3 under Mono can fail at startup due NESC module loading.

Implemented fix in `runner.py`:
1. Detect PAT3 startup failure signature.
2. Create isolated runtime copy under `runs/<run_id>/pat3_mono_compat_runtime`.
3. Compile and inject a minimal NESC shim DLL.
4. Re-run PAT automatically.

### 6.3 Parser behavior
`parse_probability` handles outputs like:
- `Probability = 0.123`
- `with prob 0.123`
- `Probability [0.842890, 0.842890]`

### 6.4 Mock mode
Deterministic logistic-style mapping from key parameters:
- monotonic behavior preserved
- no PAT install required for tests/CI

---

## 7. LLM Planner Protocol

Prompt/schema files:
- `coach/agent/prompts.py`
- `coach/agent/schemas.py`
- `coach/agent/planner.py`

The planner produces structured tool calls (JSON), e.g.:
- `ResolvePlayers`
- `LoadStats`
- `BuildModel`
- `RunPAT`
- `BatchSensitivity`
- `SummarizeResults`

Critical property:
- LLM does not directly generate probabilities.
- final numbers always come from PAT (or deterministic mock).

---

## 8. Strategy/Sensitivity Engine

Implemented in `coach/service.py`:
- baseline PAT run on baseline parameters
- candidate search over:
  - `serve_mix_A.short` in `±{0.05, 0.10, 0.20}`
  - `rally_style_A.attack` in `±{0.05, 0.10, 0.20}`
- L1 change constraint to keep changes realistic
- top-k ranking by PAT-computed win probability

Outputs:
- baseline probability
- improved probability
- delta
- best knob shifts
- top alternatives

---

## 9. How To Run

## 9.1 Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 9.2 PAT config
Use `.env` (template in `.env.example`):
```bash
PAT_CONSOLE_PATH=/absolute/path/to/PAT.Console.exe   # or PAT3.Console.exe, or PAT folder
PAT_USE_MONO=
MONO_PATH=mono
PAT_TIMEOUT_S=120
RUNS_DIR=runs
```

Check connectivity:
```bash
python scripts/pat_check.py
```

## 9.3 CLI usage
Prediction:
```bash
coach predict --a "Viktor Axelsen" --b "Kento Momota" --mode real
```

Strategy:
```bash
coach strategy --a "Viktor Axelsen" --b "Kento Momota" --mode real --budget 60
```

Direct PAT on any PCSP:
```bash
coach pat-run --pcsp examples/minimal.pcsp --mode real
```

Interactive planner chat:
```bash
coach chat --mode real
```

Batch experiments:
```bash
coach experiments --mode mock --output-dir runs/experiments
```

---

## 10. How To Use It Correctly

1. Use `predict` for matchup probability.
2. Use `strategy` when you need actionable knob adjustments.
3. Use `pat-run` for standalone model verification.
4. Always inspect run artifacts for reproducibility.

Important:
- If you run `pat-run` on an old generated `.pcsp` file, it may fail if the old model syntax is invalid.
- Current template is fixed; regenerate models via current `coach predict`/`coach strategy` for clean real-mode runs.

---

## 11. Run Artifacts and Reproducibility

Every run creates `runs/<run_id>/` containing:
- input PCSP (`matchup.pcsp`, `baseline.pcsp`, candidates)
- PAT output (`pat_output.txt`)
- PAT logs (`pat_stdout.txt`, `pat_stderr.txt`)
- machine-readable metadata (`pat_run.json`, `summary.json`)
- params and input context (`params.json`, `inputs.json`, `stats.json`, etc.)

This makes every reported number traceable to a concrete model and solver output.

---

## 12. Why It Works

### 12.1 Separation of responsibilities
- LLM: planning and natural-language explanation.
- Formal engine (PAT): probability computation.

This avoids unverifiable probabilistic guessing by the LLM.

### 12.2 Parametric formalization
- Same PCSP template supports all matchups.
- Matchup specificity enters through validated injected parameters.

### 12.3 Deterministic computation path
- Given fixed data + params + PAT mode, results are reproducible.
- Mock mode is deterministic by construction.

### 12.4 Strong failure signaling
- Missing PAT binary, invalid model syntax, empty PAT output, parse failures, and timeouts are surfaced with explicit errors and artifact pointers.

---

## 13. Evidence From Real Runs (macOS + PAT3)

Observed successful real runs (February 22, 2026):

1. `coach predict --a "Viktor Axelsen" --b "Kento Momota" --mode real`
   - `run_id: predict_20260222_024253_dfe97e61`
   - `win_probability_a: 0.842890`

2. `coach pat-run --pcsp runs/predict_20260222_024253_dfe97e61/matchup.pcsp --mode real`
   - `run_id: pat_20260222_024342_273ac0d7`
   - `probability: 0.842890`

Interpretation:
- Direct PAT run and pipeline prediction agree on the same generated model probability.

---

## 14. Testing Status

Unit/integration tests run without PAT installed (mock mode), including:
- parameter validation
- model template injection
- PAT parser
- PAT runner mock + real-missing-path behavior
- path resolution and PAT error-signal handling
- end-to-end mock pipeline

Current suite status:
- `pytest -q` passes.

---

## 15. Limitations and Future Work

Current limitations:
- Sample data is match-level proxy, not full rally-by-rally telemetry.
- Strategy search is local grid-based and budget-limited.
- PAT3 compatibility fallback is practical but still environment-dependent (Mono + `mcs`).

Future improvements:
- Add rally-level data ingestion and stronger calibration.
- Add richer tactical knobs and multi-objective optimization.
- Add optional web adapters with caching/provenance.
- Expand formal properties (expected points/time, risk-aware objectives).
