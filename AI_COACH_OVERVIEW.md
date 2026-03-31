## AI Badminton Coach – Architecture and PCSP Modeling

This document explains how the `ai-coach` system works end‑to‑end, with emphasis on the **PCSP modeling**, **PAT integration**, and the **agent tooling** around it.

---

### 1. High‑level Concept

The AI Badminton Coach is a small decision‑support system for racket sports matchups. Given two players and historical rally‑level stats, it:

- **Builds probabilistic matchup parameters** from data (serve/receive win rates, style mixes, influence weights).
- **Compiles a parametric PCSP model** for a best‑of‑N badminton match.
- **Runs PAT** (or a deterministic mock) on that PCSP model to compute the **reachability probability** that Player A wins the match.
- Optionally **searches over tactical adjustments** (serve mix and rally style) to suggest strategy changes that maximize A’s chance to win.
- Wraps this in an **LLM‑driven “planner + tools” agent** that interprets natural‑language coaching questions.

At the center is `BadmintonCoachService` (`coach/service.py`) which exposes two core operations:

- **`predict(...)`** – compute P(A wins match vs B).
- **`strategy(...)`** – run a small search over style changes for Player A and propose better strategies.

---

### 2. Data layer – Local CSV adapter and stats

**Files:** `coach/data/adapters/local_csv.py`, `coach/data/stats_builder.py`

- **`LocalCSVAdapter`**
  - Reads from `sample_players.csv` and `sample_matches.csv` by default.
  - Provides:
    - `players_df` / `matches_df` – lazy‑loaded `pandas` tables.
    - `resolve_player(name)` – fuzzy name resolution (normalized name, contains, and close‑match suggestions).
    - `get_player_matches(player_id, window, as_of_date)` – last `window` matches for a player (with a larger underlying tail to get enough data).
    - `get_player_params(player_id, window, as_of_date)` – aggregate per‑player statistics:
      - Smoothed **serve win probability** and **receive win probability** using Laplace‑style smoothing.
      - Weighted averages for **serve mix** (short vs flick) and **rally style** (attack / neutral / safe).
      - Basic derived metrics like total matches, overall win rate, etc.
    - `get_head_to_head(player_a_id, player_b_id, window, as_of_date)` – summary of A’s historical performance vs B (win rate, serve/receive success).

- **`MatchupStats` and `build_matchup_params(...)`**
  - `build_matchup_params` is the main bridge from raw data to **PCSP parameters**:
    - Resolves both players (by explicit ID or fuzzy name).
    - Calls `get_player_params` for each and `get_head_to_head` for the pair.
    - Estimates **influence weights** (`InfluenceWeights`) using a simple regression:
      - Regresses point‑share \( (a\_points / (a\_points + b\_points) - 0.5) \) on:
        - Short‑serve rate difference.
        - Attack rate difference.
        - Negative of safe‑style difference (more safe from B should tilt in B’s favor).
      - Clips and returns positive weights `w_short`, `w_attack`, `w_safe`.
    - Blends head‑to‑head stats into player A’s base serve/receive win rates when there is enough direct matchup data.
    - Builds **`PlayerParams`** for each player and wraps them plus `InfluenceWeights` into a **`MatchupParams`** object.
  - Returns both:
    - `MatchupParams` – structured, validated parameters used for PCSP generation.
    - `MatchupStats` – diagnostic view of the raw stats (used for artifacts and explanations).

---

### 3. Parameter models – From data to PCSP context

**File:** `coach/model/params.py`

The parameter layer is implemented as a set of **Pydantic models**:

- **`ServeMix`**
  - Fields: `short`, `flick` (both in \[0, 1\]) and must sum to 1.0.

- **`RallyStyleMix`**
  - Fields: `attack`, `neutral`, `safe` (all in \[0, 1\]) and must sum to 1.0.

- **`InfluenceWeights`**
  - Fields: `w_short`, `w_attack`, `w_safe` (each \(> 0\) and ≤ 0.3).
  - These describe how much differences in serve mix / style should move effective rally win probabilities.

- **`PlayerParams`**
  - Identity and base performance:
    - `player_id`, `name` – validated non‑empty strings.
    - `base_srv_win`, `base_rcv_win` – smoothed base probabilities of winning a rally on serve / receive.
  - Style:
    - `serve_mix: ServeMix`
    - `rally_style: RallyStyleMix`
  - `sample_matches` – how much data the estimates are based on.

- **`MatchupParams`**
  - Holds `player_a: PlayerParams`, `player_b: PlayerParams`, `weights: InfluenceWeights`.
  - Game‑structure parameters:
    - `target` (points to win a game, default 21),
    - `cap` (max points, default 30),
    - `best_of` (matches are best‑of N games and N must be odd).
  - **Effective probabilities (`effective_probabilities`)**
    - Computes a style‑dependent delta:
      - Uses weights and style differences:
        - Short‑serve rate difference \(A.short - B.short\).
        - Attack style difference \(A.attack - B.attack\).
        - Safe style difference \(B.safe - A.safe\) (penalizing A’s chances when B plays more safely).
      - Applies this `delta` to A’s base serve and receive win rates, then clamps into \[0, 1\].
    - Returns:
      - `pA_srv_win`, `pA_rcv_win`, `pB_srv_win`, `pB_rcv_win`.
  - **Adjustments and distance:**
    - `with_adjustments(serve_short_delta, attack_delta)`:
      - Builds a new `MatchupParams` where **only Player A’s style** is changed:
        - Adjusts `serve_mix.short` by `serve_short_delta` (clamped to \[0.01, 0.99\]), recomputes `flick = 1 - short`.
        - Adjusts `rally_style.attack` by `attack_delta` while keeping `neutral + safe` proportions consistent and total sum 1.
    - `l1_change_from(baseline)`:
      - Computes an L1 distance between Player A’s old and new serve/styling vectors (serve mix and rally style both included).
      - Used to constrain how far a proposed strategy can deviate from baseline.
  - **Template context (`to_template_context`)**
    - Converts `MatchupParams` into a dict of **string and integer substitutions** for the PCSP template:
      - Game structure (target, cap, best_of, games_to_win).
      - Effective probabilities in **floating and scaled integer** forms, e.g. `pA_srv_win`, `pA_srv_win_w`, `pA_srv_lose_w`, etc.
      - Raw base probabilities and style parameters for both players.
      - Weights `w_short`, `w_attack`, `w_safe`.
      - Player names for comments.

This layer is where the **domain assumptions** live:

- How style and serve mix influence rally win rates.
- How far you are allowed to change a style vector (L1 bound used in strategy search).
- What game structure is modeled (21‑point games, best‑of‑3 by default).

---

### 4. PCSP model – Badminton match as a reachability problem

**File:** `coach/model/templates/badminton_rally_template.pcsp`

The PCSP template encodes a full **best‑of‑N badminton match** as a discrete‑time probabilistic model:

- **State variables**
  - `a_games`, `b_games` – games currently won by A/B.
  - `a_points`, `b_points` – points in the current game.
  - `server` – who is serving (A or B).

- **Game‑winning conditions**
  - A wins a game when:
    - \(a\_points ≥ TARGET\) and \(a\_points − b\_points ≥ 2\), or
    - `a_points == CAP`.
  - Similarly for B.
  - `TARGET` and `CAP` are parameterized via template variables.

- **Match structure**
  - `BadmintonMatch = Rally;`
  - `Rally` process:
    - While neither player has enough games to win the match, the model chooses:
      - `ServeA` when `server == A`,
      - `ServeB` when `server == B`.

- **Probabilistic transitions**
  - **Serving as A**:
    - `ServeA` uses a `pcase` with weights:
      - `pA_srv_win_w`: A wins the rally on serve → `UpdateA`.
      - `pA_srv_lose_w`: B wins on A’s serve → `UpdateB`.
  - **Serving as B**:
    - `ServeB` uses `pA_rcv_win_w` / `pA_rcv_lose_w`.
  - `UpdateA` / `UpdateB`:
    - Increment points, set next server, then go to `CheckGame`.
  - `CheckGame`:
    - Handles game completion, resetting points, advancing games won, and transitioning either to the next game or to `MatchDone`.

- **Property of interest**
  - `#define A_WinsMatch a_games == GAMES_TO_WIN;`
  - `#assert BadmintonMatch reaches A_WinsMatch with prob;`
  - The PCSP safety/reachability property **asks PAT** for the probability that Player A wins the match under the given rally probabilities.

**How parameters connect to this:**

- The template is filled with values from `MatchupParams.to_template_context()`.
- Critically, the **effective probabilities** already embed influence from:
  - Style differences (attack/safe).
  - Serve mix differences (short vs flick).
  - Influence weights learned from history.
- PCSP itself stays **agnostic to “why”** probabilities have certain values; it just consumes them to compute a mathematically sound match‑win probability via model checking or simulation, depending on PAT settings.

---

### 5. PAT integration – Real vs mock execution

**Files:** `coach/pat/runner.py`, `coach/pat/parser.py`, `coach/pat/mock_pat.py`

- **`run_pat(...)`**
  - Normalizes paths, prepares a run directory, and creates:
    - `pat_stdout.txt`, `pat_stderr.txt`, `pat_output.txt`, and JSON summaries for traceability.
  - **Modes:**
    - `"mock"`:
      - Uses `mock_run(...)` to produce deterministic output without PAT installed.
      - Writes a simple text result with `with prob X.XXXXXX`.
      - Always returns `ok=True` with a synthetic command.
    - `"real"`:
      - Requires a real PAT Console (`PAT_CONSOLE_PATH`) and optionally Mono on non‑Windows.
      - Builds the PAT command line: either `mono PAT.Console.exe -pcsp matchup.pcsp out.txt` or direct `PAT.Console.exe -pcsp ...`.
      - Handles retries and a **PAT3/Mono compatibility shim** where necessary:
        - Detects specific error patterns and, if needed, constructs an isolated runtime directory and compiles a small `NESC` shim to work around PAT3 issues under Mono.
      - Parses PAT’s output file with `parse_probability`, packages logs and errors into a JSON payload.
      - Enforces consistency (non‑empty output, etc.) and writes human‑readable hints when failures happen.

- **`parse_probability(text)`**
  - Flexible parser for PAT console textual outputs:
    - Looks for contexts mentioning “probability” or “with prob”.
    - Extracts floating‑point numbers in the neighborhood.
    - Accepts both single numbers and ranges like `Probability [0.123, 0.123]`.
    - Uses the **last** contextual match to allow for multi‑stage verification traces.

- **`mock_run(pcsp_path, out_path)`**
  - Provides a deterministic mapping from PCSP parameters to a probability:
    - Reads PCSP file, extracts key parameters (serve/win probabilities, style and serve mix for A/B).
    - Constructs a linear combination of:
      - Deviations from 0.5 in A’s serve/receive win rates.
      - Edges in serve mix, attack style, and safe style.
    - Passes this through a logistic function and clamps to \[0.01, 0.99\].
  - Writes a minimal PAT‑like output file and returns a structured result, including the **mocked probability**.

---

### 6. Service layer – Prediction and strategy search

**File:** `coach/service.py`

- **`BadmintonCoachService`**
  - Entry‑point abstraction used by:
    - CLI / scripts.
    - Agent tools.
    - UI components.
  - Configuration and inputs:
    - `adapter: LocalCSVAdapter` – where stats come from.
    - `template_name` – PCSP template file (`badminton_rally_template.pcsp` by default).
    - `runs_root` – root directory where each call’s artifacts are stored.
    - `config: CoachConfig` – PAT/Mono paths, timeouts, and run directories from environment.

- **`predict(...)` flow**
  1. Create a new run directory (`predict-<timestamp>` unless `run_id` is provided).
  2. Build **matchup parameters & stats** via `build_matchup_params`.
  3. Build **PCSP model**:
     - `build_matchup_model(params, template_name, out_path=run_dir / "matchup.pcsp")`.
     - Renders PCSP template and writes `params.json` plus the template context.
  4. Execute PAT:
     - `_execute_pat(pcsp_path, run_dir, mode, pat_path, timeout_s)`.
     - Wraps `run_pat` and probability parsing into a `PATExecution` dataclass.
  5. Construct a **`PredictionResult`**:
     - Includes run metadata, players, probability, parameters, stats, and PAT execution details.
  6. Write artifacts:
     - `inputs.json` (task, players, window, mode, run id).
     - `stats.json` (detailed stats and weights).
     - `prediction_result.json` (probability and PAT metadata).
     - `summary.json` (compact question, players, params snapshot, final probability with timestamp).

- **`strategy(...)` flow**
  1. Create a run directory (`strategy-<timestamp>`).
  2. Reuse `build_matchup_params` to get baseline `MatchupParams` and `MatchupStats`.
  3. Build baseline PCSP and run PAT to get **baseline probability**.
  4. Generate candidate strategies via `_generate_candidates(baseline, l1_bound)`:
     - Grid over `serve_short_delta` and `attack_delta` in \{-0.20, -0.10, -0.05, 0.05, 0.10, 0.20\}, plus a 0 delta.
     - Skip the trivial (0, 0) adjustment.
     - For each adjustment:
       - Build an adjusted `MatchupParams` using `with_adjustments`.
       - Compute L1 distance to baseline; keep only those within `l1_bound` (default 0.3).
     - Sort by `(l1_change, |serve_delta| + |attack_delta|)` to prefer **minimal, interpretable changes**.
  5. Evaluate up to `budget` candidates (default 60):
     - For each candidate:
       - Build a new PCSP file under `run_dir / candidates/`.
       - Run PAT (or mock).
       - If probability parsed successfully, create a `StrategyCandidate` with:
         - Deltas in serve_short and attack.
         - L1 change.
         - Resulting match‑win probability.
  6. Rank all successful candidates by probability descending.
  7. Produce a **`StrategyResult`**:
     - `baseline_probability`, `improved_probability`, `delta`.
     - The **best candidate** and up to 5 **top alternatives**, each with deltas and probabilities.
     - Baseline and best `MatchupParams` for deeper inspection.
  8. Write artifacts:
     - `inputs.json`, `stats.json`, `strategy_result.json`, and a `summary.json` mirroring prediction.
     - `top_alternatives.csv` listing the ranked adjustments and their probabilities for easy analysis.

Implementation decisions worth noting:

- Strategy search is **small, discrete, and explainable**:
  - Only two knobs: serve short‑serve fraction and attack fraction.
  - Hard L1 bound ensures recommendations are realistic tweaks, not wild style changes.
- Data and model parameters are fully **externally logged** to enable experiment reproducibility and manual inspection.

---

### 7. Agent layer – Planner and executor

**File:** `coach/agent/planner.py`, `coach/agent/llm_client.py`, plus schemas/prompts.

- **Planner (`Planner`)**
  - Main job: turn a **natural‑language question** into a structured **plan** (tool calls + arguments).
  - First tries LLM‑based planning:
    - If an `LLMClient` is available and enabled, it sends a planning prompt and tries to parse a JSON `Plan` from the response.
  - If planning via LLM fails or is not enabled, uses a **heuristic fallback**:
    - Detects `task_type`:
      - If the query contains words like “strategy”, “adjust”, “improve”, “beat”, “optimize”, “what should…”, “tactic”, etc., classify as **strategy**.
      - Otherwise, treat as **prediction**.
    - Extracts players by:
      - Matching full names in the adapter’s player list.
      - Matching surnames.
      - Parsing patterns like `"A vs B"` or `"between A and B"` and resolving with `adapter.resolve_player`.
    - Extracts **constraints** from question text (e.g. mentions of aggression, safety, serve) and tags them for possible summary usage.
  - Builds a **`Plan` object** that lists a deterministic tool pipeline:
    - For prediction:
      - `ResolvePlayers` → `LoadStats` → `BuildModel` → `RunPAT` → `SummarizeResults`.
    - For strategy:
      - `ResolvePlayers` → `LoadStats` → `BatchSensitivity` → `SummarizeResults`.

- **Executor (`AgentExecutor`)**
  - Orchestrates actual computation using deterministic Python tools, regardless of whether the plan came from LLM or heuristics:
    - Uses `BadmintonCoachService` (and thus all the PCSP machinery) as the core engine.
    - Keeps a `tool_trace` list describing each tool invocation and its inputs/outputs for transparency.
  - **Execution flow:**
    - Calls `planner.create_plan(...)` to get a `Plan`.
    - Resolves players via `LocalCSVAdapter`.
    - For **prediction**:
      - Calls `service.predict(...)`.
      - Logs `LoadStats`, `BuildModel`, and `RunPAT` outputs into `tool_trace`.
    - For **strategy**:
      - Calls `service.strategy(...)`.
      - Logs a synthetic `BatchSensitivity` output, capturing baseline, improved probability, delta, and candidate details.
  - **Summarization:**
    - If `LLMClient` is present and enabled, uses it to generate a natural‑language answer from the question and structured payload.
    - Otherwise, uses a small deterministic template:
      - Prediction: “A has X% win probability vs B…”
      - Strategy: baseline and improved probabilities plus suggested deltas in serve mix and attack style.

This architecture separates **reasoning about which tools to run** from **the tools themselves**, making it easy to:

- Swap in more advanced planning logic.
- Integrate additional tools (e.g. batch experiments, visualization).
- Maintain a clear, auditable execution trace.

---

### 8. PCSP and modeling choices – Key design decisions

- **Discrete rally‑level model**
  - The PCSP template models rallies as independent probabilistic events with fixed win probabilities conditioned on server.
  - Game and match structure follows badminton rules (21 points, 2‑point lead, cap at 30, best‑of‑odd games).

- **Style‑driven probabilities instead of raw frequencies**
  - Rather than treat per‑match rally probabilities as immutable, the system:
    - Aggregates historical events to estimate *baseline* serve/receive success.
    - Uses **serve mix** and **rally style** differences, scaled by learned **influence weights**, to adjust effective rally win probabilities.
  - This makes it possible to **simulate hypothetical style changes** (e.g. attack more, serve safer) without requiring historical matches for every possible style.

- **Linear delta + logistic / PAT probability**
  - In mock mode, a logistic layer is used to approximate how local changes map into match‑level probabilities, making experiments fast and deterministic.
  - In real mode, PAT’s model checking gives an exact or numerically accurate **reachability probability** for winning the match.

- **L1‑bounded strategy search**
  - Constraining adjustments by L1 distance ensures that recommended strategies are:
    - Small perturbations around the current style, not unrealistic overhauls.
    - Easy to communicate (“serve X% more short, be Y% more aggressive in rallies”).

- **Full artifact logging**
  - Every run creates a self‑contained directory with:
    - Inputs, stats, parameters, PCSP file, PAT outputs, and summaries.
  - This supports:
    - Offline analysis and plotting (`coach/analysis/*` modules).
    - Reproducibility and debugging of PAT or data issues.

---

### 9. How everything fits together (end‑to‑end)

1. **User question** – e.g. “What are the chances that Player A beats Player B?” or “How should A adjust strategy to beat B?”.
2. **Planner** – parses the question, identifies players and task type, builds a tool plan.
3. **Executor** – runs the plan with Python tools:
   - loads data via `LocalCSVAdapter`,
   - builds `MatchupParams` / `MatchupStats`,
   - renders a PCSP model from the template,
   - runs PAT (or the mock),
   - optionally runs strategy search by trying multiple adjusted parameter sets.
4. **PCSP + PAT** – compute the match‑win probability as a reachability probability.
5. **Agent summary** – aggregates the results and converts them into a coaching‑style explanation, along with structured JSON artifacts in a run directory.

This combination of **data‑driven parametric modeling**, **PCSP/PAT reachability analysis**, and an **LLM‑aware planning layer** is what makes the AI Coach both **interpretable** and **programmable** from natural language.

