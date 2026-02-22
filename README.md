# AI Badminton Coach (LLM + Formal Methods)

End-to-end badminton coaching pipeline where the LLM plans tool calls and PAT/PCSP# computes probabilities.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or:

```bash
pip install -r requirements.txt
```

## LLM Setup (Gemini)

For `coach chat` LLM planning/summarization, set a Gemini API key:

```bash
export GEMINI_API_KEY=your_key_here
```

Notes:
- Default chat model is `gemini-2.5-flash`.
- If no key is set, chat still works using deterministic heuristic planning.

## PAT Setup

1. Download PAT:
   - https://www.comp.nus.edu.sg/~pat/patdownload.htm
2. Set env vars (copy `.env.example` to `.env` and edit):

```bash
PAT_CONSOLE_PATH=/absolute/path/to/PAT.Console.exe   # or PAT3.Console.exe
PAT_USE_MONO=
MONO_PATH=mono
PAT_TIMEOUT_S=120
RUNS_DIR=runs
```

- `PAT_USE_MONO` empty = auto.
- Auto behavior:
  - Windows: no mono.
  - macOS/Linux + `.exe` PAT path: mono enabled.
- `PAT_CONSOLE_PATH` may point either to the console executable or the PAT install directory.
- On macOS with PAT3, the runner auto-applies a compatibility fallback for known Mono NESC startup issues.

3. Validate connectivity:

```bash
python scripts/pat_check.py
```

If check fails, script prints exact next steps.

## CLI

```bash
python -m coach.cli --help
```

### PAT direct run

Mock smoke test:

```bash
coach pat-run --pcsp examples/minimal.pcsp --mode mock
```

Real PAT run:

```bash
coach pat-run --pcsp runs/.../matchup.pcsp --mode real
```

### Prediction / strategy

```bash
coach predict --a "Viktor Axelsen" --b "Kento Momota" --mode mock
coach strategy --a "Viktor Axelsen" --b "Kento Momota" --mode mock --budget 60
```

Use `--mode real` to execute PAT Console.

## Run Artifacts

Every run writes `runs/<run_id>/` with:
- input `.pcsp`
- `pat_output.txt`
- `pat_stdout.txt`
- `pat_stderr.txt`
- `pat_run.json`
- `summary.json`

## Tests

No PAT installation required for tests.

```bash
pytest -q
```

## Real PAT Checklist

1. PAT downloaded and extracted.
2. `PAT_CONSOLE_PATH` points to `PAT.Console.exe`/`PAT3.Console.exe` (or PAT install directory).
3. macOS/Linux: Mono installed (`mono --version` works).
4. macOS/Linux + PAT3: Mono C# compiler available (`mcs --version`) for auto-compat fallback.
5. `python scripts/pat_check.py` passes (version + module smoke check).
6. `coach pat-run --pcsp examples/minimal.pcsp --mode real` returns probability.
7. `coach predict ... --mode real` and `coach strategy ... --mode real` run successfully.
