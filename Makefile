VENV_PYTHON := $(shell if [ -x .venv/bin/python ]; then printf '%s' .venv/bin/python; else printf '%s' python3; fi)
PYTHON ?= $(VENV_PYTHON)
PYTEST := PYTHONNOUSERSITE=1 $(PYTHON) -m pytest
RUFF := PYTHONNOUSERSITE=1 $(PYTHON) -m ruff

.PHONY: install lint test validate-inference predict-demo strategy-demo chat experiments

install:
	$(PYTHON) -m pip install -e .[dev]

lint:
	$(RUFF) check coach scripts tests

test:
	$(PYTEST) -q

validate-inference:
	PYTHONNOUSERSITE=1 $(PYTHON) scripts/validate_inference_thresholds.py

predict-demo:
	$(PYTHON) -m coach.cli predict --a "Viktor AXELSEN" --b "Kento MOMOTA" --mode mock --window 30

strategy-demo:
	$(PYTHON) -m coach.cli strategy --a "Viktor AXELSEN" --b "Kento MOMOTA" --mode mock --window 30 --budget 60

chat:
	$(PYTHON) -m coach.cli chat --mode mock

experiments:
	$(PYTHON) -m coach.cli experiments --mode mock --output-dir runs/experiments
