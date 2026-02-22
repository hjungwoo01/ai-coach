PYTHON ?= python3

.PHONY: install test predict-demo strategy-demo chat experiments

install:
	$(PYTHON) -m pip install -e .[dev]

test:
	pytest -q

predict-demo:
	$(PYTHON) -m coach.cli predict --a "Viktor AXELSEN" --b "Kento MOMOTA" --mode mock --window 30

strategy-demo:
	$(PYTHON) -m coach.cli strategy --a "Viktor AXELSEN" --b "Kento MOMOTA" --mode mock --window 30 --budget 60

chat:
	$(PYTHON) -m coach.cli chat --mode mock

experiments:
	$(PYTHON) -m coach.cli experiments --mode mock --output-dir runs/experiments
