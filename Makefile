# Simple automation for your batch pipeline + web app

PY := python

.PHONY: help setup dirs features labels train score api ui all clean

help:
	@echo "Targets:"
	@echo "  setup     - install Python deps from requirements.txt"
	@echo "  dirs      - create data/models/reports folders"
	@echo "  features  - compute microstructure + options features"
	@echo "  labels    - compute forward-return labels"
	@echo "  train     - train the 1d model"
	@echo "  score     - generate predictions with the trained model"
	@echo "  api       - start FastAPI on :8000"
	@echo "  ui        - start Next.js dev server on :3000"
	@echo "  all       - features + labels + train + score (end-to-end)"
	@echo "  clean     - remove generated artifacts (safe)"

setup:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

dirs:
	@mkdir -p data/raw data/interim data/processed models reports/metrics

features: dirs
	$(PY) -m orderflow.features.microstructure
	$(PY) -m orderflow.features.options_flow

labels: dirs
	$(PY) -m orderflow.features.labeling

train: dirs
	$(PY) -m orderflow.modeling.train

score: dirs
	$(PY) -m orderflow.modeling.score

api:
	uvicorn orderflow.serving.api:app --reload --port 8000

ui:
	cd app && npm install && npm run dev

all: features labels train score

clean:
	@rm -rf data/processed/* reports/* models/*
	@echo "Cleaned generated artifacts."
