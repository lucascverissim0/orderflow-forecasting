SHELL := /bin/bash

# Binaries & paths
PY        ?= python
PIP       ?= pip
APP_DIR   ?= app
API_PORT  ?= 8000
UI_PORT   ?= 3000

.PHONY: help setup demo features labels train score all api ui clean

help:
	@echo "Targets:"
	@echo "  setup     - Install Python package (editable) + reqs, and install UI deps"
	@echo "  demo      - Generate synthetic demo data (bars + empty options)"
	@echo "  features  - Build microstructure + options features"
	@echo "  labels    - Build forward-return labels"
	@echo "  train     - Train 1D model"
	@echo "  score     - Score 1D model -> preds"
	@echo "  all       - End-to-end: features -> labels -> train -> score"
	@echo "  api       - Start FastAPI server on port $(API_PORT)"
	@echo "  ui        - Start Next.js dev server on port $(UI_PORT)"
	@echo "  clean     - Remove models/reports/processed artifacts"

setup:
	$(PIP) install -e .
	@if [ -f requirements.txt ]; then $(PIP) install -r requirements.txt; fi
	@if [ -f $(APP_DIR)/package.json ]; then npm ci --prefix $(APP_DIR); fi

demo:
	$(PY) scripts/make_demo_data.py

features:
	$(PY) -m orderflow.features.microstructure
	$(PY) -m orderflow.features.options_flow

labels:
	$(PY) -m orderflow.features.labeling

train:
	$(PY) -m orderflow.modeling.train

score:
	$(PY) -m orderflow.modeling.score

all: features labels train score

api:
	$(PY) -m uvicorn orderflow.serving.api:app --reload --port $(API_PORT)

ui:
	npm --prefix $(APP_DIR) run dev -- --port $(UI_PORT)

clean:
	@rm -rf models/*.json reports/*.json data/processed/*.parquet 2>/dev/null || true
