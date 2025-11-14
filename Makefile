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
	@echo "  setup     - Install Python package (editable) + backend reqs, and install UI deps"
	@echo "  demo      - Generate synthetic demo data (bars + empty options)"
	@echo "  features  - Build microstructure + options features"
	@echo "  labels    - Build forward-return labels"
	@echo "  train     - Train 1D model"
	@echo "  score     - Score 1D model -> preds"
	@echo "  all       - End-to-end: features -> labels -> train -> score"
	@echo "  api       - Start FastAPI backend (dev)"
	@echo "  ui        - Start Next.js frontend (dev)"
	@echo "  clean     - Remove intermediate artifacts"

setup:
	$(PIP) install -e .
	$(PIP) install -r requirements.txt
	cd $(APP_DIR) && npm install

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
	# FastAPI dev server, listen on all interfaces for Codespaces
	$(PY) -m uvicorn orderflow.serving.api:app --host 0.0.0.0 --port $(API_PORT) --reload

ui:
	# Next dev server; API base URL can be overridden from env
	cd $(APP_DIR) && \
	NEXT_PUBLIC_API_BASE_URL=$${NEXT_PUBLIC_API_BASE_URL:-http://localhost:$(API_PORT)} \
	npm run dev -- --port $(UI_PORT)

clean:
	rm -rf data/interim/* data/processed/* models/* reports/*
