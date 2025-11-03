# Orderflow Forecasting (Batch)

A lightweight, **hourly/daily** (non-realtime) pipeline that ingests market data, builds **order-flow proxies** (e.g., CVD from bars) and **derivatives flow** features (PCR, at-ask/bid bias, IV/skew deltas), trains simple models for **1d/1w/1m** horizons, and serves results to a minimal web UI.

## What this repo will contain (short)
- `src/orderflow/data/`: batch ingestion + alignment to hourly/daily bars
- `src/orderflow/features/`: CVD proxy, options flow, liquidity metrics, labels
- `src/orderflow/modeling/`: walk-forward training + scoring
- `src/orderflow/serving/`: FastAPI endpoints for the UI
- `app/`: small React dashboard for visualizing signals

## Quick start (will expand as we add files)
1. **Python**: 3.10+ recommended.
2. Create a virtual env and install deps (we’ll add the dependency file next):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   # requirements.txt will be provided in the next step
