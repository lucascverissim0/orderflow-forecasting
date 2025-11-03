"""
api.py
------
Minimal FastAPI backend to serve symbols, timeseries (features), and predictions
to the frontend. Reads Parquet artifacts produced by the batch pipeline.

Endpoints:
- GET /health
- GET /symbols
- GET /timeseries?symbol=BTC-USD&start=2024-01-01&end=2025-01-01
- GET /predictions?symbol=BTC-USD&start=2024-01-01&end=2025-01-01
- GET /latest?symbol=BTC-USD

Run locally:
  uvicorn orderflow.serving.api:app --reload --port 8000
"""

from __future__ import annotations
from typing import Optional, List
import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from orderflow.utils.config import get_config
from orderflow.utils.io import load_or_empty_parquet, read_parquet


app = FastAPI(title="Orderflow Forecasting API", version="0.1.0")

# CORS (allow local Next.js, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------ helpers ------------ #

def _dtindex(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df.sort_index()

def _filter(df: pd.DataFrame, symbol: Optional[str], start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if df.empty:
        return df
    if symbol and "symbol" in df.columns:
        df = df[df["symbol"] == symbol]
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]
    return df

def _join_features() -> pd.DataFrame:
    micro = load_or_empty_parquet("data/processed/microstructure.parquet")
    opts  = load_or_empty_parquet("data/processed/options_features.parquet")
    preds = load_or_empty_parquet("data/processed/preds_1d.parquet")

    if not micro.empty:
        micro = _dtindex(micro)
    if not opts.empty:
        opts = _dtindex(opts)
    if not preds.empty:
        preds = _dtindex(preds)

    # Start with micro
    df = micro.copy()

    # Join options
    if not opts.empty:
        if "symbol" in df.columns:
            df = (
                df.reset_index()
                  .merge(opts.reset_index(), on=["timestamp","symbol"], how="left", suffixes=("", "_opt"))
                  .set_index("timestamp")
                  .sort_index()
            )
        else:
            df = df.join(opts, how="left", lsuffix="", rsuffix="_opt")

    # Join predictions
    if not preds.empty:
        if "symbol" in df.columns and "symbol" in preds.columns:
            df = (
                df.reset_index()
                  .merge(preds.reset_index(), on=["timestamp","symbol"], how="left")
                  .set_index("timestamp")
                  .sort_index()
            )
        else:
            df = df.join(preds[["proba_up_1d","signal_1d"]], how="left")

    return df


# ------------ routes ------------ #

@app.get("/health")
def health():
    cfg = get_config()
    return {"status": "ok", "symbols": cfg.symbols, "frequency": cfg.frequency}

@app.get("/symbols")
def symbols():
    cfg = get_config()
    # Prefer symbols found in microstructure if available
    micro = load_or_empty_parquet("data/processed/microstructure.parquet")
    if not micro.empty and "symbol" in micro.columns:
        syms = sorted([str(s) for s in micro["symbol"].dropna().unique().tolist()])
        if syms:
            return {"symbols": syms}
    return {"symbols": cfg.symbols}

@app.get("/timeseries")
def timeseries(
    symbol: Optional[str] = Query(None, description="Symbol to filter (e.g., BTC-USD)"),
    start: Optional[str] = Query(None, description="ISO start date (e.g., 2024-01-01)"),
    end: Optional[str]   = Query(None, description="ISO end date (e.g., 2025-01-01)"),
    limit: int = Query(5000, ge=1, le=20000, description="Max rows returned"),
):
    df = _join_features()
    if df.empty:
        return {"rows": []}

    df = _filter(df, symbol, start, end)
    if df.empty:
        return {"rows": []}

    # keep a sensible subset for the UI; extend as needed
    cols_preferred = [
        "symbol","open","high","low","close","volume",
        "ret_1","delta_vol","cvd_proxy","cvd_z_50","cvd_z_200","bar_imbalance","vwap",
        "pcr","at_ask_bias","opt_vol_intensity","iv_atm","skew_25d","d_iv_atm","d_skew_25d",
        "proba_up_1d","signal_1d"
    ]
    cols = [c for c in cols_preferred if c in df.columns]
    out = df[cols].tail(limit).reset_index().rename(columns={"index": "timestamp"})
    # ensure timestamp is ISO
    out["timestamp"] = out["timestamp"].astype("datetime64[ns]").dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT").astype(str)
    return {"rows": out.to_dict(orient="records")}

@app.get("/predictions")
def predictions(
    symbol: Optional[str] = Query(None),
    start: Optional[str] = Query(None),
    end: Optional[str]   = Query(None),
    limit: int = Query(5000, ge=1, le=20000),
):
    preds = load_or_empty_parquet("data/processed/preds_1d.parquet")
    if preds.empty:
        return {"rows": []}
    preds = _dtindex(preds)
    preds = _filter(preds, symbol, start, end)
    if preds.empty:
        return {"rows": []}
    cols = ["symbol","proba_up_1d","signal_1d"] if "symbol" in preds.columns else ["proba_up_1d","signal_1d"]
    out = preds[cols].tail(limit).reset_index()
    out["timestamp"] = out["timestamp"].astype("datetime64[ns]").dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT").astype(str)
    return {"rows": out.to_dict(orient="records")}

@app.get("/latest")
def latest(symbol: Optional[str] = Query(None)):
    df = _join_features()
    if df.empty:
        raise HTTPException(status_code=404, detail="No data available.")
    if symbol and "symbol" in df.columns:
        df = df[df["symbol"] == symbol]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for symbol {symbol}.")
    row = df.tail(1)
    payload = row.reset_index().to_dict(orient="records")[0]
    # stringify timestamp
    payload["timestamp"] = str(payload["timestamp"])
    return payload
