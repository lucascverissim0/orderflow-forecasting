"""
FastAPI backend for the orderflow-forecasting MVP (batch).

Endpoints
- GET /health
- GET /symbols                     -> ["BTC-USD", "ETH-USD", ...]  (plain list)
- GET /timeseries?symbol=&start=&end=
- GET /predictions?symbol=&start=&end=
- GET /latest?symbol=

Notes
- CORS is permissive for local/Codespaces dev. Lock down for prod.
- All file reads are resilient; empty/missing inputs return [] / {}.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from orderflow.utils.config import get_config
from orderflow.utils.io import read_parquet

DATA_INTERIM = Path("data/interim")
DATA_PROCESSED = Path("data/processed")

app = FastAPI(title="orderflow-forecasting API", version="0.1.0")

# ---- Dev CORS (broad; tighten for prod) -------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ok for dev; prefer specific frontend origin(s) in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Helpers ----------------------------------------------------------------


def _ensure_dt(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'timestamp' and 'symbol' are normal columns."""
    if "timestamp" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "timestamp"})
        else:
            for alt in ("datetime", "time", "date", "ts"):
                if alt in df.columns:
                    df = df.rename(columns={alt: "timestamp"})
                    break

    if "symbol" not in df.columns:
        if isinstance(df.index, pd.MultiIndex) and "symbol" in df.index.names:
            df = df.reset_index()
        else:
            for alt in ("ticker", "asset", "sym"):
                if alt in df.columns:
                    df = df.rename(columns={alt: "symbol"})
                    break

    return df


def _read_micro() -> pd.DataFrame:
    p = DATA_PROCESSED / "microstructure.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["timestamp", "symbol"])
    df = read_parquet(str(p))
    df = _normalize_keys(df)
    df = _ensure_dt(df, "timestamp")
    return df


def _read_options_features() -> pd.DataFrame:
    p = DATA_PROCESSED / "options_features.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = read_parquet(str(p))
    df = _normalize_keys(df)
    df = _ensure_dt(df, "timestamp")
    return df


def _read_preds() -> pd.DataFrame:
    p = DATA_PROCESSED / "preds_1d.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["timestamp", "symbol", "pred_1d"])
    df = read_parquet(str(p))
    df = _normalize_keys(df)
    df = _ensure_dt(df, "timestamp")
    return df


def _merge_features() -> pd.DataFrame:
    micro = _read_micro()
    opts = _read_options_features()

    if micro.empty:
        return micro

    if not opts.empty:
        df = pd.merge(
            micro,
            opts,
            on=["timestamp", "symbol"],
            how="left",
            suffixes=("", "_opt"),
            validate="m:1",
        )
    else:
        df = micro

    if {"symbol", "timestamp"} <= set(df.columns):
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


def _filter_by_date(
    df: pd.DataFrame, start: Optional[str], end: Optional[str]
) -> pd.DataFrame:
    if start:
        df = df[df["timestamp"] >= pd.to_datetime(start, utc=True, errors="coerce")]
    if end:
        df = df[df["timestamp"] <= pd.to_datetime(end, utc=True, errors="coerce")]
    return df


# ---- Routes -----------------------------------------------------------------


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/symbols", response_model=List[str])
def symbols() -> List[str]:
    """
    Return a plain list so the UI dropdown populates cleanly.

    Uses config.symbols from configs/settings.yaml via get_config().
    """
    cfg = get_config()
    syms = list(getattr(cfg, "symbols", [])) or []
    return syms


@app.get("/timeseries")
def timeseries(
    symbol: str = Query(..., description="Symbol as in your config"),
    start: Optional[str] = Query(None, description="ISO date/time"),
    end: Optional[str] = Query(None, description="ISO date/time"),
):
    df = _merge_features()
    if df.empty:
        return []

    df = df[df["symbol"] == symbol]
    if df.empty:
        return []

    df = _filter_by_date(df, start, end)

    # Keep a small, UI-friendly set of columns if present
    cols = ["timestamp", "symbol", "close", "volume", "cvd", "pcr", "at_ask_bias"]
    present = [c for c in cols if c in df.columns]
    if not present:
        return []

    out = df[present].copy()

    # Convert timestamp to ISO strings for JSON
    if pd.api.types.is_datetime64_any_dtype(out["timestamp"]):
        out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        out["timestamp"] = pd.to_datetime(
            out["timestamp"], utc=True, errors="coerce"
        ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return out.to_dict(orient="records")


@app.get("/predictions")
def predictions(
    symbol: str = Query(...),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    preds = _read_preds()
    if preds.empty:
        return []

    preds = preds[preds["symbol"] == symbol]
    if preds.empty:
        return []

    preds = _filter_by_date(preds, start, end)

    out = preds[["timestamp", "symbol", "pred_1d"]].copy()

    if pd.api.types.is_datetime64_any_dtype(out["timestamp"]):
        out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        out["timestamp"] = pd.to_datetime(
            out["timestamp"], utc=True, errors="coerce"
        ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return out.to_dict(orient="records")


@app.get("/latest")
def latest(symbol: str = Query(...)):
    feats = _merge_features()
    preds = _read_preds()

    row = None
    if not feats.empty:
        f = feats[feats["symbol"] == symbol]
        if not f.empty:
            row = f.iloc[-1].to_dict()

    if row and not preds.empty:
        p = preds[preds["symbol"] == symbol]
        if not p.empty:
            # align on latest timestamp available in feats
            ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
            p = p[p["timestamp"] <= ts].sort_values("timestamp").tail(1)
            if not p.empty:
                row["pred_1d"] = float(p["pred_1d"].iloc[0])

    # normalize timestamp to string
    if row and isinstance(row.get("timestamp"), pd.Timestamp):
        row["timestamp"] = row["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ")

    return row or {}
