"""
Minimal FastAPI backend + HTML dashboard for orderflow-forecasting (batch).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from orderflow.utils.config import get_config
from orderflow.utils.io import read_parquet

DATA_PROCESSED = Path("data/processed")

app = FastAPI(title="orderflow-forecasting API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- helpers ----------------------------------------------------------


def _ensure_dt(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
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


# ---------- HTML dashboard ---------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def root_page():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Orderflow Forecasting (Batch)</title>
  <style>
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
           background: #0f172a; color: #e5e7eb; margin: 0; padding: 24px; }
    h1 { font-size: 26px; margin-bottom: 4px; }
    h2 { font-size: 18px; margin-top: 24px; margin-bottom: 8px; }
    p { margin: 4px 0 8px 0; }
    .controls { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0; }
    select, button {
      padding: 4px 8px; border-radius: 4px; border: 1px solid #374151;
      background: #020617; color: inherit;
    }
    button { cursor: pointer; }
    button:disabled { opacity: 0.5; cursor: default; }
    .card { border: 1px solid #1f2937; border-radius: 8px; padding: 12px;
            margin-top: 8px; background: #020617; }
    table { border-collapse: collapse; width: 100%; font-size: 12px; margin-top: 8px; }
    th, td { border-bottom: 1px solid #1f2937; padding: 4px 6px; text-align: left; }
    th { font-weight: 600; color: #9ca3af; }
    .error { border-color: #fca5a5; background: #450a0a; color: #fecaca; }
    .status { font-size: 12px; color: #9ca3af; margin-top: 8px; }
  </style>
</head>
<body>
  <h1>Orderflow Forecasting (Batch)</h1>
  <p>Minimal dashboard served directly by FastAPI.</p>

  <div class="controls">
    <label>
      Symbol
      <select id="symbol"></select>
    </label>
    <button id="refresh">Refresh</button>
  </div>

  <div id="error" class="card error" style="display:none;"></div>

  <h2>Latest snapshot</h2>
  <div id="latest" class="card">
    <p>No data yet. Try Refresh after running the pipeline.</p>
  </div>

  <h2>Timeseries + 1d predictions</h2>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Timestamp (UTC)</th>
          <th>Close</th>
          <th>Volume</th>
          <th>CVD</th>
          <th>PCR</th>
          <th>At-Ask Bias</th>
          <th>Pred 1d</th>
        </tr>
      </thead>
      <tbody id="tbody">
        <tr><td colspan="7">No rows to display.</td></tr>
      </tbody>
    </table>
  </div>

  <div class="status" id="status"></div>

  <script>
    const symbolEl = document.getElementById("symbol");
    const refreshBtn = document.getElementById("refresh");
    const errorEl = document.getElementById("error");
    const latestEl = document.getElementById("latest");
    const tbodyEl = document.getElementById("tbody");
    const statusEl = document.getElementById("status");

    function showError(msg) {
      errorEl.textContent = msg;
      errorEl.style.display = "block";
    }
    function clearError() {
      errorEl.style.display = "none";
      errorEl.textContent = "";
    }
    async function fetchJSON(path) {
      const res = await fetch(path);
      if (!res.ok) throw new Error(path + " -> " + res.status);
      return res.json();
    }

    async function loadSymbols() {
      try {
        clearError();
        const syms = await fetchJSON("/symbols");
        symbolEl.innerHTML = "";
        if (!syms || syms.length === 0) {
          const opt = document.createElement("option");
          opt.value = "";
          opt.textContent = "No symbols";
          symbolEl.appendChild(opt);
          refreshBtn.disabled = true;
          statusEl.textContent = "No symbols from backend.";
          return;
        }
        syms.forEach(s => {
          const opt = document.createElement("option");
          opt.value = s;
          opt.textContent = s;
          symbolEl.appendChild(opt);
        });
        refreshBtn.disabled = false;
        statusEl.textContent = "Loaded " + syms.length + " symbols.";
      } catch (err) {
        console.error(err);
        showError(err.message || "Failed to fetch symbols");
      }
    }

    async function loadData() {
      const sym = symbolEl.value;
      if (!sym) return;
      clearError();
      statusEl.textContent = "Loading data for " + sym + "...";

      const params = new URLSearchParams({ symbol: sym });

      try {
        const [ts, preds, latest] = await Promise.all([
          fetchJSON("/timeseries?" + params.toString()),
          fetchJSON("/predictions?" + params.toString()),
          fetchJSON("/latest?symbol=" + encodeURIComponent(sym)),
        ]);

        const predByTs = new Map();
        (preds || []).forEach(p => predByTs.set(p.timestamp, p.pred_1d));

        tbodyEl.innerHTML = "";
        const rows = ts || [];
        if (rows.length === 0) {
          const tr = document.createElement("tr");
          const td = document.createElement("td");
          td.colSpan = 7;
          td.textContent = "No rows to display.";
          tr.appendChild(td);
          tbodyEl.appendChild(tr);
        } else {
          rows.forEach(r => {
            const tr = document.createElement("tr");
            const cells = [
              r.timestamp,
              r.close ?? "-",
              r.volume ?? "-",
              r.cvd ?? "-",
              r.pcr ?? "-",
              r.at_ask_bias ?? "-",
              predByTs.get(r.timestamp) ?? "-"
            ];
            cells.forEach(val => {
              const td = document.createElement("td");
              td.textContent = val;
              tr.appendChild(td);
            });
            tbodyEl.appendChild(tr);
          });
        }

        latestEl.innerHTML = "";
        if (!latest || !latest.timestamp) {
          latestEl.innerHTML = "<p>No data yet. Try Refresh after running the pipeline.</p>";
        } else {
          const fields = [
            ["Time", latest.timestamp],
            ["Close", latest.close],
            ["Volume", latest.volume],
            ["CVD", latest.cvd],
            ["PCR", latest.pcr],
            ["At-Ask Bias", latest.at_ask_bias],
            ["Pred 1d", latest.pred_1d],
          ];
          fields.forEach(([label, value]) => {
            if (value === undefined || value === null) return;
            const p = document.createElement("p");
            p.innerHTML = "<b>" + label + ":</b> " + value;
            latestEl.appendChild(p);
          });
        }

        statusEl.textContent = "Loaded " + rows.length + " rows for " + sym + ".";
      } catch (err) {
        console.error(err);
        showError(err.message || "Failed to fetch data");
        statusEl.textContent = "";
      }
    }

    refreshBtn.addEventListener("click", loadData);
    symbolEl.addEventListener("change", loadData);

    loadSymbols().then(loadData).catch(console.error);
  </script>
</body>
</html>
    """


# ---------- JSON endpoints ---------------------------------------------------


@app.get("/health")
def health():
    cfg = get_config()
    syms = list(getattr(cfg, "symbols", [])) or []
    return {"status": "ok", "symbols": syms}


@app.get("/symbols", response_model=List[str])
def symbols() -> List[str]:
    cfg = get_config()
    syms = list(getattr(cfg, "symbols", [])) or []
    return syms


@app.get("/timeseries")
def timeseries(
    symbol: str = Query(...),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    df = _merge_features()
    if df.empty:
        return []
    df = df[df["symbol"] == symbol]
    if df.empty:
        return []
    df = _filter_by_date(df, start, end)

    cols = ["timestamp", "symbol", "close", "volume", "cvd", "pcr", "at_ask_bias"]
    present = [c for c in cols if c in df.columns]
    if not present:
        return []

    out = df[present].copy()
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
            ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
            p = p[p["timestamp"] <= ts].sort_values("timestamp").tail(1)
            if not p.empty:
                row["pred_1d"] = float(p["pred_1d"].iloc[0])

    if row and isinstance(row.get("timestamp"), pd.Timestamp):
        row["timestamp"] = row["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ")

    return row or {}
