"""
labeling.py
-----------
Computes **forward returns** for multiple horizons (e.g., 1d/1w/1m) from bar data.
- Uses log returns: ln(C_{t+H} / C_t)
- Works with hourly or daily bars (configured in `configs/settings.yaml`)
- Handles single-symbol frames (DatetimeIndex) or multi-symbol frames with a 'symbol' column.

Input:
  data/interim/bars.parquet  # columns: ['timestamp','symbol','open','high','low','close','volume']
                             # or index: DatetimeIndex with 'close' (+ optional 'symbol' column)

Output:
  data/processed/labels.parquet  # columns: ['fwd_ret_1d','fwd_ret_1w','fwd_ret_1m', ...]
"""

from __future__ import annotations
import math
from typing import Dict, List
import pandas as pd

from orderflow.utils.config import get_config
from orderflow.utils.io import read_parquet, write_parquet


# ---------- Helpers ----------

def _horizon_periods(freq: str, horizon: str) -> int:
    """
    Map a human horizon ('1d','1w','1m') to a number of bars given the pipeline frequency.
    This assumes regular bars (hourly/daily). For trading vs calendar days, we choose calendar-based
    approximations which are robust for swing horizons:
      - 1d = 1 day
      - 1w = 7 days
      - 1m = 30 days
    """
    freq = freq.lower()
    if freq not in {"1h", "1d"}:
        raise ValueError(f"Unsupported frequency '{freq}'. Use '1h' or '1d'.")

    day_bars = 24 if freq == "1h" else 1

    if horizon == "1d":
        return day_bars * 1
    if horizon == "1w":
        return day_bars * 7
    if horizon == "1m":
        return day_bars * 30

    # Accept numeric suffix like '3d', '2w', '6m'
    unit = horizon[-1].lower()
    val = int(horizon[:-1])
    if unit == "d":
        return day_bars * val
    if unit == "w":
        return day_bars * (7 * val)
    if unit == "m":
        return day_bars * (30 * val)

    raise ValueError(f"Unrecognized horizon format: {horizon}")


def _forward_log_return(close: pd.Series, periods: int) -> pd.Series:
    """
    Compute forward log return over 'periods' bars:
      ln(close.shift(-periods) / close)
    """
    future = close.shift(-periods)
    return (future / close).apply(lambda x: math.log(x) if pd.notna(x) and x > 0 else pd.NA)


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Bars must have a DatetimeIndex or a 'timestamp' column.")
    return df.sort_index()


# ---------- Main ----------

def compute_labels(bars: pd.DataFrame, horizons: List[str], freq: str) -> pd.DataFrame:
    """
    Returns a DataFrame with forward-return columns for each horizon.
    Preserves a 'symbol' column if present; otherwise labels apply to the single series.
    """
    bars = _ensure_datetime_index(bars)

    label_cols: Dict[str, pd.Series] = {}

    if "symbol" in bars.columns:
        labels_list = []
        for sym, g in bars.groupby("symbol", sort=False):
            g = g.sort_index()
            for hz in horizons:
                p = _horizon_periods(freq, hz)
                colname = f"fwd_ret_{hz}"
                labels_list.append(
                    _forward_log_return(g["close"], p).rename(colname).to_frame().assign(symbol=sym)
                )
        labels = pd.concat(labels_list).sort_index()
        # Pivot multiple horizons into columns (keeping symbol column)
        labels = labels.pivot_table(index=["timestamp", "symbol"], values=list({s.name for s in labels_list}), aggfunc="first")
        labels = labels.reset_index()
        labels = labels.set_index("timestamp")
        return labels
    else:
        for hz in horizons:
            p = _horizon_periods(freq, hz)
            label_cols[f"fwd_ret_{hz}"] = _forward_log_return(bars["close"], p)

        labels = pd.DataFrame(label_cols, index=bars.index)
        return labels


def main():
    cfg = get_config()
    horizons = cfg.horizons
    freq = cfg.frequency

    # Load bars
    bars = read_parquet("data/interim/bars.parquet")

    # Compute labels
    labels = compute_labels(bars, horizons=horizons, freq=freq)

    # Write out
    write_parquet(labels, "data/processed/labels.parquet")
    print(f"Wrote labels for horizons {horizons} to data/processed/labels.parquet")


if __name__ == "__main__":
    main()
