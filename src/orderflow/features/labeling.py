"""
Labeling: forward returns for multiple horizons.
- Reads bars from data/interim/bars.parquet (expects columns: timestamp, symbol, close)
- Computes forward returns per horizon (1d/1w/1m) using calendar time deltas
- Writes data/processed/labels.parquet
"""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

from orderflow.utils.config import get_config
from orderflow.utils.io import read_parquet, write_parquet


# ---------------------------------------------------------------------
# Local helper
# ---------------------------------------------------------------------

def ensure_parent_dir(path: str | Path) -> None:
    """Ensure the parent directory of `path` exists."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Labeling logic
# ---------------------------------------------------------------------

HORIZON_TO_TIMEDelta = {
    "1d": pd.Timedelta(days=1),
    "1w": pd.Timedelta(weeks=1),
    "1m": pd.Timedelta(days=30),  # approximate monthly horizon
}


def _ensure_datetime(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def _forward_return_for_horizon(bars: pd.DataFrame, horizon: str) -> pd.Series:
    """
    Compute forward return for one horizon as a named Series indexed by (timestamp, symbol).
    ret_h = close(t + horizon) / close(t) - 1
    """
    if horizon not in HORIZON_TO_TIMEDelta:
        raise ValueError(f"Unknown horizon '{horizon}'. Expected one of {list(HORIZON_TO_TIMEDelta)}")

    delta = HORIZON_TO_TIMEDelta[horizon]

    df = bars[["timestamp", "symbol", "close"]].copy()
    df = _ensure_datetime(df, "timestamp")
    df = df.dropna(subset=["timestamp", "symbol", "close"]).sort_values(["symbol", "timestamp"])

    # Create future frame shifted back by delta
    future = df.copy()
    future["timestamp"] = future["timestamp"] - delta
    future = future.rename(columns={"close": "close_future"})

    # merge_asof aligns each t with t+delta
    merged = pd.merge_asof(
        df.sort_values("timestamp"),
        future.sort_values("timestamp"),
        by="symbol",
        on="timestamp",
        direction="forward",
    ).sort_values(["symbol", "timestamp"])

    ret = (merged["close_future"] / merged["close"] - 1.0).rename(f"ret_{horizon}")

    # MultiIndex Series
    ret.index = pd.MultiIndex.from_frame(merged[["timestamp", "symbol"]], names=["timestamp", "symbol"])
    return ret


def compute_labels(bars: pd.DataFrame, horizons: list[str]) -> pd.DataFrame:
    """Return DataFrame indexed by (timestamp, symbol) with columns ret_<horizon>."""
    series_list = []
    for h in horizons:
        s = _forward_return_for_horizon(bars, h)
        series_list.append(s)

    labels = pd.concat(series_list, axis=1)
    labels = labels.sort_index()
    return labels


# ---------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------

def main():
    cfg = get_config()

    bars_path = os.path.join("data", "interim", "bars.parquet")
    out_path = os.path.join("data", "processed", "labels.parquet")
    ensure_parent_dir(out_path)

    if not os.path.exists(bars_path):
        raise FileNotFoundError(f"{bars_path} not found. Run scripts/make_demo_data.py first.")

    bars = read_parquet(bars_path)
    bars = _ensure_datetime(bars, "timestamp")

    horizons = cfg.get("horizons", ["1d", "1w", "1m"])
    labels = compute_labels(bars, horizons=horizons)

    write_parquet(labels.reset_index(), out_path)
    print(f"Wrote labels to {out_path}")


if __name__ == "__main__":
    main()
