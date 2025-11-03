"""
microstructure.py
-----------------
Bar-based orderflow features for *hourly/daily* data (no tick/quotes required).

Outputs (per timestamp and symbol):
- cvd_proxy: Cumulative Volume Delta using bar direction (close vs open) to sign volume
- delta_vol: Signed volume per bar (building block of CVD)
- ret_1: One-bar log return
- vol_rolling_{n}: Rolling realized volatility (std of returns) over n bars
- vwap: Running VWAP using (high+low+close)/3 as typical price
- bar_imbalance: Up/Down volume imbalance proxy in [-1, +1]
- cvd_z_{n}: Z-score of CVD over a rolling window (gives regime context)

Input:
  data/interim/bars.parquet
  Required columns:
    ['timestamp','symbol','open','high','low','close','volume']
  or a single-series DataFrame indexed by DatetimeIndex with those OHLCV columns.

Output:
  data/processed/microstructure.parquet
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from orderflow.utils.config import get_config
from orderflow.utils.io import read_parquet, write_parquet


def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Bars must have a DatetimeIndex or a 'timestamp' column.")
    return df.sort_index()


def _cvd_proxy(df: pd.DataFrame) -> pd.Series:
    """
    Signs volume by bar direction:
      +volume if close > open
      -volume if close < open
      0 if equal
    Then cumulatively sums.
    """
    sign = np.sign(df["close"] - df["open"])
    delta = sign * df["volume"]
    return delta.cumsum().rename("cvd_proxy"), delta.rename("delta_vol")


def _bar_imbalance(df: pd.DataFrame) -> pd.Series:
    """
    Up/Down volume imbalance proxy in [-1,+1].
    """
    up = np.where(df["close"] > df["open"], df["volume"], 0.0)
    dn = np.where(df["close"] < df["open"], df["volume"], 0.0)
    denom = (up + dn)
    with np.errstate(divide="ignore", invalid="ignore"):
        imb = (up - dn) / np.where(denom == 0, np.nan, denom)
    return pd.Series(imb, index=df.index, name="bar_imbalance").fillna(0.0)


def _running_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Running VWAP using typical price = (H+L+C)/3.
    """
    typ = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_pv = (typ * df["volume"]).cumsum()
    cum_v = (df["volume"]).cumsum().replace(0, np.nan)
    vwap = (cum_pv / cum_v).rename("vwap")
    return vwap


def _rolling_vol(ret: pd.Series, n: int) -> pd.Series:
    """
    Realized volatility proxy: rolling std of 1-bar log returns.
    """
    return ret.rolling(n, min_periods=max(2, n // 3)).std().rename(f"vol_rolling_{n}")


def _zscore(series: pd.Series, n: int, name: str) -> pd.Series:
    roll = series.rolling(n, min_periods=max(2, n // 3))
    z = (series - roll.mean()) / roll.std(ddof=0)
    return z.rename(name)


def _compute_for_one(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = _ensure_dt_index(df)

    # One-bar log return
    ret_1 = np.log(df["close"]).diff().rename("ret_1")

    # CVD proxy & signed volume
    cvd, delta_vol = _cvd_proxy(df)

    # Rolling CVD z-scores for regime/context
    cvd_z_50  = _zscore(cvd, 50,  "cvd_z_50")
    cvd_z_200 = _zscore(cvd, 200, "cvd_z_200")

    # Bar imbalance and VWAP
    bar_imb = _bar_imbalance(df)
    vwap = _running_vwap(df)

    # Volatility horizons (pick sensible windows for 1h/1d)
    if freq.lower() == "1h":
        vol_wins = [24, 24*7]         # ~1d, ~1w
    else:  # "1d"
        vol_wins = [5, 22]            # ~1w, ~1m

    vol_cols = [ _rolling_vol(ret_1, n) for n in vol_wins ]

    out = pd.concat(
        [ret_1, delta_vol, cvd, cvd_z_50, cvd_z_200, bar_imb, vwap] + vol_cols,
        axis=1
    )
    return out


def _by_symbol(bars: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Handles multi-symbol input (expects 'symbol' column).
    """
    parts = []
    for sym, g in bars.groupby("symbol", sort=False):
        feats = _compute_for_one(g, freq=freq)
        feats = feats.assign(symbol=sym)
        parts.append(feats)
    out = pd.concat(parts).sort_index()
    return out


def main():
    cfg = get_config()
    freq = cfg.frequency

    bars = read_parquet("data/interim/bars.parquet")

    if "symbol" in bars.columns:
        feats = _by_symbol(bars, freq=freq)
    else:
        feats = _compute_for_one(bars, freq=freq)

    write_parquet(feats, "data/processed/microstructure.parquet")
    print("Wrote microstructure features to data/processed/microstructure.parquet")


if __name__ == "__main__":
    main()
