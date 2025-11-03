"""
options_flow.py
---------------
Computes options/derivatives flow features at the same cadence as your bars (hourly/daily).

This module expects a pre-aggregated table per timestamp and symbol with columns like:
  ['timestamp','symbol','put_volume','call_volume',
   'at_ask_volume','at_bid_volume','total_volume',
   'iv_atm','iv_25d_call','iv_25d_put','open_interest']

If you don’t have all of these, the code handles missing columns gracefully.

Input:
  data/interim/options_agg.parquet

Output:
  data/processed/options_features.parquet
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from orderflow.utils.config import get_config
from orderflow.utils.io import read_parquet, write_parquet, load_or_empty_parquet


def _safe_ratio(num: pd.Series, den: pd.Series, name: str) -> pd.Series:
    den = den.replace(0, np.nan)
    out = (num / den).fillna(0.0)
    return out.rename(name)


def _diff(series: pd.Series, periods: int = 1, name: str | None = None) -> pd.Series:
    out = series.diff(periods)
    return out.rename(name or f"{series.name}_diff{periods}")


def _zscore(series: pd.Series, n: int, name: str) -> pd.Series:
    roll = series.rolling(n, min_periods=max(2, n // 3))
    z = (series - roll.mean()) / roll.std(ddof=0)
    return z.rename(name)


def _features_for_one(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    # Ensure DatetimeIndex
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df = df.sort_index()

    # Required-like columns (fallback to zeros if missing)
    def col(name, default=0.0):
        return df[name] if name in df.columns else pd.Series(default, index=df.index, name=name)

    put_vol  = col("put_volume")
    call_vol = col("call_volume")
    ask_vol  = col("at_ask_volume")
    bid_vol  = col("at_bid_volume")
    tot_vol  = col("total_volume", default=(put_vol + call_vol))

    # Core flows
    pcr = _safe_ratio(put_vol, call_vol.replace(0, np.nan), "pcr")
    at_ask_bias = _safe_ratio(ask_vol - bid_vol, tot_vol.replace(0, np.nan), "at_ask_bias")

    # Intensity (scaled)
    vol_intensity = (tot_vol.rolling(5, min_periods=1).mean()).rename("opt_vol_intensity")

    # Implied vol + skew proxies
    iv_atm      = col("iv_atm", default=np.nan)
    iv25c       = col("iv_25d_call", default=np.nan)
    iv25p       = col("iv_25d_put", default=np.nan)
    skew_25d    = (iv25p - iv25c).rename("skew_25d")
    d_iv_atm    = _diff(iv_atm, periods=1, name="d_iv_atm")
    d_skew_25d  = _diff(skew_25d, periods=1, name="d_skew_25d")

    # Open interest change (uses level today vs previous bar)
    oi          = col("open_interest", default=np.nan)
    d_oi        = _diff(oi, periods=1, name="d_oi")

    # Z-scores to capture extremes
    pcr_z_50    = _zscore(pcr, 50, "pcr_z_50")
    bias_z_50   = _zscore(at_ask_bias, 50, "at_ask_bias_z_50")

    out = pd.concat(
        [pcr, at_ask_bias, vol_intensity, iv_atm, skew_25d, d_iv_atm, d_skew_25d, oi, d_oi, pcr_z_50, bias_z_50],
        axis=1
    )
    return out


def _by_symbol(opts: pd.DataFrame, freq: str) -> pd.DataFrame:
    parts = []
    for sym, g in opts.groupby("symbol", sort=False):
        feats = _features_for_one(g, freq=freq).assign(symbol=sym)
        parts.append(feats)
    return pd.concat(parts).sort_index()


def main():
    cfg = get_config()
    freq = cfg.frequency

    # Load aggregated options/derivs data (or empty if not present yet)
    opts = load_or_empty_parquet("data/interim/options_agg.parquet")

    if opts.empty:
        # Create an empty shell with the expected index so downstream joins don’t break
        print("options_agg.parquet not found or empty; writing empty options_features.parquet.")
        write_parquet(pd.DataFrame(), "data/processed/options_features.parquet")
        return

    if "symbol" in opts.columns:
        feats = _by_symbol(opts, freq=freq)
    else:
        feats = _features_for_one(opts, freq=freq)

    write_parquet(feats, "data/processed/options_features.parquet")
    print("Wrote options features to data/processed/options_features.parquet")


if __name__ == "__main__":
    main()
