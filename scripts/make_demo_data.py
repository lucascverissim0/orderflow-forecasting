"""
make_demo_data.py
-----------------
Generate a small synthetic OHLCV dataset for multiple symbols so you can run:
  make all

Outputs:
  data/interim/bars.parquet
  data/interim/options_agg.parquet  (minimal shell; optional in pipeline)

The bars are a simple geometric random walk with reasonable scales per asset.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

from orderflow.utils.config import get_config
from orderflow.utils.io import write_parquet


# Scales for initial price + volume by symbol
DEFAULT_SCALES = {
    "BTC-USD":        {"price": 40000, "vol": 1_000},
    "ETH-USD":        {"price": 2500,  "vol": 2_000},
    "SPX500USD":      {"price": 5200,  "vol": 500_000},
    "NASDAQ100USD":   {"price": 18500, "vol": 400_000},
    "RUSSELL2000USD": {"price": 2100,  "vol": 350_000},
    "STOXX50EUSD":    {"price": 5000,  "vol": 200_000},
    "GER40EUR":       {"price": 18000, "vol": 180_000},
    "FTSE100GBP":     {"price": 8300,  "vol": 160_000},
    "XAUUSD":         {"price": 2400,  "vol": 50_000},
    "XAGUSD":         {"price": 28,    "vol": 80_000},
}

def simulate_bars(symbol: str, idx: pd.DatetimeIndex, price0: float, vol0: float) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    # daily-ish vol scaled by frequency
    dt = (idx[1] - idx[0]).total_seconds()
    day = 86400.0
    sigma_day = 0.02  # 2% daily vol baseline
    sigma = sigma_day * np.sqrt(dt / day)

    # geometric random walk on close
    rets = rng.normal(0.0, sigma, size=len(idx))
    close = price0 * np.exp(np.cumsum(rets))
    # open/high/low
    open_ = np.r_[price0, close[:-1]]
    # intra-bar noise
    hi_noise = rng.normal(0.001, 0.001, size=len(idx))
    lo_noise = rng.normal(-0.001, 0.001, size=len(idx))
    high = np.maximum.reduce([open_, close, open_ * (1 + hi_noise), close * (1 + hi_noise)])
    low  = np.minimum.reduce([open_, close, open_ * (1 + lo_noise),  close * (1 + lo_noise)])

    # volume with some autocorrelation
    base_vol = vol0
    vol = np.abs(rng.normal(base_vol, base_vol * 0.2, size=len(idx))).astype(float)

    df = pd.DataFrame({
        "timestamp": idx,
        "symbol": symbol,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    })
    return df

def main():
    cfg = get_config()
    freq = cfg.frequency.lower()  # "1h" or "1d"

    # date range: last ~180 days/hourly periods
    end = pd.Timestamp.utcnow().floor("h")
    if freq == "1h":
        start = end - pd.Timedelta(days=180)
    elif freq == "1d":
        start = end - pd.Timedelta(days=720)  # ~2 years of dailies
    else:
        raise ValueError("Only 1h or 1d supported in this demo script.")
    idx = pd.date_range(start=start, end=end, freq=freq, tz="UTC")

    frames = []
    for sym in cfg.symbols:
        sc = DEFAULT_SCALES.get(sym, {"price": 100.0, "vol": 10_000})
        frames.append(simulate_bars(sym, idx, sc["price"], sc["vol"]))

    bars = pd.concat(frames, ignore_index=True)
    # Ensure numeric and clean
    bars = bars.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    os.makedirs("data/interim", exist_ok=True)
    write_parquet(bars, "data/interim/bars.parquet")

    # Minimal empty options aggregate (so pipeline won't break if you don't have options yet)
    opts = pd.DataFrame(columns=[
        "timestamp","symbol","put_volume","call_volume",
        "at_ask_volume","at_bid_volume","total_volume",
        "iv_atm","iv_25d_call","iv_25d_put","open_interest"
    ])
    write_parquet(opts, "data/interim/options_agg.parquet")

    print("Wrote:")
    print("  - data/interim/bars.parquet")
    print("  - data/interim/options_agg.parquet (empty shell)")

if __name__ == "__main__":
    main()
