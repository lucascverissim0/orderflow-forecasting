"""
score.py
--------
Loads the trained 1d model and produces probability scores for the latest data.

Inputs:
  - models/xgb_1d.pkl
  - data/processed/microstructure.parquet
  - data/processed/options_features.parquet  (optional)
  - data/processed/labels.parquet            (optional; used only to trim lookahead leakage)

Outputs:
  - data/processed/preds_1d.parquet
    columns: ['proba_up_1d','signal_1d','timestamp','symbol'(opt)]

Notes:
- Aligns feature columns to the trained model (adds missing as 0, drops extras).
- Applies a simple decision rule: signal = 1 if proba_up >= 0.55, -1 if <= 0.45, else 0.
"""

from __future__ import annotations
import os
import pickle
import numpy as np
import pandas as pd
from typing import List

from orderflow.utils.config import get_config
from orderflow.utils.io import read_parquet, write_parquet, load_or_empty_parquet


def _join_features() -> pd.DataFrame:
    micro = read_parquet("data/processed/microstructure.parquet")
    opts = load_or_empty_parquet("data/processed/options_features.parquet")

    # Ensure timestamp index
    if "timestamp" in micro.columns:
        micro = micro.set_index("timestamp").sort_index()
    if not opts.empty and "timestamp" in opts.columns:
        opts = opts.set_index("timestamp").sort_index()

    if opts.empty:
        feats = micro.copy()
    else:
        if "symbol" in micro.columns:
            feats = (
                micro.reset_index()
                .merge(opts.reset_index(), on=["timestamp", "symbol"], how="left", suffixes=("", "_opt"))
                .set_index("timestamp")
                .sort_index()
            )
        else:
            feats = micro.join(opts, how="left", lsuffix="", rsuffix="_opt")

    # Conservative forward-fill for slow-moving features
    feats = feats.groupby(feats.get("symbol", pd.Series(index=feats.index))).apply(
        lambda g: g.ffill(limit=2)
    ).reset_index(level=0, drop=True) if "symbol" in feats.columns else feats.ffill(limit=2)

    return feats


def _prepare_matrix_for_inference(feats: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """
    Conform the features to the model's expected columns.
    - Add missing columns (0.0)
    - Keep order identical
    - Drop non-numeric and target-like columns
    """
    X = feats.copy()
    if "symbol" in X.columns:
        X = X.drop(columns=["symbol"])

    # Keep only numeric
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Add any missing features the model expects
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0

    # Drop extras not in the model
    X = X[feature_names]

    return X


def main():
    cfg = get_config()

    # 1) Load model
    model_path = os.path.join(cfg.paths.models, "xgb_1d.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found: {model_path}. Run train.py first.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 2) Build features (same join as in train.py)
    feats = _join_features()

    # Keep a copy of identifiers for output
    idx = feats.index
    sym = feats["symbol"] if "symbol" in feats.columns else None

    # 3) Conform feature matrix to model columns
    model_feature_names = list(model.get_booster().feature_names) if hasattr(model, "get_booster") else list(feats.select_dtypes(include=[np.number]).columns)
    X = _prepare_matrix_for_inference(feats, model_feature_names)

    # 4) Predict probabilities
    proba_up = model.predict_proba(X)[:, 1]

    # 5) Simple decision rule for a clean signal
    signal = np.where(proba_up >= 0.55, 1, np.where(proba_up <= 0.45, -1, 0))

    # 6) Assemble output
    out = pd.DataFrame({
        "proba_up_1d": proba_up,
        "signal_1d": signal,
    }, index=idx)

    if sym is not None:
        out["symbol"] = sym.values

    # Optional: remove rows that would leak lookahead if labels exist (drop last horizon)
    labels = load_or_empty_parquet("data/processed/labels.parquet")
    if not labels.empty:
        # Keep only rows with known label timestamps to avoid scoring on incomplete bars
        if "timestamp" in labels.columns:
            labels = labels.set_index("timestamp")
        aligned = out.index.intersection(labels.index)
        out = out.loc[aligned]

    # 7) Write predictions
    write_parquet(out, "data/processed/preds_1d.parquet")
    print("Wrote predictions to data/processed/preds_1d.parquet")
    # Quick tail print
    print(out.tail(5))


if __name__ == "__main__":
    main()
