"""
train.py
--------
Trains a first-pass classifier to predict the sign of the **1d forward return**
using bar microstructure + options features.

Inputs (Parquet):
  - data/processed/microstructure.parquet
  - data/processed/options_features.parquet   (can be empty)
  - data/processed/labels.parquet             (must contain 'fwd_ret_1d')

Outputs:
  - models/xgb_1d.pkl                         (pickled model)
  - reports/metrics/model_1d.json             (AUC, accuracy, etc.)
  - reports/metrics/feature_importance_1d.csv (gain-based importance)
"""

from __future__ import annotations
import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List

from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from orderflow.utils.config import get_config
from orderflow.utils.io import read_parquet, write_parquet, load_or_empty_parquet


def _join_features() -> pd.DataFrame:
    """Join microstructure + options features on timestamp (and symbol if present)."""
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
        # Join with awareness of 'symbol'
        if "symbol" in micro.columns:
            on_cols = ["timestamp", "symbol"]
            feats = (
                micro.reset_index()
                .merge(opts.reset_index(), on=on_cols, how="left", suffixes=("", "_opt"))
                .set_index("timestamp")
                .sort_index()
            )
        else:
            feats = micro.join(opts, how="left", lsuffix="", rsuffix="_opt")

    # Forward-fill IV/skew etc. modestly to reduce NaNs (safe for daily/hourly)
    feats = feats.groupby(feats.get("symbol", pd.Series(index=feats.index))).apply(
        lambda g: g.ffill(limit=2)
    ).reset_index(level=0, drop=True) if "symbol" in feats.columns else feats.ffill(limit=2)

    return feats


def _load_labels() -> pd.DataFrame:
    labels = read_parquet("data/processed/labels.parquet")
    if "timestamp" in labels.columns:
        labels = labels.set_index("timestamp").sort_index()
    return labels


def _prepare_matrix(feats: pd.DataFrame, labels: pd.DataFrame, target_col: str = "fwd_ret_1d"):
    """Align X and y, create binary target (ret > 0)."""
    # Join on timestamp (+ symbol if present)
    if "symbol" in feats.columns and "symbol" in labels.columns:
        df = (
            feats.reset_index()
            .merge(labels.reset_index()[["timestamp", "symbol", target_col]], on=["timestamp", "symbol"], how="inner")
            .set_index("timestamp")
            .sort_index()
        )
    else:
        df = feats.join(labels[[target_col]], how="inner")

    # Drop columns not usable as numeric features
    drop_cols = {"symbol"}
    X = df.drop(columns=[c for c in df.columns if c in drop_cols or c == target_col])
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna()
    # Align y
    y = (df.loc[X.index, target_col] > 0).astype(int)

    # Remove any leftover NaNs in y
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    return X, y


def _time_series_cv_score(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """Simple walk-forward CV for diagnostics."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs: List[float] = []
    accs: List[float] = []
    briers: List[float] = []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(y_te, proba))
        accs.append(accuracy_score(y_te, pred))
        briers.append(brier_score_loss(y_te, proba))

    return {
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "cv_acc_mean": float(np.mean(accs)),
        "cv_brier_mean": float(np.mean(briers)),
    }


def main():
    cfg = get_config()
    target_col = "fwd_ret_1d"

    # 1) Build feature matrix
    feats = _join_features()
    labels = _load_labels()

    if target_col not in labels.columns:
        raise ValueError(f"Target '{target_col}' not found in labels.parquet")

    X, y = _prepare_matrix(feats, labels, target_col=target_col)
    if len(X) < 200:
        raise ValueError("Not enough samples to train (need at least ~200 rows).")

    # 2) Model from config (with sensible defaults)
    params = {
        "max_depth": 3,
        "n_estimators": 200,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "n_jobs": 4,
        "random_state": 42,
    }
    # Allow overriding via configs/settings.yaml
    params.update(cfg.model.params or {})
    model = XGBClassifier(**params)

    # 3) Quick time-series CV diagnostics
    metrics_cv = _time_series_cv_score(model, X, y, n_splits=5)

    # 4) Fit on full data and persist
    model.fit(X, y)
    os.makedirs(cfg.paths.models, exist_ok=True)
    with open(os.path.join(cfg.paths.models, "xgb_1d.pkl"), "wb") as f:
        pickle.dump(model, f)

    # 5) Train-set metrics (for sanity)
    proba_full = model.predict_proba(X)[:, 1]
    pred_full = (proba_full >= 0.5).astype(int)
    metrics_full = {
        "train_auc": float(roc_auc_score(y, proba_full)),
        "train_acc": float(accuracy_score(y, pred_full)),
        "train_brier": float(brier_score_loss(y, proba_full)),
    }

    # 6) Save metrics + feature importance
    os.makedirs(cfg.paths.reports, exist_ok=True)
    os.makedirs(os.path.join(cfg.paths.reports, "metrics"), exist_ok=True)

    out_metrics = {**metrics_cv, **metrics_full, "n_samples": int(len(X)), "n_features": int(X.shape[1])}
    with open(os.path.join(cfg.paths.reports, "metrics", "model_1d.json"), "w") as f:
        json.dump(out_metrics, f, indent=2)

    try:
        fi = pd.DataFrame({
            "feature": X.columns,
            "importance_gain": model.get_booster().get_score(importance_type="gain").values()
        })
        # Align—missing features in booster.get_score won’t be listed; handle robustly
        booster_map = model.get_booster().get_score(importance_type="gain")
        fi = pd.DataFrame({
            "feature": list(booster_map.keys()),
            "importance_gain": list(booster_map.values())
        }).sort_values("importance_gain", ascending=False)
        write_parquet(fi, os.path.join(cfg.paths.reports, "metrics", "feature_importance_1d.parquet"))
        fi.to_csv(os.path.join(cfg.paths.reports, "metrics", "feature_importance_1d.csv"), index=False)
    except Exception as e:
        print(f"Feature importance export skipped: {e}")

    print("Training complete.")
    print(json.dumps(out_metrics, indent=2))


if __name__ == "__main__":
    main()
