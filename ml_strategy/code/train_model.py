#!/usr/bin/env python3
"""
Meffs XAUUSD Multi-Timeframe – Training Script
==============================================
Trains a LightGBM model to predict the next M15 bar direction
using features from M5 + M15 + H1.

Data expected under:
  data/xauusd_m5/*.csv
  data/xauusd_m15/*.csv
  data/xauusd_h1/*.csv

CSV columns: timestamp,open,high,low,close  (timestamp = Unix ms)

Usage (from ml_strategy/):
  python code/train_model.py \\
    --m15_dir ../data/xauusd_m15 \\
    --h1_dir  ../data/xauusd_h1 \\
    --m5_dir  ../data/xauusd_m5 \\
    --out_dir model

Outputs:
  model/lgb_xauusd_multi_tf.txt
  model/lgb_xauusd_multi_tf.pkl
  model/feat_cols.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

try:
    import joblib
except ImportError:
    joblib = None


def add_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    df = df.copy()
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]

    for p in [1, 2, 3, 5, 8, 13]:
        df[f"{prefix}ret_{p}"] = c.pct_change(p)

    df[f"{prefix}rsi_14"] = RSIIndicator(c, 14).rsi()
    df[f"{prefix}rsi_7"] = RSIIndicator(c, 7).rsi()

    macd = MACD(c)
    df[f"{prefix}macd"] = macd.macd()
    df[f"{prefix}macd_sig"] = macd.macd_signal()
    df[f"{prefix}macd_hist"] = macd.macd_diff()

    bb = BollingerBands(c, 20, 2)
    df[f"{prefix}bb_pct"] = bb.bollinger_pband()
    df[f"{prefix}bb_width"] = bb.bollinger_wband()

    atr = AverageTrueRange(h, l, c, 14)
    df[f"{prefix}atr"] = atr.average_true_range()
    df[f"{prefix}atr_pct"] = df[f"{prefix}atr"] / c

    df[f"{prefix}ema_9"] = EMAIndicator(c, 9).ema_indicator()
    df[f"{prefix}ema_21"] = EMAIndicator(c, 21).ema_indicator()
    df[f"{prefix}sma_50"] = SMAIndicator(c, 50).sma_indicator()
    df[f"{prefix}ema9_21"] = (df[f"{prefix}ema_9"] - df[f"{prefix}ema_21"]) / c
    df[f"{prefix}close_ema9"] = (c - df[f"{prefix}ema_9"]) / c
    df[f"{prefix}close_sma50"] = (c - df[f"{prefix}sma_50"]) / c

    try:
        adx = ADXIndicator(h, l, c, 14)
        df[f"{prefix}adx"] = adx.adx()
        df[f"{prefix}di_pos"] = adx.adx_pos()
        df[f"{prefix}di_neg"] = adx.adx_neg()
    except Exception:
        pass

    stoch = StochasticOscillator(h, l, c, 14, 3)
    df[f"{prefix}stoch_k"] = stoch.stoch()
    df[f"{prefix}stoch_d"] = stoch.stoch_signal()

    df[f"{prefix}body"] = (c - o) / c
    df[f"{prefix}range"] = (h - l) / c
    df[f"{prefix}upper_wick"] = (h - np.maximum(c, o)) / c
    df[f"{prefix}lower_wick"] = (np.minimum(c, o) - l) / c

    df[f"{prefix}vol_10"] = df[f"{prefix}ret_1"].rolling(10).std()
    df[f"{prefix}vol_20"] = df[f"{prefix}ret_1"].rolling(20).std()
    return df


def load_ohlc_folder(folder: str) -> pd.DataFrame:
    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "timestamp" not in df.columns or len(df) == 0:
                continue
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df["timestamp"] = df["timestamp"].astype("int64")
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna()
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"Skip {f.name}: {e}")
    if not dfs:
        raise ValueError(f"No valid data in {folder}")
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    mask = (df["high"] - df["low"]) > 0.05
    print(f"  {folder}: {mask.sum():,} bars after flat-filter (from {len(df):,})")
    return df[mask].reset_index(drop=True)


def build_multi_tf_frame(m15: pd.DataFrame, h1: pd.DataFrame, m5: pd.DataFrame) -> pd.DataFrame:
    print("Computing features...")
    m15f = add_features(m15, "")
    h1f = add_features(h1, "h1_")
    m5f = add_features(m5, "m5_")

    base = m15f.copy()
    h1_cols = [c for c in h1f.columns if c.startswith("h1_")]
    base = pd.merge_asof(
        base.sort_values("timestamp"),
        h1f[["timestamp"] + h1_cols].sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    m5_keep = [
        c
        for c in m5f.columns
        if c.startswith("m5_")
        and any(
            x in c
            for x in [
                "rsi_14",
                "rsi_7",
                "macd",
                "bb_pct",
                "atr_pct",
                "ema9_21",
                "ret_1",
                "ret_3",
                "ret_5",
                "vol_10",
                "adx",
                "stoch",
                "body",
                "range",
            ]
        )
    ]
    base = pd.merge_asof(
        base.sort_values("timestamp"),
        m5f[["timestamp"] + m5_keep].sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    return base


def get_feature_columns(df: pd.DataFrame) -> list:
    cols = [
        c
        for c in df.columns
        if c not in ["timestamp", "datetime", "open", "high", "low", "close", "target"]
        and df[c].dtype in [np.float64, np.float32, np.int64, float, int]
    ]
    cols = [
        c
        for c in cols
        if not any(x in c for x in ["ema_9", "ema_21", "sma_50", "atr"])
        or "pct" in c
        or "ema9_21" in c
        or "close_" in c
    ]
    return cols


def main():
    parser = argparse.ArgumentParser(description="Train Meffs XAUUSD multi-TF LightGBM model")
    parser.add_argument("--m15_dir", type=str, required=True)
    parser.add_argument("--h1_dir", type=str, required=True)
    parser.add_argument("--m5_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="model")
    parser.add_argument("--train_end", type=str, default="2025-01-01", help="Train data < this date")
    parser.add_argument("--val_end", type=str, default="2025-07-01", help="Val data < this date")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    m15 = load_ohlc_folder(args.m15_dir)
    h1 = load_ohlc_folder(args.h1_dir)
    m5 = load_ohlc_folder(args.m5_dir)

    frame = build_multi_tf_frame(m15, h1, m5)
    frame["target"] = (frame["close"].shift(-1) > frame["close"]).astype(float)

    feat_cols = get_feature_columns(frame)
    print(f"Features: {len(feat_cols)}")

    df = frame.dropna(subset=feat_cols + ["target"]).copy()
    df["target"] = df["target"].astype(int)
    print(f"Samples: {len(df):,} | pos rate: {df['target'].mean():.3f}")

    train = df[df.datetime < args.train_end]
    val = df[(df.datetime >= args.train_end) & (df.datetime < args.val_end)]
    test = df[df.datetime >= args.val_end]
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    X_tr, y_tr = train[feat_cols], train["target"]
    X_va, y_va = val[feat_cols], val["target"]
    X_te, y_te = test[feat_cols], test["target"]

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 47,
        "learning_rate": 0.025,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 80,
        "reg_alpha": 0.2,
        "reg_lambda": 1.5,
        "verbose": -1,
        "n_jobs": -1,
        "seed": args.seed,
    }

    dtrain = lgb.Dataset(X_tr, y_tr)
    dval = lgb.Dataset(X_va, y_va, reference=dtrain)

    print("Training LightGBM...")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2500,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(120), lgb.log_evaluation(250)],
    )

    def evaluate(name, X, y):
        proba = model.predict(X)
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y, pred)
        auc = roc_auc_score(y, proba)
        print(f"\n=== {name} === Acc={acc:.4f} AUC={auc:.4f}")
        print(classification_report(y, pred, digits=4))
        return acc, auc

    evaluate("TRAIN", X_tr, y_tr)
    evaluate("VAL", X_va, y_va)
    evaluate("TEST (OOS)", X_te, y_te)

    txt_path = out_dir / "lgb_xauusd_multi_tf.txt"
    model.save_model(str(txt_path))
    print(f"\nSaved {txt_path}")

    feat_path = out_dir / "feat_cols.json"
    feat_path.write_text(json.dumps(feat_cols))
    print(f"Saved {feat_path}")

    if joblib is not None:
        pkl_path = out_dir / "lgb_xauusd_multi_tf.pkl"
        payload = {
            "model": model,
            "feat_cols": feat_cols,
            "meta": {
                "symbol": "XAUUSD",
                "base_tf": "M15",
                "features": "M5+M15+H1",
                "target": "next_bar_direction",
                "train_end": args.train_end,
                "val_end": args.val_end,
                "recommended_long": 0.58,
                "recommended_short": 0.42,
                "lot_default": 0.05,
            },
        }
        joblib.dump(payload, pkl_path, compress=3)
        print(f"Saved {pkl_path}")
    else:
        print("joblib not installed – skipped .pkl (pip install joblib)")

    imp = pd.Series(model.feature_importance(importance_type="gain"), index=feat_cols)
    print("\nTop 15 features:")
    print(imp.sort_values(ascending=False).head(15).to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
