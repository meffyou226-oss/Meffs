#!/usr/bin/env python3
"""Train a LightGBM filter on zone setups. Strict time split + label purge.

The model only sees features known at the entry bar. Labels use future price
but any train row whose exit_time is after the train cutoff is dropped (purge),
so unresolved outcomes cannot leak into training.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

import lightgbm as lgb

from zone_data import load_tf
from zone_engine import ZoneConfig, run_zones, trades_to_frame

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "ml_strategy" / "model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "is_long",
    "zone_height_atr",
    "zone_impulse_atr",
    "zone_age",
    "first_touch",
    "atr",
    "rsi14",
    "ema_spread_atr",
    "dist_entry_atr",
    "close_in_zone",
    "hour",
    "dow",
    "body_atr",
    "wick_lower_atr",
    "wick_upper_atr",
    "n_active_demand",
    "n_active_supply",
    "trend_with_zone",
]

PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 24,
    "max_depth": 5,
    "min_child_samples": 40,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.2,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def _metrics(y, p, name: str) -> dict:
    pred = (p >= 0.5).astype(int)
    out = {
        f"{name}_n": int(len(y)),
        f"{name}_acc": float(accuracy_score(y, pred)) if len(y) else None,
        f"{name}_pos_rate": float(np.mean(y)) if len(y) else None,
    }
    if len(y) and len(np.unique(y)) > 1:
        out[f"{name}_auc"] = float(roc_auc_score(y, p))
    else:
        out[f"{name}_auc"] = None
    return out


def threshold_scan(y: np.ndarray, p: np.ndarray, r: np.ndarray) -> dict:
    best = {"threshold": 0.5, "n": 0, "winrate": None, "expectancy": None}
    for thr in np.arange(0.45, 0.76, 0.01):
        mask = p >= thr
        if mask.sum() < 20:
            continue
        wr = float(y[mask].mean())
        exp = float(r[mask].mean())
        if best["expectancy"] is None or exp > best["expectancy"]:
            best = {
                "threshold": float(thr),
                "n": int(mask.sum()),
                "winrate": wr,
                "expectancy": exp,
            }
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="m15", choices=["m5", "m15", "h1"])
    ap.add_argument("--train-end", default="2025-06-30")
    ap.add_argument("--val-end", default="2025-12-31")
    ap.add_argument("--swing", type=int, default=5)
    ap.add_argument("--impulse", type=float, default=0.6)
    args = ap.parse_args()

    df = load_tf(args.tf)
    cfg = ZoneConfig(swing_len=args.swing, impulse_atr=args.impulse)
    trades, rule_stats = run_zones(df, cfg)
    tdf = trades_to_frame(trades, df)
    tdf = tdf[tdf["status"].isin([1, 2])].copy()
    tdf = tdf.dropna(subset=FEATURE_COLS + ["label_win", "entry_time", "exit_time"])
    tdf = tdf.sort_values("entry_time").reset_index(drop=True)

    train_end = pd.Timestamp(args.train_end, tz="UTC")
    val_end = pd.Timestamp(args.val_end, tz="UTC")

    # Purge: drop train samples whose outcome realizes after the cutoff.
    train = tdf[(tdf["entry_time"] < train_end) & (tdf["exit_time"] < train_end)]
    val = tdf[(tdf["entry_time"] >= train_end) & (tdf["entry_time"] < val_end)]
    test = tdf[tdf["entry_time"] >= val_end]

    print(
        f"closed trades={len(tdf)} train={len(train)} (purged) val={len(val)} test={len(test)}",
        flush=True,
    )

    X_train, y_train = train[FEATURE_COLS], train["label_win"].astype(int)
    X_val, y_val = val[FEATURE_COLS], val["label_win"].astype(int)
    X_test, y_test = test[FEATURE_COLS], test["label_win"].astype(int)

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS, reference=dtrain)
    model = lgb.train(
        PARAMS,
        dtrain,
        num_boost_round=1500,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(100)],
    )

    p_train = model.predict(X_train, num_iteration=model.best_iteration)
    p_val = model.predict(X_val, num_iteration=model.best_iteration)
    p_test = model.predict(X_test, num_iteration=model.best_iteration)

    result = {
        "tf": args.tf,
        "train_end": args.train_end,
        "val_end": args.val_end,
        "best_iteration": int(model.best_iteration or 0),
        "rule_stats_all": rule_stats,
        "rule_winrate_train": float(y_train.mean()) if len(y_train) else None,
        "rule_winrate_val": float(y_val.mean()) if len(y_val) else None,
        "rule_winrate_test": float(y_test.mean()) if len(y_test) else None,
    }
    result.update(_metrics(y_train.to_numpy(), p_train, "train"))
    result.update(_metrics(y_val.to_numpy(), p_val, "val"))
    result.update(_metrics(y_test.to_numpy(), p_test, "test"))

    result["val_threshold"] = threshold_scan(
        y_val.to_numpy(), p_val, val["r_multiple"].to_numpy()
    )
    result["test_at_val_threshold"] = None
    thr = result["val_threshold"]["threshold"]
    if len(y_test):
        mask = p_test >= thr
        if mask.sum():
            result["test_at_val_threshold"] = {
                "threshold": thr,
                "n": int(mask.sum()),
                "n_all": int(len(y_test)),
                "winrate": float(y_test.to_numpy()[mask].mean()),
                "expectancy_R": float(test["r_multiple"].to_numpy()[mask].mean()),
                "baseline_winrate": float(y_test.mean()),
                "baseline_expectancy_R": float(test["r_multiple"].mean()),
            }

    imp = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance": model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)
    result["feature_importance"] = imp.to_dict(orient="records")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = OUT_DIR / f"lgb_zone_filter_{args.tf}_{stamp}.txt"
    meta_path = OUT_DIR / f"lgb_zone_filter_{args.tf}_{stamp}_meta.json"
    model.save_model(str(model_path))
    meta_path.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    print(f"saved {model_path}")


if __name__ == "__main__":
    main()
