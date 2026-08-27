#!/usr/bin/env python3
"""Fast LightGBM price-action training across all timeframes."""
from __future__ import annotations

import os
import sys
import gc
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = REPO_ROOT / "ml_strategy" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAMES = ["m1", "m5", "m15", "h1"]
BASE_TFS = ["h1", "m15", "m5", "m1"]
PREDICT_HORIZON = 1
TRAIN_TEST_SPLIT = 0.75
WALK_FORWARD_SPLITS = 5
EARLY_STOPPING_ROUNDS = 150
NUM_BOOST_ROUND = 2000
MAX_SAMPLE_ROWS = 1_500_000
PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 48,
    "max_depth": 7,
    "min_child_samples": 80,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.05,
    "reg_lambda": 0.05,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def _read_csv_sorted(path: Path) -> pd.DataFrame:
    skip_keywords = ["<<<<<<<", ">>>>>>>", "=======", "undefined"]
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [
            line for line in f
            if line.strip() and not any(k in line for k in skip_keywords)
        ]
    from io import StringIO
    buf = StringIO("".join(lines))
    df = pd.read_csv(buf)
    df.columns = [c.strip().lower() for c in df.columns]
    needed = {"timestamp", "open", "high", "low", "close"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing columns in {path}: {needed - set(df.columns)}")
    df = df[["timestamp", "open", "high", "low", "close"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float).astype(int), unit="ms", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return df


def load_timeframe(tf: str, max_rows: int = MAX_SAMPLE_ROWS) -> pd.DataFrame:
    tf_dir = DATA_DIR / f"xauusd_{tf}"
    if not tf_dir.exists():
        raise FileNotFoundError(tf_dir)
    files = sorted(tf_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {tf_dir}")
    frames = []
    for f in files:
        try:
            frames.append(_read_csv_sorted(f))
        except Exception as e:
            print(f"[WARN] skip {f.name}: {e}", flush=True)
    if not frames:
        raise RuntimeError(f"No valid data for {tf}")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    if len(df) > max_rows:
        cutoff = df["timestamp"].iloc[-max_rows]
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    print(f"[{tf}] loaded {len(df):,} rows {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}", flush=True)
    return df


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ma_up = up.ewm(com=window - 1, adjust=False).mean()
    ma_down = down.ewm(com=window - 1, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def stochastic(df: pd.DataFrame, k_window: int = 14, d_window: int = 3):
    low_min = df["low"].rolling(k_window, min_periods=k_window).min()
    high_max = df["high"].rolling(k_window, min_periods=k_window).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_window, min_periods=d_window).mean()
    return k, d


def adx(df: pd.DataFrame, window: int = 14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_ = tr.rolling(window, min_periods=window).mean()
    plus_di = 100 * (plus_dm.rolling(window, min_periods=window).mean() / atr_.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(window, min_periods=window).mean() / atr_.replace(0, np.nan))
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx_line = dx.rolling(window, min_periods=window).mean()
    return adx_line, plus_di, minus_di


def make_features(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    rng = (h - l).replace(0, np.nan)

    feats = {
        f"{tf}_body_pct": ((c - o).abs() / rng).fillna(0),
        f"{tf}_upper_shadow_pct": ((h - np.maximum(o, c)) / rng).fillna(0),
        f"{tf}_lower_shadow_pct": ((np.minimum(o, c) - l) / rng).fillna(0),
        f"{tf}_close_pos": ((c - l) / rng).fillna(0.5),
        f"{tf}_is_green": (c > o).astype(int),
    }

    for w in [1, 2, 3, 5, 10, 20]:
        feats[f"{tf}_ret_{w}"] = c.pct_change(w)

    for w in [5, 10, 20, 50]:
        feats[f"{tf}_sma_{w}"] = (c / sma(c, w)) - 1
        feats[f"{tf}_ema_{w}"] = (c / ema(c, w)) - 1

    feats[f"{tf}_ema_fast_slow"] = ema(c, 10) - ema(c, 20)
    feats[f"{tf}_sma_fast_slow"] = sma(c, 10) - sma(c, 20)

    vol = atr(df, 14)
    feats[f"{tf}_atr_14"] = vol
    feats[f"{tf}_atr_pct"] = (vol / c).replace([np.inf, -np.inf], 0).fillna(0)
    for w in [10, 20]:
        std = c.diff().rolling(w, min_periods=w).std()
        feats[f"{tf}_vol_{w}"] = (std / c).replace([np.inf, -np.inf], 0).fillna(0)

    sma20 = sma(c, 20)
    bb_std = c.diff().rolling(20, min_periods=20).std()
    bb_upper = sma20 + 2 * bb_std
    bb_lower = sma20 - 2 * bb_std
    bb_width = bb_upper - bb_lower
    bb_pos = ((c - bb_lower) / bb_width).fillna(0.5)
    feats[f"{tf}_bb_width"] = bb_width
    feats[f"{tf}_bb_pos"] = bb_pos
    feats[f"{tf}_bb_squeeze"] = (bb_width / c).replace([np.inf, -np.inf], 0).fillna(0)

    feats[f"{tf}_rsi_14"] = rsi(c, 14)
    feats[f"{tf}_rsi_7"] = rsi(c, 7)

    macd_line, signal_line, hist = macd(c)
    feats[f"{tf}_macd"] = macd_line
    feats[f"{tf}_macd_signal"] = signal_line
    feats[f"{tf}_macd_hist"] = hist

    k, d = stochastic(df)
    feats[f"{tf}_stoch_k"] = k
    feats[f"{tf}_stoch_d"] = d
    feats[f"{tf}_stoch_kd"] = (k - d)

    adx_line, plus_di, minus_di = adx(df)
    feats[f"{tf}_adx"] = adx_line
    feats[f"{tf}_plus_di"] = plus_di
    feats[f"{tf}_minus_di"] = minus_di
    feats[f"{tf}_di_diff"] = (plus_di - minus_di)

    for w in [10, 20, 50]:
        feats[f"{tf}_dist_high_{w}"] = ((c - h.rolling(w, min_periods=w).max()) / c).replace([np.inf, -np.inf], 0).fillna(0)
        feats[f"{tf}_dist_low_{w}"] = ((c - l.rolling(w, min_periods=w).min()) / c).replace([np.inf, -np.inf], 0).fillna(0)

    up = (c > o).astype(int)
    down = (c < o).astype(int)
    up_streak = up.copy()
    down_streak = down.copy()
    for i in range(1, 6):
        up_streak = up_streak + up.shift(i).fillna(0)
        down_streak = down_streak + down.shift(i).fillna(0)
    feats[f"{tf}_up_streak"] = up_streak
    feats[f"{tf}_down_streak"] = down_streak

    feats[f"{tf}_gap"] = ((o - c.shift(1)) / c.shift(1)).replace([np.inf, -np.inf], 0).fillna(0)

    out = pd.DataFrame(feats, index=df.index)
    out["timestamp"] = pd.to_datetime(df["timestamp"].values, utc=True)
    return out


def resample_to_m1(df_tf: pd.DataFrame, tf: str) -> pd.DataFrame:
    tf_min = {"m5": 5, "m15": 15, "h1": 60}.get(tf)
    if tf_min is None:
        return df_tf
    df_tf = df_tf.set_index("timestamp")
    if df_tf.index.tz is None:
        df_tf.index = pd.to_datetime(df_tf.index).tz_localize("UTC")
    else:
        df_tf.index = pd.to_datetime(df_tf.index).tz_convert("UTC")
    rule = f"{tf_min}min"
    o = df_tf["open"].resample(rule, label="right", closed="right").first()
    h = df_tf["high"].resample(rule, label="right", closed="right").max()
    l = df_tf["low"].resample(rule, label="right", closed="right").min()
    c = df_tf["close"].resample(rule, label="right", closed="right").last()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=o.index)
    out.index.name = "timestamp"
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.reset_index()
    out["timestamp"] = pd.to_datetime(out["timestamp"].values, utc=True)
    return out


def make_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    future_ret = df["close"].shift(-horizon) / df["close"] - 1
    return (future_ret > 0).astype(int)


def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.replace([np.inf, -np.inf], np.nan)
    num = num.ffill().bfill()
    return num


def prepare_X_y(df: pd.DataFrame):
    df = sanitize(df)
    drop_cols = {"timestamp", "close", "target"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values
    y = df["target"].values
    return X, y, feature_cols


def train_eval_holdout(X, y, feature_cols):
    split = int(len(X) * TRAIN_TEST_SPLIT)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    X_test_s = sc.transform(X_test)

    dtrain = lgb.Dataset(X_train_s, label=y_train, feature_name=feature_cols)
    dtest = lgb.Dataset(X_test_s, label=y_test, reference=dtrain, feature_name=feature_cols)
    model = lgb.train(
        PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtest],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=500),
        ],
    )
    proba = model.predict(X_test_s, num_iteration=model.best_iteration)
    best_acc, best_thr = 0.0, 0.5
    for thr in np.linspace(0.4, 0.6, 41):
        acc = accuracy_score(y_test, (proba >= thr).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    auc = roc_auc_score(y_test, proba)
    return {
        "accuracy_default": accuracy_score(y_test, (proba >= 0.5).astype(int)),
        "accuracy_optimal": best_acc,
        "optimal_threshold": best_thr,
        "auc": auc,
        "best_iteration": model.best_iteration,
        "features": len(feature_cols),
    }, model, sc, feature_cols


def walk_forward(X, y, feature_cols):
    tscv = TimeSeriesSplit(n_splits=WALK_FORWARD_SPLITS)
    scores = []
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        dtrain = lgb.Dataset(X_tr_s, label=y_tr, feature_name=feature_cols)
        dtest = lgb.Dataset(X_te_s, label=y_te, reference=dtrain, feature_name=feature_cols)
        model = lgb.train(
            PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dtest],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=500),
            ],
        )
        proba = model.predict(X_te_s, num_iteration=model.best_iteration)
        pred = (proba >= 0.5).astype(int)
        scores.append(accuracy_score(y_te, pred))
        print(f"  fold {fold}/{WALK_FORWARD_SPLITS} acc={scores[-1]:.4f}", flush=True)
    return float(np.mean(scores)), float(np.std(scores)), scores


def build_dataset(base_tf: str) -> pd.DataFrame:
    t0 = time.time()
    print(f"\n=== Building dataset base={base_tf} ===", flush=True)
    data: Dict[str, pd.DataFrame] = {}
    for tf in TIMEFRAMES:
        data[tf] = load_timeframe(tf)

    base = data[base_tf].copy()
    base_m1 = resample_to_m1(base, base_tf) if base_tf != "m1" else base.copy()

    feat_frames = {}
    for tf in TIMEFRAMES:
        if tf == base_tf:
            native = base
        elif tf == "m1":
            native = data["m1"]
        else:
            higher = data[tf].set_index("timestamp")
            m1_idx = base_m1.set_index("timestamp").index
            higher_m1 = higher.reindex(m1_idx, method="ffill")
            higher_m1 = higher_m1.dropna(subset=["open", "high", "low", "close"]).reset_index()
            higher_m1["timestamp"] = pd.to_datetime(higher_m1["timestamp"].values, utc=True)
            native = higher_m1
        feat_frames[tf] = make_features(native, tf)

    merged = base_m1[["timestamp", "close"]].copy()
    merged["timestamp"] = pd.to_datetime(merged["timestamp"].values, utc=True)
    for tf in TIMEFRAMES:
        tf_feats = feat_frames[tf]
        merged = merged.merge(tf_feats, on="timestamp", how="left")

    merged = merged.dropna(subset=[c for c in merged.columns if c not in ("timestamp", "close")])
    merged = merged.reset_index(drop=True)
    target = make_target(merged, horizon=PREDICT_HORIZON)
    merged["target"] = target.values
    merged = merged.dropna(subset=["target"]).reset_index(drop=True)
    print(f"  shape={merged.shape} bal={merged['target'].mean():.2%} t={time.time()-t0:.1f}s", flush=True)
    return merged


def main():
    t0 = time.time()
    results: List[Dict[str, Any]] = []
    best_model = None
    best_acc = -1.0
    best_meta = None
    best_scaler = None
    best_cols = None
    best_df = None
    best_base_tf = None

    for base_tf in BASE_TFS:
        gc.collect()
        try:
            df = build_dataset(base_tf)
        except Exception as e:
            print(f"[ERROR] dataset build failed for {base_tf}: {e}", flush=True)
            continue

        X, y, cols = prepare_X_y(df)
        print(f"[{base_tf}] training on {X.shape[0]:,} samples, {X.shape[1]} features", flush=True)
        t1 = time.time()

        info, model, scaler, _ = train_eval_holdout(X, y, cols)
        wf_mean, wf_std, wf_scores = walk_forward(X, y, cols)

        info["base_tf"] = base_tf
        info["walk_forward_mean"] = wf_mean
        info["walk_forward_std"] = wf_std
        info["walk_forward_scores"] = wf_scores
        info["train_samples"] = int(len(X) * TRAIN_TEST_SPLIT)
        info["test_samples"] = len(X) - info["train_samples"]
        results.append(info)

        print(
            f"[{base_tf}] default={info['accuracy_default']:.4f} "
            f"optimal={info['accuracy_optimal']:.4f} thr={info['optimal_threshold']:.2f} "
            f"auc={info['auc']:.4f} wf={wf_mean:.4f}+/-{wf_std:.4f} "
            f"t={time.time()-t1:.1f}s",
            flush=True,
        )

        if info["accuracy_optimal"] > best_acc:
            best_acc = info["accuracy_optimal"]
            best_model = model
            best_meta = info
            best_scaler = scaler
            best_cols = cols
            best_df = df
            best_base_tf = base_tf

        del df, X, y
        gc.collect()

    print("\n=== SUMMARY ===", flush=True)
    for r in results:
        print(
            f"{r['base_tf']:>3} | default={r['accuracy_default']:.4f} "
            f"optimal={r['accuracy_optimal']:.4f} thr={r['optimal_threshold']:.2f} "
            f"auc={r['auc']:.4f} wf={r['walk_forward_mean']:.4f}+/-{r['walk_forward_std']:.4f}",
            flush=True,
        )

    if best_model is not None and best_df is not None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_path = MODEL_DIR / f"lgb_price_action_{best_base_tf}_{ts}.txt"
        best_model.save_model(str(model_path))
        meta = {
            "base_tf": best_base_tf,
            "accuracy_default": best_meta["accuracy_default"],
            "accuracy_optimal": best_meta["accuracy_optimal"],
            "optimal_threshold": best_meta["optimal_threshold"],
            "auc": best_meta["auc"],
            "walk_forward_mean": best_meta["walk_forward_mean"],
            "walk_forward_std": best_meta["walk_forward_std"],
            "best_iteration": best_meta["best_iteration"],
            "features": best_meta["features"],
            "feature_cols": best_cols,
            "train_samples": best_meta["train_samples"],
            "test_samples": best_meta["test_samples"],
            "trained_at": ts,
        }
        meta_path = MODEL_DIR / f"lgb_price_action_{best_base_tf}_{ts}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"\nBest model saved: {model_path}", flush=True)
        print(f"Meta saved: {meta_path}", flush=True)

        # Retrain combined multi-TF on best base using full data
        print("\n=== Combined Multi-TF ===", flush=True)
        X, y, cols = prepare_X_y(best_df)
        split = int(len(X) * TRAIN_TEST_SPLIT)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        dtrain = lgb.Dataset(X_tr_s, label=y_tr, feature_name=cols)
        dtest = lgb.Dataset(X_te_s, label=y_te, reference=dtrain, feature_name=cols)
        combined = lgb.train(
            PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dtest],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=500),
            ],
        )
        proba = combined.predict(X_te_s, num_iteration=combined.best_iteration)
        acc = accuracy_score(y_te, (proba >= 0.5).astype(int))
        auc = roc_auc_score(y_te, proba)
        print(f"Combined acc={acc:.4f} auc={auc:.4f} best_iter={combined.best_iteration}", flush=True)
        cts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        cpath = MODEL_DIR / f"lgb_price_action_combined_{cts}.txt"
        combined.save_model(str(cpath))
        print(f"Combined model saved: {cpath}", flush=True)

    print(f"Total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
