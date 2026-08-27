#!/usr/bin/env python3
"""Strict OOS evaluation: train on first 75%, val from train, test on true last 25%."""
from __future__ import annotations

import json
import gc
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

import lightgbm as lgb

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = REPO_ROOT / "ml_strategy" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAMES = ["m1", "m5", "m15", "h1"]
BASE_TFS = ["m15", "m5"]
TRAIN_TEST_SPLIT = 0.75
VAL_SPLIT = 0.85  # validation split within train
EARLY_STOPPING_ROUNDS = 150
NUM_BOOST_ROUND = 2000
MAX_ROWS = 1_500_000

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
        lines = [line for line in f if line.strip() and not any(k in line for k in skip_keywords)]
    from io import StringIO
    buf = StringIO("".join(lines))
    df = pd.read_csv(buf)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[["timestamp", "open", "high", "low", "close"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float).astype(int), unit="ms", utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return df


def load_timeframe(tf: str) -> pd.DataFrame:
    files = sorted((DATA_DIR / f"xauusd_{tf}").glob("*.csv"))
    frames = []
    for f in files:
        try:
            frames.append(_read_csv_sorted(f))
        except Exception as e:
            print(f"[WARN] skip {f.name}: {e}", flush=True)
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    if len(df) > MAX_ROWS:
        cutoff = df["timestamp"].iloc[-MAX_ROWS]
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    print(f"[{tf}] loaded {len(df):,} rows", flush=True)
    return df


def ema(s, span): return s.ewm(span=span, adjust=False).mean()
def sma(s, window): return s.rolling(window, min_periods=window).mean()


def atr(df, window=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def rsi(close, window=14):
    delta = close.diff()
    up, down = delta.clip(lower=0), (-delta).clip(lower=0)
    ma_up = up.ewm(com=window - 1, adjust=False).mean()
    ma_down = down.ewm(com=window - 1, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close, fast=12, slow=26, signal=9):
    ema_fast, ema_slow = ema(close, fast), ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def stochastic(df, k_window=14, d_window=3):
    low_min = df["low"].rolling(k_window, min_periods=k_window).min()
    high_max = df["high"].rolling(k_window, min_periods=k_window).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    return k, k.rolling(d_window, min_periods=d_window).mean()


def adx(df, window=14):
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff().where((high.diff() > -low.diff()) & (high.diff() > 0), 0)
    minus_dm = (-low.diff()).where((-low.diff() > high.diff()) & (-low.diff() > 0), 0)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_ = tr.rolling(window, min_periods=window).mean()
    plus_di = 100 * plus_dm.rolling(window, min_periods=window).mean() / atr_.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(window, min_periods=window).mean() / atr_.replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.rolling(window, min_periods=window).mean(), plus_di, minus_di


def make_features(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    df = df.copy()
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
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
        feats[f"{tf}_vol_{w}"] = (c.diff().rolling(w, min_periods=w).std() / c).replace([np.inf, -np.inf], 0).fillna(0)
    sma20 = sma(c, 20)
    bb_std = c.diff().rolling(20, min_periods=20).std()
    bb_width = sma20 + 2 * bb_std - (sma20 - 2 * bb_std)
    bb_pos = ((c - (sma20 - 2 * bb_std)) / bb_width).fillna(0.5)
    feats[f"{tf}_bb_width"], feats[f"{tf}_bb_pos"] = bb_width, bb_pos
    feats[f"{tf}_bb_squeeze"] = (bb_width / c).replace([np.inf, -np.inf], 0).fillna(0)
    feats[f"{tf}_rsi_14"], feats[f"{tf}_rsi_7"] = rsi(c, 14), rsi(c, 7)
    macd_line, signal_line, hist = macd(c)
    feats[f"{tf}_macd"], feats[f"{tf}_macd_signal"], feats[f"{tf}_macd_hist"] = macd_line, signal_line, hist
    k, d = stochastic(df)
    feats[f"{tf}_stoch_k"], feats[f"{tf}_stoch_d"], feats[f"{tf}_stoch_kd"] = k, d, k - d
    adx_line, plus_di, minus_di = adx(df)
    feats[f"{tf}_adx"], feats[f"{tf}_plus_di"], feats[f"{tf}_minus_di"], feats[f"{tf}_di_diff"] = adx_line, plus_di, minus_di, plus_di - minus_di
    for w in [10, 20, 50]:
        feats[f"{tf}_dist_high_{w}"] = ((c - h.rolling(w, min_periods=w).max()) / c).replace([np.inf, -np.inf], 0).fillna(0)
        feats[f"{tf}_dist_low_{w}"] = ((c - l.rolling(w, min_periods=w).min()) / c).replace([np.inf, -np.inf], 0).fillna(0)
    up, down = (c > o).astype(int), (c < o).astype(int)
    up_streak, down_streak = up.copy(), down.copy()
    for i in range(1, 6):
        up_streak = up_streak + up.shift(i).fillna(0)
        down_streak = down_streak + down.shift(i).fillna(0)
    feats[f"{tf}_up_streak"], feats[f"{tf}_down_streak"] = up_streak, down_streak
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
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index()
    out["timestamp"] = pd.to_datetime(out["timestamp"].values, utc=True)
    return out


def make_target(df, horizon=1):
    return (df["close"].shift(-horizon) / df["close"] - 1 > 0).astype(int)


def sanitize(df):
    num = df.select_dtypes(include=[np.number]).copy()
    return num.replace([np.inf, -np.inf], np.nan).ffill().bfill()


def prepare_X_y(df):
    df = sanitize(df)
    drop = {"timestamp", "close", "target"}
    cols = [c for c in df.columns if c not in drop]
    return df[cols].values, df["target"].values, cols


def build_dataset(base_tf: str) -> pd.DataFrame:
    print(f"\n=== Building dataset base={base_tf} ===", flush=True)
    data = {tf: load_timeframe(tf) for tf in TIMEFRAMES}
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
            higher_m1 = higher.reindex(m1_idx, method="ffill").dropna(subset=["open", "high", "low", "close"]).reset_index()
            higher_m1["timestamp"] = pd.to_datetime(higher_m1["timestamp"].values, utc=True)
            native = higher_m1
        feat_frames[tf] = make_features(native, tf)
    merged = base_m1[["timestamp", "close"]].copy()
    merged["timestamp"] = pd.to_datetime(merged["timestamp"].values, utc=True)
    for tf in TIMEFRAMES:
        merged = merged.merge(feat_frames[tf], on="timestamp", how="left")
    merged = merged.dropna(subset=[c for c in merged.columns if c not in ("timestamp", "close")]).reset_index(drop=True)
    merged["target"] = make_target(merged).values
    merged = merged.dropna(subset=["target"]).reset_index(drop=True)
    print(f"  shape={merged.shape} bal={merged['target'].mean():.2%}", flush=True)
    return merged


def strict_oos(df: pd.DataFrame, base_tf: str):
    X, y, cols = prepare_X_y(df)
    split = int(len(X) * TRAIN_TEST_SPLIT)
    val_split = int(split * VAL_SPLIT)

    X_train, X_val, X_test = X[:val_split], X[val_split:split], X[split:]
    y_train, y_val, y_test = y[:val_split], y[val_split:split], y[split:]

    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    X_val_s = sc.transform(X_val)
    X_test_s = sc.transform(X_test)

    dtrain = lgb.Dataset(X_train_s, label=y_train, feature_name=cols)
    dval = lgb.Dataset(X_val_s, label=y_val, reference=dtrain, feature_name=cols)
    dtest = lgb.Dataset(X_test_s, label=y_test, reference=dtrain, feature_name=cols)

    model = lgb.train(
        PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=500),
        ],
    )

    proba_test = model.predict(X_test_s, num_iteration=model.best_iteration)
    best_acc, best_thr = 0.0, 0.5
    for thr in np.linspace(0.4, 0.6, 41):
        acc = accuracy_score(y_test, (proba_test >= thr).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    acc_default = accuracy_score(y_test, (proba_test >= 0.5).astype(int))
    auc = roc_auc_score(y_test, proba_test)

    # Also evaluate on val set to check overfitting
    proba_val = model.predict(X_val_s, num_iteration=model.best_iteration)
    acc_val_default = accuracy_score(y_val, (proba_val >= 0.5).astype(int))
    auc_val = roc_auc_score(y_val, proba_val)

    result = {
        "base_tf": base_tf,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "val_accuracy_default": acc_val_default,
        "val_auc": auc_val,
        "test_accuracy_default": acc_default,
        "test_accuracy_optimal": best_acc,
        "test_optimal_threshold": best_thr,
        "test_auc": auc,
        "best_iteration": model.best_iteration,
        "features": len(cols),
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"lgb_price_action_oos_{base_tf}_{ts}.txt"
    model.save_model(str(model_path))
    meta_path = MODEL_DIR / f"lgb_price_action_oos_{base_tf}_{ts}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(result | {"feature_cols": cols, "trained_at": ts}, f, indent=2, default=str)
    print(f"[{base_tf}] OOS test acc={best_acc:.4f} auc={auc:.4f} best_iter={model.best_iteration}", flush=True)
    print(f"[{base_tf}] Model saved: {model_path.name}", flush=True)
    return result


def main():
    results = []
    for base_tf in BASE_TFS:
        gc.collect()
        try:
            df = build_dataset(base_tf)
        except Exception as e:
            print(f"[ERROR] {base_tf}: {e}", flush=True)
            continue
        results.append(strict_oos(df, base_tf))
        del df
        gc.collect()

    print("\n=== STRICT OOS RESULTS ===", flush=True)
    for r in results:
        print(
            f"{r['base_tf']:>3} | test={r['test_accuracy_optimal']:.4f} "
            f"(def={r['test_accuracy_default']:.4f}) thr={r['test_optimal_threshold']:.2f} "
            f"auc={r['test_auc']:.4f} val={r['val_accuracy_default']:.4f} "
            f"iter={r['best_iteration']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
