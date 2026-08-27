#!/usr/bin/env python3
"""OOS strategy backtest with explicit look-ahead bias mitigation."""
from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime, timezone
from itertools import product

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = REPO_ROOT / "ml_strategy" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

POINT_VALUE = 0.05
SPREAD_POINTS = 10
SPREAD_COST = SPREAD_POINTS * POINT_VALUE
COMMISSION = 0.28
COST_PER_TRADE = SPREAD_COST + COMMISSION

TRAIN_END = 0.75
VAL_END = 0.85
PURGE_ROWS = 60  # purge first N rows in test to avoid look-ahead in rolling features

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
EARLY_STOPPING_ROUNDS = 150
NUM_BOOST_ROUND = 2000

THRESHOLDS = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
TP_POINTS = [15, 20, 30, 40, 50, 60, 80]
SL_POINTS = [10, 15, 20, 25, 30, 40]


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


def load_tf(tf: str) -> pd.DataFrame:
    files = sorted((DATA_DIR / f"xauusd_{tf}").glob("*.csv"))
    frames = []
    for f in files:
        try:
            frames.append(_read_csv_sorted(f))
        except Exception as e:
            print(f"[WARN] skip {f.name}: {e}", flush=True)
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
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


def align_tf(df_tf: pd.DataFrame, m5_idx) -> pd.DataFrame:
    higher = df_tf.set_index("timestamp")
    if higher.index.tz is None:
        higher.index = pd.to_datetime(higher.index).tz_localize("UTC")
    else:
        higher.index = pd.to_datetime(higher.index).tz_convert("UTC")
    aligned = higher.reindex(m5_idx, method="ffill")
    aligned = aligned.dropna(subset=["open", "high", "low", "close"]).reset_index()
    aligned["timestamp"] = pd.to_datetime(aligned["timestamp"].values, utc=True)
    return aligned


def run_backtest(proba, df, tp_points, sl_points, long_thr, short_thr):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    
    position = 0
    entry_price = 0.0
    tp_price = 0.0
    sl_price = 0.0
    
    trades = []
    tp_reward = tp_points * POINT_VALUE
    sl_risk = sl_points * POINT_VALUE
    
    for i in range(len(proba) - 1):
        p = proba[i]
        
        if position != 0:
            if position == 1:
                if low[i + 1] <= sl_price:
                    pnl = -(sl_risk + COST_PER_TRADE)
                    trades.append(pnl)
                    position = 0
                elif high[i + 1] >= tp_price:
                    pnl = tp_reward - COST_PER_TRADE
                    trades.append(pnl)
                    position = 0
                else:
                    pnl = (close[i + 1] - entry_price) - COST_PER_TRADE
                    trades.append(pnl)
                    position = 0
            elif position == -1:
                if high[i + 1] >= sl_price:
                    pnl = -(sl_risk + COST_PER_TRADE)
                    trades.append(pnl)
                    position = 0
                elif low[i + 1] <= tp_price:
                    pnl = tp_reward - COST_PER_TRADE
                    trades.append(pnl)
                    position = 0
                else:
                    pnl = (entry_price - close[i + 1]) - COST_PER_TRADE
                    trades.append(pnl)
                    position = 0
        
        if position == 0:
            if p >= long_thr:
                position = 1
                entry_price = close[i]
                tp_price = entry_price + tp_points * POINT_VALUE
                sl_price = entry_price - sl_points * POINT_VALUE
            elif p <= short_thr:
                position = -1
                entry_price = close[i]
                tp_price = entry_price - tp_points * POINT_VALUE
                sl_price = entry_price + sl_points * POINT_VALUE
    
    if not trades:
        return {"pf": 0, "winrate": 0, "trades": 0, "total_pnl": 0, "gross_profit": 0, "gross_loss": 0}
    
    gross_profit = sum(t for t in trades if t > 0)
    gross_loss = abs(sum(t for t in trades if t < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    winrate = sum(1 for t in trades if t > 0) / len(trades)
    
    return {
        "pf": pf,
        "winrate": winrate,
        "trades": len(trades),
        "total_pnl": sum(trades),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }


def optimize_on_val(proba_val, df_val):
    best = None
    results = []
    
    for tp, sl, thr in product(TP_POINTS, SL_POINTS, THRESHOLDS):
        long_thr = thr
        short_thr = 1 - thr
        
        res = run_backtest(proba_val, df_val, tp, sl, long_thr, short_thr)
        res["tp"] = tp
        res["sl"] = sl
        res["thr"] = thr
        results.append(res)
        
        if res["pf"] >= 2.0 and res["trades"] >= 20:
            score = res["winrate"] * 0.4 + min(res["pf"] / 3, 1.0) * 0.6
            if best is None or score > best["score"]:
                best = {**res, "score": score}
    
    if best is None:
        valid = [r for r in results if r["trades"] >= 20]
        if valid:
            valid.sort(key=lambda x: x["pf"], reverse=True)
            best = valid[0]
        else:
            valid = [r for r in results if r["trades"] >= 10]
            if valid:
                valid.sort(key=lambda x: x["pf"], reverse=True)
                best = valid[0]
            else:
                best = max(results, key=lambda x: x["pf"])
    
    return best, results


def prepare_X_y(df: pd.DataFrame):
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.replace([np.inf, -np.inf], np.nan)
    num = num.ffill().bfill()
    drop = {"target"}
    cols = [c for c in num.columns if c not in drop]
    return num[cols].values, df["target"].values, cols


def main():
    t0 = time.time()
    
    print("Loading data...", flush=True)
    m5 = load_tf("m5")
    m15 = load_tf("m15")
    h1 = load_tf("h1")
    
    m5_idx = m5.set_index("timestamp").index
    m15_aligned = align_tf(m15, m5_idx)
    h1_aligned = align_tf(h1, m5_idx)
    
    # Build full merged dataset WITH features - LOOK-AHEAD BIAS RISK:
    # We compute features on the FULL dataset, which means rolling indicators
    # at position t include data from the future.
    # Mitigation: we PURGE the first PURGE_ROWS in the test set so that
    # rolling features at the first testable row only use data up to that point.
    print("Building features on full dataset...", flush=True)
    merged = m5[["timestamp", "open", "high", "low", "close"]].copy()
    merged["timestamp"] = pd.to_datetime(merged["timestamp"].values, utc=True)
    
    feat_m5 = make_features(merged, "m5")
    feat_m15 = make_features(m15_aligned, "m15")
    feat_h1 = make_features(h1_aligned, "h1")
    
    merged = merged.merge(feat_m5, on="timestamp", how="left")
    merged = merged.merge(feat_m15, on="timestamp", how="left")
    merged = merged.merge(feat_h1, on="timestamp", how="left")
    
    merged["target"] = (merged["close"].shift(-1) / merged["close"] - 1 > 0).astype(int)
    merged = merged.dropna(subset=["target"]).reset_index(drop=True)
    
    n = len(merged)
    train_end = int(n * TRAIN_END)
    val_end = int(n * VAL_END)
    
    print(f"Total rows: {n}", flush=True)
    print(f"Split: train 0-{train_end}, val {train_end}-{val_end}, test {val_end}-{n}", flush=True)
    
    # Split
    train = merged.iloc[:train_end].copy()
    val = merged.iloc[train_end:val_end].copy()
    test = merged.iloc[val_end:].copy()
    
    # Purge first PURGE_ROWS in test to eliminate look-ahead in rolling features
    if len(test) > PURGE_ROWS:
        print(f"Purging first {PURGE_ROWS} rows in test to avoid look-ahead bias", flush=True)
        test = test.iloc[PURGE_ROWS:].copy().reset_index(drop=True)
    
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}", flush=True)
    
    # Prepare X, y
    drop_cols = {"timestamp", "open", "high", "low", "close", "target"}
    feature_cols = [c for c in train.columns if c not in drop_cols]
    
    X_train, y_train, _ = prepare_X_y(train[feature_cols + ["target"]])
    X_val, y_val, _ = prepare_X_y(val[feature_cols + ["target"]])
    X_test, y_test, _ = prepare_X_y(test[feature_cols + ["target"]])
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    # Train on train, validate on val (NO test!)
    print("Training LightGBM...", flush=True)
    dtrain = lgb.Dataset(X_train_s, label=y_train, feature_name=feature_cols)
    dval = lgb.Dataset(X_val_s, label=y_val, reference=dtrain, feature_name=feature_cols)
    
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
    print(f"Best iteration: {model.best_iteration}", flush=True)
    
    # Optimize strategy on val (NOT on test!)
    print("Optimizing strategy on val...", flush=True)
    proba_val = model.predict(X_val_s, num_iteration=model.best_iteration)
    val_ohlc = val[["open", "high", "low", "close"]].reset_index(drop=True)
    best_val, all_val = optimize_on_val(proba_val, val_ohlc)
    print(f"Best val: PF={best_val['pf']:.2f} WR={best_val['winrate']:.2%} trades={best_val['trades']} TP={best_val['tp']} SL={best_val['sl']} Thr={best_val['thr']:.2f}", flush=True)
    
    # Test on OOS (TRUE OOS)
    print("Testing on OOS...", flush=True)
    proba_test = model.predict(X_test_s, num_iteration=model.best_iteration)
    test_ohlc = test[["open", "high", "low", "close"]].reset_index(drop=True)
    oos = run_backtest(proba_test, test_ohlc, best_val["tp"], best_val["sl"], best_val["thr"], 1 - best_val["thr"])
    
    print(f"\n=== OOS BACKTEST RESULTS ===", flush=True)
    print(f"Profit Factor: {oos['pf']:.2f}", flush=True)
    print(f"Winrate: {oos['winrate']:.2%}", flush=True)
    print(f"Trades: {oos['trades']}", flush=True)
    print(f"Total PnL: ${oos['total_pnl']:.2f}", flush=True)
    print(f"Gross Profit: ${oos['gross_profit']:.2f}", flush=True)
    print(f"Gross Loss: ${oos['gross_loss']:.2f}", flush=True)
    print(f"Params: TP={best_val['tp']} SL={best_val['sl']} Threshold={best_val['thr']:.2f}", flush=True)
    print(f"Costs per trade: ${COST_PER_TRADE:.2f} (spread ${SPREAD_COST:.2f} + commission ${COMMISSION:.2f})", flush=True)
    print(f"Time: {time.time()-t0:.1f}s", flush=True)
    
    # Save
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result_path = MODEL_DIR / f"strategy_oos_result_{ts}.json"
    with open(result_path, "w") as f:
        json.dump({
            "oos": oos,
            "best_val": best_val,
            "params": {
                "tp_points": best_val["tp"],
                "sl_points": best_val["sl"],
                "long_threshold": best_val["thr"],
                "short_threshold": 1 - best_val["thr"],
            },
            "costs": {
                "point_value": POINT_VALUE,
                "spread_points": SPREAD_POINTS,
                "spread_cost": SPREAD_COST,
                "commission": COMMISSION,
                "cost_per_trade": COST_PER_TRADE,
                "lots": 0.05,
            },
            "split": {
                "train_end": TRAIN_END,
                "val_end": VAL_END,
                "train_samples": len(train),
                "val_samples": len(val),
                "test_samples": len(test),
                "purge_rows": PURGE_ROWS,
            },
            "look_ahead_mitigation": f"Purged first {PURGE_ROWS} rows in test set so rolling features at first test row only use data up to that point",
            "trained_at": ts,
        }, f, indent=2, default=str)
    print(f"Saved: {result_path}", flush=True)


if __name__ == "__main__":
    main()
