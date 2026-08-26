#!/usr/bin/env python3
"""
Meffs XAUUSD Multi-TF Strategy (LOOKAHEAD-FIXED)
================================================
H1/M5 joined only after bar completion (available_at = open + period).
See results/LOOKAHEAD_BIAS_NOTICE.md

Usage:
  python strategy_backtest.py --m15_dir ../data/xauusd_m15 --h1_dir ../data/xauusd_h1 --m5_dir ../data/xauusd_m5 --lot 0.05
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "lgb_xauusd_multi_tf.txt"
FEAT_PATH = ROOT / "model" / "feat_cols.json"
PKL_PATH = ROOT / "model" / "lgb_xauusd_multi_tf.pkl"


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
        raise ValueError("No valid data")
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    mask = (df["high"] - df["low"]) > 0.05
    return df[mask].reset_index(drop=True)


def build_multi_tf_frame(m15, h1, m5):
    """No lookahead: only COMPLETED H1/M5 bars."""
    m15f = add_features(m15, "")
    h1f = add_features(h1, "h1_")
    m5f = add_features(m5, "m5_")
    H1_MS, M5_MS = 60 * 60 * 1000, 5 * 60 * 1000
    base = m15f.copy()
    h1_cols = [c for c in h1f.columns if c.startswith("h1_")]
    h1_align = h1f[["timestamp"] + h1_cols].copy()
    h1_align["available_at"] = h1_align["timestamp"] + H1_MS
    base = pd.merge_asof(
        base.sort_values("timestamp"),
        h1_align.sort_values("available_at").drop(columns=["timestamp"]),
        left_on="timestamp", right_on="available_at", direction="backward",
    ).drop(columns=["available_at"], errors="ignore")
    m5_keep = [
        c for c in m5f.columns if c.startswith("m5_") and any(
            x in c for x in ["rsi_14", "rsi_7", "macd", "bb_pct", "atr_pct", "ema9_21",
                            "ret_1", "ret_3", "ret_5", "vol_10", "adx", "stoch", "body", "range"]
        )
    ]
    m5_align = m5f[["timestamp"] + m5_keep].copy()
    m5_align["available_at"] = m5_align["timestamp"] + M5_MS
    base = pd.merge_asof(
        base.sort_values("timestamp"),
        m5_align.sort_values("available_at").drop(columns=["timestamp"]),
        left_on="timestamp", right_on="available_at", direction="backward",
    ).drop(columns=["available_at"], errors="ignore")
    return base


def run_backtest(df, proba_col="proba", entry_long=0.55, entry_short=0.45,
                 hold_bars=1, lot_size=0.05, spread=0.40, slippage=0.05):
    point_value = 100.0 * lot_size
    cost = (spread + slippage) * point_value
    trades, equity = [], [0.0]
    position, entry_price, entry_idx = 0, 0.0, 0
    for i in range(len(df) - hold_bars):
        proba, price = df.loc[i, proba_col], df.loc[i, "close"]
        if position != 0 and (i - entry_idx) >= hold_bars:
            exit_price = df.loc[i, "close"]
            pnl = (exit_price - entry_price) * position * point_value - cost
            trades.append({"entry_time": df.loc[entry_idx, "datetime"], "exit_time": df.loc[i, "datetime"],
                           "side": "LONG" if position == 1 else "SHORT", "pnl_usd": pnl, "proba": df.loc[entry_idx, proba_col]})
            equity.append(equity[-1] + pnl)
            position = 0
        if position == 0:
            if proba >= entry_long:
                position, entry_price, entry_idx = 1, price, i
            elif proba <= entry_short:
                position, entry_price, entry_idx = -1, price, i
    tdf = pd.DataFrame(trades)
    if len(tdf) == 0:
        print("No trades"); return tdf
    wins = tdf[tdf.pnl_usd > 0]
    print(f"Trades: {len(tdf)} | Winrate: {len(wins)/len(tdf)*100:.1f}% | PnL: ${tdf.pnl_usd.sum():,.2f}")
    return tdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m15_dir", type=str)
    parser.add_argument("--h1_dir", type=str)
    parser.add_argument("--m5_dir", type=str)
    parser.add_argument("--lot", type=float, default=0.05)
    parser.add_argument("--long_th", type=float, default=0.55)
    parser.add_argument("--short_th", type=float, default=0.45)
    parser.add_argument("--hold", type=int, default=1)
    parser.add_argument("--predict_only", action="store_true")
    parser.add_argument("--out", type=str, default="trades_out.csv")
    args = parser.parse_args()

    if PKL_PATH.exists():
        import joblib
        d = joblib.load(PKL_PATH)
        model, feat_cols = d["model"], d["feat_cols"]
        print("Loaded .pkl")
    elif MODEL_PATH.exists():
        model = lgb.Booster(model_file=str(MODEL_PATH))
        feat_cols = json.loads(FEAT_PATH.read_text())
        print("Loaded .txt")
    else:
        raise FileNotFoundError("Place model in model/ (after leak-free retrain)")

    if not args.m15_dir:
        print("Need --m15_dir --h1_dir --m5_dir")
        return

    m15 = load_ohlc_folder(args.m15_dir)
    h1 = load_ohlc_folder(args.h1_dir) if args.h1_dir else m15
    m5 = load_ohlc_folder(args.m5_dir) if args.m5_dir else m15
    frame = build_multi_tf_frame(m15, h1, m5).dropna(subset=feat_cols).reset_index(drop=True)
    frame["proba"] = model.predict(frame[feat_cols])

    if args.predict_only:
        print(frame.tail(10)[["datetime", "close", "proba"]])
        return

    trades = run_backtest(frame, entry_long=args.long_th, entry_short=args.short_th,
                          hold_bars=args.hold, lot_size=args.lot)
    if len(trades):
        trades.to_csv(args.out, index=False)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
