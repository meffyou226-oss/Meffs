#!/usr/bin/env python3
"""
Meffs XAUUSD Multi-Timeframe Strategy
=====================================
- Model: LightGBM trained on M5 + M15 + H1 features
- Target: Next M15 bar direction
- OOS period used in development: 2025-07-01 → 2026-08-26

Requirements:
    pip install lightgbm pandas numpy ta scikit-learn

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
        raise ValueError("No valid data loaded")
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    mask = (df["high"] - df["low"]) > 0.05
    return df[mask].reset_index(drop=True)


def build_multi_tf_frame(m15: pd.DataFrame, h1: pd.DataFrame, m5: pd.DataFrame) -> pd.DataFrame:
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
        c for c in m5f.columns
        if c.startswith("m5_") and any(
            x in c for x in ["rsi_14", "rsi_7", "macd", "bb_pct", "atr_pct", "ema9_21",
                            "ret_1", "ret_3", "ret_5", "vol_10", "adx", "stoch", "body", "range"]
        )
    ]
    base = pd.merge_asof(
        base.sort_values("timestamp"),
        m5f[["timestamp"] + m5_keep].sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    return base


def run_backtest(
    df: pd.DataFrame,
    proba_col: str = "proba",
    entry_long: float = 0.58,
    entry_short: float = 0.42,
    hold_bars: int = 1,
    lot_size: float = 0.05,
    spread: float = 0.40,
    slippage: float = 0.05,
) -> pd.DataFrame:
    point_value = 100.0 * lot_size
    cost = (spread + slippage) * point_value
    trades = []
    equity = [0.0]
    position = 0
    entry_price = 0.0
    entry_idx = 0
    for i in range(len(df) - hold_bars):
        proba = df.loc[i, proba_col]
        price = df.loc[i, "close"]
        if position != 0 and (i - entry_idx) >= hold_bars:
            exit_price = df.loc[i, "close"]
            pnl_points = (exit_price - entry_price) * position
            pnl_usd = pnl_points * point_value - cost
            trades.append({
                "entry_time": df.loc[entry_idx, "datetime"],
                "exit_time": df.loc[i, "datetime"],
                "side": "LONG" if position == 1 else "SHORT",
                "entry": entry_price,
                "exit": exit_price,
                "pnl_points": pnl_points,
                "pnl_usd": pnl_usd,
                "proba": df.loc[entry_idx, proba_col],
            })
            equity.append(equity[-1] + pnl_usd)
            position = 0
        if position == 0:
            if proba >= entry_long:
                position = 1
                entry_price = price
                entry_idx = i
            elif proba <= entry_short:
                position = -1
                entry_price = price
                entry_idx = i
    if position != 0:
        i = len(df) - 1
        exit_price = df.loc[i, "close"]
        pnl_points = (exit_price - entry_price) * position
        pnl_usd = pnl_points * point_value - cost
        trades.append({
            "entry_time": df.loc[entry_idx, "datetime"],
            "exit_time": df.loc[i, "datetime"],
            "side": "LONG" if position == 1 else "SHORT",
            "entry": entry_price,
            "exit": exit_price,
            "pnl_points": pnl_points,
            "pnl_usd": pnl_usd,
            "proba": df.loc[entry_idx, proba_col],
        })
        equity.append(equity[-1] + pnl_usd)
    tdf = pd.DataFrame(trades)
    if len(tdf) == 0:
        print("No trades generated.")
        return tdf
    wins = tdf[tdf.pnl_usd > 0]
    losses = tdf[tdf.pnl_usd <= 0]
    total = tdf.pnl_usd.sum()
    wr = len(wins) / len(tdf) * 100
    pf = wins.pnl_usd.sum() / abs(losses.pnl_usd.sum()) if len(losses) and losses.pnl_usd.sum() != 0 else float("inf")
    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak).min()
    print("=" * 60)
    print(f"Trades: {len(tdf):,} | Winrate: {wr:.1f}%")
    print(f"Total PnL: ${total:,.2f}")
    print(f"Avg Win: ${wins.pnl_usd.mean():.2f} | Avg Loss: ${losses.pnl_usd.mean():.2f}")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Max Drawdown: ${dd:,.2f}")
    print(f"Final Equity: ${eq[-1]:,.2f}")
    print(f"Long: {(tdf.side == 'LONG').sum()} | Short: {(tdf.side == 'SHORT').sum()}")
    print("=" * 60)
    return tdf


def main():
    parser = argparse.ArgumentParser(description="Meffs XAUUSD Multi-TF Strategy")
    parser.add_argument("--m15_dir", type=str, help="Folder with M15 CSVs")
    parser.add_argument("--h1_dir", type=str, help="Folder with H1 CSVs")
    parser.add_argument("--m5_dir", type=str, help="Folder with M5 CSVs")
    parser.add_argument("--lot", type=float, default=0.05, help="Lot size (default 0.05)")
    parser.add_argument("--long_th", type=float, default=0.58, help="Long threshold")
    parser.add_argument("--short_th", type=float, default=0.42, help="Short threshold")
    parser.add_argument("--hold", type=int, default=1, help="Hold bars")
    parser.add_argument("--predict_only", action="store_true", help="Only print latest signals")
    parser.add_argument("--out", type=str, default="trades_out.csv", help="Output trades CSV")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = lgb.Booster(model_file=str(MODEL_PATH))
    with open(FEAT_PATH) as f:
        feat_cols = json.load(f)

    if not args.m15_dir:
        print("No --m15_dir given. Please provide your data folders.")
        print("Example:")
        print("  python strategy_backtest.py --m15_dir ../data/xauusd_m15 --h1_dir ../data/xauusd_h1 --m5_dir ../data/xauusd_m5 --lot 0.05")
        return

    print("Loading data...")
    m15 = load_ohlc_folder(args.m15_dir)
    h1 = load_ohlc_folder(args.h1_dir) if args.h1_dir else m15
    m5 = load_ohlc_folder(args.m5_dir) if args.m5_dir else m15

    print("Building multi-TF features...")
    frame = build_multi_tf_frame(m15, h1, m5)
    frame = frame.dropna(subset=feat_cols).reset_index(drop=True)

    print("Predicting...")
    frame["proba"] = model.predict(frame[feat_cols])
    frame["signal"] = 0
    frame.loc[frame["proba"] >= args.long_th, "signal"] = 1
    frame.loc[frame["proba"] <= args.short_th, "signal"] = -1

    if args.predict_only:
        last = frame.tail(10)[["datetime", "close", "proba", "signal"]]
        print("\nLatest signals:")
        print(last.to_string(index=False))
        return

    print(f"\nRunning backtest (lot={args.lot}, long>={args.long_th}, short<={args.short_th}, hold={args.hold})...")
    trades = run_backtest(
        frame,
        entry_long=args.long_th,
        entry_short=args.short_th,
        hold_bars=args.hold,
        lot_size=args.lot,
    )
    if len(trades) > 0:
        trades.to_csv(args.out, index=False)
        print(f"Trades saved → {args.out}")


if __name__ == "__main__":
    main()
