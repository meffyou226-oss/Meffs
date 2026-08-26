#!/usr/bin/env python3
"""
Meffs XAUUSD – H1 Trend Baseline (leak-free, cost-aware)
========================================================
Rule (no ML):
  Long:  EMA21 > EMA50 and ADX >= 25
  Short: EMA21 < EMA50 and ADX >= 25
  Only London+NY session hours (07–20 UTC)
  Entry: next H1 open after signal bar closes
  Exit:  TP = 2.0 * ATR | SL = 1.0 * ATR | time stop 24 H1 bars

After fixing lookahead bias, M15 next-bar ML ≈ random.
This simple rule showed a modest positive OOS edge under realistic spread.

Usage:
  python strategy_h1_trend.py --h1_dir ../data/xauusd_h1 --lot 0.05
  python strategy_h1_trend.py --h1_dir ../data/xauusd_h1 --from_date 2025-07-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ta.trend import ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange


def load_h1(folder: str) -> pd.DataFrame:
    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        raise FileNotFoundError(folder)
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "timestamp" not in df.columns:
                continue
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna()
            if len(df):
                dfs.append(df)
        except Exception as e:
            print("skip", f.name, e)
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df[(df["high"] - df["low"]) > 0.05].reset_index(drop=True)


def run(df: pd.DataFrame, lot: float = 0.05, spread: float = 0.40,
        tp_mult: float = 2.0, sl_mult: float = 1.0, horizon: int = 24,
        adx_min: float = 25.0, session=(7, 20)):
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    df = df.copy()
    df["ema21"] = EMAIndicator(c, 21).ema_indicator()
    df["ema50"] = EMAIndicator(c, 50).ema_indicator()
    df["atr"] = AverageTrueRange(h, l, c, 14).average_true_range()
    df["adx"] = ADXIndicator(h, l, c, 14).adx()
    df["hour"] = df["datetime"].dt.hour
    df["session"] = df["hour"].between(session[0], session[1])

    df["side"] = 0
    df.loc[(df["ema21"] > df["ema50"]) & (df["adx"] >= adx_min), "side"] = 1
    df.loc[(df["ema21"] < df["ema50"]) & (df["adx"] >= adx_min), "side"] = -1

    cl, hi, lo, atr = df["close"].values, df["high"].values, df["low"].values, df["atr"].values
    n = len(df)
    trades = []

    for i in range(n - horizon - 2):
        side = int(df["side"].iloc[i])
        if side == 0 or not bool(df["session"].iloc[i]):
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        entry = df["open"].iloc[i + 1]
        entry_time = df["datetime"].iloc[i + 1]
        tp = entry + side * tp_mult * a
        sl = entry - side * sl_mult * a
        outcome = 0
        exit_price = entry
        exit_time = entry_time
        for j in range(1, horizon + 1):
            idx = i + 1 + j
            if idx >= n:
                break
            if side == 1:
                if hi[idx] >= tp:
                    outcome, exit_price = 1, tp
                    exit_time = df["datetime"].iloc[idx]
                    break
                if lo[idx] <= sl:
                    outcome, exit_price = -1, sl
                    exit_time = df["datetime"].iloc[idx]
                    break
            else:
                if lo[idx] <= tp:
                    outcome, exit_price = 1, tp
                    exit_time = df["datetime"].iloc[idx]
                    break
                if hi[idx] >= sl:
                    outcome, exit_price = -1, sl
                    exit_time = df["datetime"].iloc[idx]
                    break
        if outcome == 0:
            last = min(i + 1 + horizon, n - 1)
            exit_price = cl[last]
            exit_time = df["datetime"].iloc[last]

        pnl_points = (exit_price - entry) * side
        point_value = 100.0 * lot
        cost = spread * point_value
        pnl_usd = pnl_points * point_value - cost
        pnl_R = pnl_points / a
        trades.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "side": "LONG" if side == 1 else "SHORT",
            "entry": entry,
            "exit": exit_price,
            "atr": a,
            "pnl_points": pnl_points,
            "pnl_usd": pnl_usd,
            "pnl_R": pnl_R,
            "outcome": outcome,
        })

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        print("No trades")
        return tdf

    wins = tdf[tdf.pnl_usd > 0]
    losses = tdf[tdf.pnl_usd <= 0]
    eq = tdf["pnl_usd"].cumsum()
    dd = (eq - eq.cummax()).min()
    pf = wins.pnl_usd.sum() / abs(losses.pnl_usd.sum()) if len(losses) and losses.pnl_usd.sum() != 0 else float("inf")

    print("=" * 60)
    print(f"Trades: {len(tdf):,} | Winrate: {len(wins)/len(tdf)*100:.1f}%")
    print(f"Total PnL: ${tdf.pnl_usd.sum():,.2f} | Avg: ${tdf.pnl_usd.mean():.2f}")
    print(f"Profit Factor: {pf:.2f} | Max DD: ${dd:,.2f}")
    print(f"Avg R (gross): {tdf.pnl_R.mean():.3f}")
    print(f"Long: {(tdf.side=='LONG').sum()} | Short: {(tdf.side=='SHORT').sum()}")
    print("=" * 60)
    return tdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1_dir", required=True)
    ap.add_argument("--lot", type=float, default=0.05)
    ap.add_argument("--spread", type=float, default=0.40)
    ap.add_argument("--from_date", type=str, default=None)
    ap.add_argument("--out", type=str, default="h1_trend_trades.csv")
    args = ap.parse_args()

    df = load_h1(args.h1_dir)
    if args.from_date:
        df = df[df.datetime >= args.from_date].reset_index(drop=True)
        print(f"Filtered from {args.from_date}: {len(df)} bars")
    trades = run(df, lot=args.lot, spread=args.spread)
    if len(trades):
        trades.to_csv(args.out, index=False)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
