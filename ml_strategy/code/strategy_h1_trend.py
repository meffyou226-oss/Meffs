#!/usr/bin/env python3
"""
Meffs XAUUSD – H1 Trend Strategy v2 (improved, leak-free)
=========================================================
Primary rule:
  Long:  EMA21 > EMA50 > EMA100  and  ADX >= 28
  Short: EMA21 < EMA50 < EMA100  and  ADX >= 28
  Session: London only 07–12 UTC
  Entry: next H1 open after signal bar is closed
  Exit:  TP 3.0xATR | SL 1.5xATR | time stop 36 H1 bars

v1 vs v2 (tests, spread $0.40, 0.05 lot):
  v1 OOS: more trades, ~0.09 R/trade
  v2 OOS: fewer trades, higher avg R
  Params ranked on same OOS window → re-check on new data.

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


def run(
    df: pd.DataFrame,
    lot: float = 0.05,
    spread: float = 0.40,
    tp_mult: float = 3.0,
    sl_mult: float = 1.5,
    horizon: int = 36,
    adx_min: float = 28.0,
    session=(7, 12),
    use_htf_stack: bool = True,
) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]
    df = df.copy()
    df["ema21"] = EMAIndicator(c, 21).ema_indicator()
    df["ema50"] = EMAIndicator(c, 50).ema_indicator()
    df["ema100"] = EMAIndicator(c, 100).ema_indicator()
    df["atr"] = AverageTrueRange(h, l, c, 14).average_true_range()
    df["adx"] = ADXIndicator(h, l, c, 14).adx()
    df["hour"] = df["datetime"].dt.hour
    df["session"] = df["hour"].between(session[0], session[1])

    df["side"] = 0
    if use_htf_stack:
        long_c = (df["ema21"] > df["ema50"]) & (df["ema50"] > df["ema100"]) & (df["adx"] >= adx_min)
        short_c = (df["ema21"] < df["ema50"]) & (df["ema50"] < df["ema100"]) & (df["adx"] >= adx_min)
    else:
        long_c = (df["ema21"] > df["ema50"]) & (df["adx"] >= adx_min)
        short_c = (df["ema21"] < df["ema50"]) & (df["adx"] >= adx_min)
    df.loc[long_c, "side"] = 1
    df.loc[short_c, "side"] = -1

    cl, hi, lo, atr = df["close"].values, df["high"].values, df["low"].values, df["atr"].values
    n = len(df)
    trades = []
    point_value = 100.0 * lot

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
        cost = spread * point_value
        pnl_usd = pnl_points * point_value - cost
        trades.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "side": "LONG" if side == 1 else "SHORT",
            "entry": entry,
            "exit": exit_price,
            "atr": a,
            "pnl_points": pnl_points,
            "pnl_usd": pnl_usd,
            "pnl_R": pnl_points / a,
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
    print("H1 Trend v2 | London 07-12 | EMA21>50>100 + ADX | TP3/SL1.5")
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
    ap.add_argument("--out", type=str, default="h1_trend_v2_trades.csv")
    ap.add_argument("--adx", type=float, default=28.0)
    ap.add_argument("--tp", type=float, default=3.0)
    ap.add_argument("--sl", type=float, default=1.5)
    ap.add_argument("--horizon", type=int, default=36)
    args = ap.parse_args()

    df = load_h1(args.h1_dir)
    if args.from_date:
        df = df[df.datetime >= args.from_date].reset_index(drop=True)
        print(f"From {args.from_date}: {len(df)} bars")
    trades = run(
        df, lot=args.lot, spread=args.spread,
        tp_mult=args.tp, sl_mult=args.sl, horizon=args.horizon, adx_min=args.adx,
    )
    if len(trades):
        trades.to_csv(args.out, index=False)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
