#!/usr/bin/env python3
"""
Meffs XAUUSD – Regime Ensemble Strategy
=======================================
  1) TREND: EMA21>50>100, ADX>=28, London 07-12 UTC, TP3/SL1.5, 36 bars
  2) MR (optional): RSI extreme + ADX<20 + BB extreme, TP1.5/SL1, 16 bars
  3) VOL model: half size if elevated vol predicted (next ~12 H1)

Usage:
  python strategy_regime_ensemble.py --h1_dir ../data/xauusd_h1 --lot 0.05
  python strategy_regime_ensemble.py --h1_dir ../data/xauusd_h1 --from_date 2025-07-01 --no_mr
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands

try:
    import joblib
except ImportError:
    joblib = None

FEAT_VOL = [
    "ret_1", "ret_3", "ret_5", "ret_8", "atr_pct", "vol20", "vol_regime",
    "bb_w", "adx", "rsi", "ema9_21", "ema21_50", "body", "range_", "hour", "macd_h",
]

def load_h1(folder: str) -> pd.DataFrame:
    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        raise FileNotFoundError(folder)
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "timestamp" not in df.columns: continue
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna()
            if len(df): dfs.append(df)
        except Exception as e:
            print("skip", f.name, e)
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df[(df["high"] - df["low"]) > 0.05].reset_index(drop=True)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c, h, l, o = d["close"], d["high"], d["low"], d["open"]
    d["ema9"] = EMAIndicator(c, 9).ema_indicator()
    d["ema21"] = EMAIndicator(c, 21).ema_indicator()
    d["ema50"] = EMAIndicator(c, 50).ema_indicator()
    d["ema100"] = EMAIndicator(c, 100).ema_indicator()
    d["atr"] = AverageTrueRange(h, l, c, 14).average_true_range()
    d["atr_pct"] = d["atr"] / c
    d["adx"] = ADXIndicator(h, l, c, 14).adx()
    d["rsi"] = RSIIndicator(c, 14).rsi()
    bb = BollingerBands(c, 20, 2)
    d["bb_pct"] = bb.bollinger_pband()
    d["bb_w"] = bb.bollinger_wband()
    for k in [1, 3, 5, 8]:
        d[f"ret_{k}"] = c.pct_change(k)
    d["vol20"] = d["ret_1"].rolling(20).std()
    d["atr_ma50"] = d["atr"].rolling(50).mean()
    d["vol_regime"] = (d["atr"] / d["atr_ma50"]).clip(0.3, 3)
    d["ema9_21"] = (d["ema9"] - d["ema21"]) / c
    d["ema21_50"] = (d["ema21"] - d["ema50"]) / c
    d["body"] = (c - o) / c
    d["range_"] = (h - l) / c
    d["macd_h"] = MACD(c).macd_diff()
    d["hour"] = d["datetime"].dt.hour
    return d

def run(df, lot=0.05, spread=0.40, use_mr=True, use_vol_size=True, vol_model=None, vol_thr=0.55):
    df = add_features(df)
    n = len(df)
    cl, hi, lo, atr = df["close"].values, df["high"].values, df["low"].values, df["atr"].values
    PV = 100.0 * lot
    p_vol = np.full(n, np.nan)
    if use_vol_size and vol_model is not None:
        valid = df[FEAT_VOL].dropna().index
        p_vol[valid] = vol_model.predict(df.loc[valid, FEAT_VOL])
    side_t = np.zeros(n, dtype=int)
    up = (df["ema21"] > df["ema50"]) & (df["ema50"] > df["ema100"]) & (df["adx"] >= 28)
    dn = (df["ema21"] < df["ema50"]) & (df["ema50"] < df["ema100"]) & (df["adx"] >= 28)
    side_t[up.values] = 1; side_t[dn.values] = -1
    side_m = np.zeros(n, dtype=int)
    m_up = (df["rsi"] < 28) & (df["adx"] < 20) & (df["bb_pct"] < 0.15)
    m_dn = (df["rsi"] > 72) & (df["adx"] < 20) & (df["bb_pct"] > 0.85)
    side_m[m_up.values] = 1; side_m[m_dn.values] = -1
    hours = df["hour"].values
    trades = []

    def path(i, side, tp_m, sl_m, horizon):
        a = atr[i]
        if not np.isfinite(a) or a <= 0: return None
        entry = df["open"].iloc[i + 1]
        tp, sl = entry + side * tp_m * a, entry - side * sl_m * a
        for j in range(1, horizon + 1):
            idx = i + 1 + j
            if idx >= n: break
            if side == 1:
                if hi[idx] >= tp: return tp_m, tp
                if lo[idx] <= sl: return -sl_m, sl
            else:
                if lo[idx] <= tp: return tp_m, tp
                if hi[idx] >= sl: return -sl_m, sl
        last = min(i + 1 + horizon, n - 1)
        return (cl[last] - entry) / a * side, cl[last]

    for i in range(100, n - 40):
        a = atr[i]
        if not np.isfinite(a) or a <= 0: continue
        size = 0.5 if (use_vol_size and np.isfinite(p_vol[i]) and p_vol[i] >= vol_thr) else 1.0
        if side_t[i] != 0 and 7 <= hours[i] <= 12:
            res = path(i, int(side_t[i]), 3.0, 1.5, 36)
            if res:
                r, _ = res
                usd = (r * a * PV - spread * PV) * size
                trades.append({"entry_time": df["datetime"].iloc[i+1], "branch": "trend",
                    "side": "LONG" if side_t[i]==1 else "SHORT", "size_mult": size,
                    "pnl_R": r - spread/a, "pnl_usd": usd})
                continue
        if use_mr and side_m[i] != 0:
            res = path(i, int(side_m[i]), 1.5, 1.0, 16)
            if res:
                r, _ = res
                usd = (r * a * PV - spread * PV) * size
                trades.append({"entry_time": df["datetime"].iloc[i+1], "branch": "mr",
                    "side": "LONG" if side_m[i]==1 else "SHORT", "size_mult": size,
                    "pnl_R": r - spread/a, "pnl_usd": usd})

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        print("No trades"); return tdf
    wins = tdf[tdf.pnl_usd > 0]; losses = tdf[tdf.pnl_usd <= 0]
    eq = tdf["pnl_usd"].cumsum(); dd = (eq - eq.cummax()).min()
    pf = wins.pnl_usd.sum() / abs(losses.pnl_usd.sum()) if len(losses) and losses.pnl_usd.sum() else float("inf")
    print("=" * 60)
    print("Regime Ensemble | Trend London + optional MR + Vol sizing")
    print(f"Trades: {len(tdf):,} | WR: {len(wins)/len(tdf)*100:.1f}%")
    print(f"PnL: ${tdf.pnl_usd.sum():,.2f} | Avg: ${tdf.pnl_usd.mean():.2f}")
    print(f"PF: {pf:.2f} | Max DD: ${dd:,.2f}")
    print(f"Branches: {tdf.branch.value_counts().to_dict()}")
    print(f"Half-size trades: {(tdf.size_mult < 1).sum()}")
    print("=" * 60)
    return tdf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1_dir", required=True)
    ap.add_argument("--lot", type=float, default=0.05)
    ap.add_argument("--spread", type=float, default=0.40)
    ap.add_argument("--from_date", type=str, default=None)
    ap.add_argument("--out", type=str, default="regime_trades.csv")
    ap.add_argument("--no_mr", action="store_true")
    ap.add_argument("--no_vol_size", action="store_true")
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args()
    vol_model = None
    model_path = args.model
    if model_path is None:
        for p in [Path(__file__).resolve().parent.parent / "model" / "regime_ensemble.pkl",
                  Path("regime_ensemble.pkl")]:
            if p.exists():
                model_path = str(p); break
    if model_path and joblib is not None:
        blob = joblib.load(model_path)
        vol_model = blob.get("model_vol")
        print(f"Loaded vol model from {model_path}")
    df = load_h1(args.h1_dir)
    if args.from_date:
        df = df[df.datetime >= args.from_date].reset_index(drop=True)
        print(f"From {args.from_date}: {len(df)} bars")
    trades = run(df, lot=args.lot, spread=args.spread, use_mr=not args.no_mr,
                 use_vol_size=not args.no_vol_size and vol_model is not None, vol_model=vol_model)
    if len(trades):
        trades.to_csv(args.out, index=False)
        print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
