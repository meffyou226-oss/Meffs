#!/usr/bin/env python3
"""Load XAUUSD CSVs from the repo data/ folder."""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def _read_csv_sorted(path: Path) -> pd.DataFrame:
    skip_keywords = ["<<<<<<<", ">>>>>>>", "=======", "undefined"]
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line for line in f if line.strip() and not any(k in line for k in skip_keywords)]
    df = pd.read_csv(StringIO("".join(lines)))
    df.columns = [c.strip().lower() for c in df.columns]
    needed = {"timestamp", "open", "high", "low", "close"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing columns in {path}: {needed - set(df.columns)}")
    df = df[["timestamp", "open", "high", "low", "close"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float).astype(int), unit="ms", utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return df


def load_tf(tf: str, drop_flat: bool = True) -> pd.DataFrame:
    tf_dir = DATA_DIR / f"xauusd_{tf}"
    files = sorted(
        p
        for p in tf_dir.glob("*.csv")
        if not (p.stem.endswith("_H1") or p.stem.endswith("_H2"))
    )
    if not files:
        raise FileNotFoundError(tf_dir)
    frames = []
    for f in files:
        try:
            frames.append(_read_csv_sorted(f))
        except Exception as e:
            print(f"[WARN] skip {f.name}: {e}", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    if drop_flat:
        rng = df["high"] - df["low"]
        df = df[rng > 1e-8].reset_index(drop=True)
    print(
        f"[{tf}] loaded {len(df):,} rows {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}",
        flush=True,
    )
    return df
