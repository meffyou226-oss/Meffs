#!/usr/bin/env python3
"""Apply a trained zone-filter model OOS. Features at entry only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from zone_data import load_tf
from zone_engine import ZoneConfig, run_zones, trades_to_frame
from train_zone_filter import FEATURE_COLS

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "ml_strategy" / "model"


def latest_model(tf: str) -> Path:
    files = sorted(MODEL_DIR.glob(f"lgb_zone_filter_{tf}_*.txt"))
    if not files:
        raise FileNotFoundError(f"No lgb_zone_filter_{tf}_*.txt in {MODEL_DIR}")
    return files[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="m15")
    ap.add_argument("--model", default="")
    ap.add_argument("--threshold", type=float, default=0.74)
    ap.add_argument("--oos-start", default="2026-01-01")
    args = ap.parse_args()

    df = load_tf(args.tf)
    trades, _ = run_zones(df, ZoneConfig())
    tdf = trades_to_frame(trades, df)
    tdf = tdf[tdf["status"].isin([1, 2])].copy()
    oos = tdf[tdf["entry_time"] >= pd.Timestamp(args.oos_start, tz="UTC")].copy()

    model_path = Path(args.model) if args.model else latest_model(args.tf)
    booster = lgb.Booster(model_file=str(model_path))
    oos["p_win"] = booster.predict(oos[FEATURE_COLS])
    picked = oos[oos["p_win"] >= args.threshold]

    def pack(name, part):
        if not len(part):
            return {name: "empty"}
        return {
            name: {
                "n": int(len(part)),
                "winrate": float(part["label_win"].mean()),
                "expectancy_R": float(part["r_multiple"].mean()),
                "tp2_rate": float(part["tp2_hit"].mean()),
            }
        }

    out = {
        "model": model_path.name,
        "threshold": args.threshold,
        "oos_start": args.oos_start,
        **pack("all_oos", oos),
        **pack("filtered_oos", picked),
    }
    print(json.dumps(out, indent=2))
    dest = MODEL_DIR / f"zone_filter_oos_{args.tf}.json"
    dest.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
