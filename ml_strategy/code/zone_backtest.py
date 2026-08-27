#!/usr/bin/env python3
"""Rule-based backtest of the demand/supply zone strategy (no ML)."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from zone_data import load_tf
from zone_engine import ZoneConfig, run_zones, trades_to_frame

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "ml_strategy" / "model"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tf", default="m15", choices=["m5", "m15", "h1"])
    p.add_argument("--swing", type=int, default=5)
    p.add_argument("--impulse", type=float, default=0.6)
    p.add_argument("--rr1", type=float, default=1.0)
    p.add_argument("--rr2", type=float, default=2.0)
    p.add_argument("--sl-buffer", type=float, default=0.15)
    args = p.parse_args()

    df = load_tf(args.tf)
    cfg = ZoneConfig(
        swing_len=args.swing,
        impulse_atr=args.impulse,
        rr1=args.rr1,
        rr2=args.rr2,
        sl_buffer_atr=args.sl_buffer,
    )
    trades, stats = run_zones(df, cfg)
    tdf = trades_to_frame(trades, df)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    trades_path = OUT_DIR / f"zone_trades_{args.tf}_{stamp}.csv"
    stats_path = OUT_DIR / f"zone_backtest_{args.tf}_{stamp}.json"
    if len(tdf):
        tdf.to_csv(trades_path, index=False)
    payload = {
        "tf": args.tf,
        "config": vars(cfg),
        "stats": stats,
        "trades_csv": str(trades_path.name) if len(tdf) else None,
        "note": "Look-ahead free: pivots confirmed after swing_len bars; SL before TP same bar.",
    }
    stats_path.write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str))
    print(f"wrote {stats_path}")


if __name__ == "__main__":
    main()
