#!/usr/bin/env python3
"""Reassemble split LightGBM model files into lgb_xauusd_multi_tf.txt"""
from pathlib import Path
parts = sorted(Path(__file__).parent.glob("model_part_*"))
if not parts:
    raise SystemExit("No model_part_* files found. Download them or place the full lgb_xauusd_multi_tf.txt here.")
out = Path(__file__).parent / "lgb_xauusd_multi_tf.txt"
with open(out, "wb") as f:
    for p in parts:
        f.write(p.read_bytes())
print(f"Wrote {out} ({out.stat().st_size} bytes) from {len(parts)} parts")
