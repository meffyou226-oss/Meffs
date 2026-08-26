#!/usr/bin/env python3
"""Rebuild regime_ensemble.pkl from base64 parts."""
from pathlib import Path
import base64
parts = sorted(Path(__file__).parent.glob("regime_ensemble.pkl.b64.part*"))
if not parts:
    raise SystemExit("No regime_ensemble.pkl.b64.part* files found")
data = base64.b64decode("".join(p.read_text().strip() for p in parts))
out = Path(__file__).parent / "regime_ensemble.pkl"
out.write_bytes(data)
print(f"Wrote {out} ({len(data)} bytes)")
print("Load: import joblib; m = joblib.load('regime_ensemble.pkl')['model_vol']")
