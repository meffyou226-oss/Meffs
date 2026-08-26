#!/usr/bin/env python3
"""Rebuild lgb_xauusd_multi_tf.pkl from base64 parts (if present)."""
from pathlib import Path
import base64

parts = sorted(Path(__file__).parent.glob("lgb_xauusd_multi_tf.pkl.b64.part*"))
if not parts:
    raise SystemExit(
        "No .b64.part* files. Instead place lgb_xauusd_multi_tf.pkl here directly\n"
        "or download from the chat and copy into this folder."
    )
b64 = "".join(p.read_text().strip() for p in parts)
data = base64.b64decode(b64)
out = Path(__file__).parent / "lgb_xauusd_multi_tf.pkl"
out.write_bytes(data)
print(f"Wrote {out} ({len(data)} bytes)")
print("Load:")
print("  import joblib")
print("  d = joblib.load('lgb_xauusd_multi_tf.pkl')")
print("  model, feat_cols = d['model'], d['feat_cols']")
