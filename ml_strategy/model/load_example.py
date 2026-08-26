#!/usr/bin/env python3
"""Example: load the Meffs XAUUSD model (.pkl or .txt)."""
from pathlib import Path

HERE = Path(__file__).parent

def load_model():
    pkl = HERE / "lgb_xauusd_multi_tf.pkl"
    txt = HERE / "lgb_xauusd_multi_tf.txt"
    if pkl.exists():
        import joblib
        d = joblib.load(pkl)
        return d["model"], d["feat_cols"], d.get("meta", {})
    if txt.exists():
        import lightgbm as lgb, json
        model = lgb.Booster(model_file=str(txt))
        feat_cols = json.loads((HERE / "feat_cols.json").read_text())
        return model, feat_cols, {}
    raise FileNotFoundError("Put lgb_xauusd_multi_tf.pkl or .txt into model/")

if __name__ == "__main__":
    model, feat_cols, meta = load_model()
    print("Model loaded. Features:", len(feat_cols))
    print("Meta:", meta)
    print("Type:", type(model))
