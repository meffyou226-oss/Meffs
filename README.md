# Meffs

XAUUSD multi-timeframe data + ML strategy package.

## Data

OHLCV CSVs under `data/`:
- `xauusd_m5/`
- `xauusd_m15/`
- `xauusd_h1/`

Format: `timestamp,open,high,low,close` (Unix ms).

## ML Strategy

See **[ml_strategy/](ml_strategy/)** for:

- LightGBM multi-TF direction model (M5 + M15 + H1 features)
- Backtest script (`code/strategy_backtest.py`)
- OOS results summary (~71% accuracy, Jul 2025 – Aug 2026)
- Recommended thresholds (e.g. long ≥ 0.58 / short ≤ 0.42)

```bash
cd ml_strategy
pip install -r requirements.txt
# place lgb_xauusd_multi_tf.txt into model/ (see model/HOW_TO_GET_MODEL.txt)
python code/strategy_backtest.py \
  --m15_dir ../data/xauusd_m15 \
  --h1_dir  ../data/xauusd_h1 \
  --m5_dir  ../data/xauusd_m5 \
  --lot 0.05
```

**Disclaimer:** Research only. Not financial advice. Paper-trade first.
