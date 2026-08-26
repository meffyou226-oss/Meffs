# Meffs XAUUSD Multi-Timeframe Strategy Package

**Symbol:** XAUUSD (Gold)  
**Timeframes:** M5 + M15 + H1  
**Model:** LightGBM (binary direction of next M15 bar)  
**OOS Period (development):** 2025-07-01 → 2026-08-26  
**Reported OOS Accuracy:** ~71 %  

---

## Package Contents

```
ml_strategy/
├── README.md
├── requirements.txt
├── model/
│   ├── lgb_xauusd_multi_tf.txt   # Trained LightGBM model (push separately if large)
│   └── feat_cols.json
├── results/
│   └── OOS_SUMMARY.txt
└── code/
    └── strategy_backtest.py
```

---

## Quick Start

```bash
pip install -r requirements.txt

python code/strategy_backtest.py \
  --m15_dir ../data/xauusd_m15 \
  --h1_dir  ../data/xauusd_h1 \
  --m5_dir  ../data/xauusd_m5 \
  --lot 0.05 \
  --long_th 0.58 \
  --short_th 0.42 \
  --hold 1 \
  --out my_trades.csv
```

CSV format: `timestamp,open,high,low,close` (timestamp = Unix ms).

---

## Recommended Strategies (OOS)

| Name | Long | Short | Hold | Winrate | Profit Factor |
|------|------|-------|------|---------|---------------|
| B (balanced) | 0.58 | 0.42 | 1 | ~71% | ~3.45 |
| D (high conf) | 0.65 | 0.35 | 1 | ~75.5% | ~4.65 |
| C (hold 2) | 0.62 | 0.38 | 2 | ~76% | ~5.11 |

Lot 0.05 | Cost ~$2.25 RT | $5 per $1 move

---

## OOS Results (0.05 lots, Jul 2025 – Aug 2026)

- Strategy B: **+$267,224** | Max DD ~ –$1,010
- Strategy D: **+$256,976** | Max DD ~ –$1,196
- All months positive under stated assumptions.

---

## Disclaimer

Past performance is not a guarantee of future results. Research code only — not financial advice. Paper-trade first.
