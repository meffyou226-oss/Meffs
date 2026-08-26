# Regime Ensemble Strategy

## Architecture

| Component | Role | Model / Rule |
|-----------|------|----------------|
| **Trend branch** | Primary edge | EMA21>50>100, ADX>=28, London 07-12, TP3/SL1.5 |
| **MR branch** | Range markets | RSI extreme + ADX<20 + BB extreme |
| **Vol model** | Risk sizing | LightGBM: P(high vol next 12 H1) → half size if >=0.55 |

Meta-label ML on trend success **overfit** (Val AUC 0.67 → Test 0.50) — not used in production path.

Vol model OOS AUC ~0.85 (vol clustering is predictable).

## OOS 2025-07+ (0.05 lot, spread $0.40)

| Variant | n | WR | PF | $ PnL | Max DD |
|---------|---|-----|-----|-------|--------|
| Trend London only | 618 | 43% | 1.14 | ~$8.5k | ~$11k |
| **Trend + vol size** | 618 | 43% | **1.24** | **~$11.0k** | ~$10.7k |
| Full (+MR) | 656 | 43% | 1.22 | ~$10.4k | ~$10.7k |

Vol sizing improves PF and $ without changing trade count (same entries, smaller risk in high-vol forecasts).

## Caveats

- Drawdowns remain large relative to profit — size small.
- Some months (e.g. 2026 Q2) are strongly negative.
- Not financial advice; paper trade first.

## Run

```bash
python code/strategy_regime_ensemble.py --h1_dir ../data/xauusd_h1 --lot 0.05 --from_date 2025-07-01
# without MR:
python code/strategy_regime_ensemble.py --h1_dir ../data/xauusd_h1 --no_mr --lot 0.05
```

Place `regime_ensemble.pkl` in `model/` for vol sizing (or train via train_regime_vol.py).
