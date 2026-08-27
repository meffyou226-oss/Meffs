# XAU Demand / Supply Zone Strategy (Python)

Port of the TradingView indicator `XAU Demand + Supply Zone Setup Pro`.

## Look-ahead rules

- Swing pivots are only created on the **confirmation bar** (`pivot_index + swing_len`).
- Zone high/low uses only the confirmed pivot candle.
- Features for ML are taken at the **entry bar**, never from future bars.
- Exits start on the bar **after** entry.
- If SL and TP trade in the same bar, **SL wins** (conservative).
- Training uses a time split. Train rows whose `exit_time` is after `train_end` are **purged**.

## Run

From repo root:

```bash
pip install numpy pandas scikit-learn lightgbm
cd ml_strategy/code

# 1) rule-only backtest
python zone_backtest.py --tf m15

# 2) train filter (OOS after 2025-12-31)
python train_zone_filter.py --tf m15 --train-end 2025-06-30 --val-end 2025-12-31

# 3) apply filter OOS
python zone_filter_backtest.py --tf m15 --threshold 0.74 --oos-start 2026-01-01
```

H1 is also supported (`--tf h1`). M5 produces more trades but noisier zones.

## Model

LightGBM binary classifier: `label_win = TP1 before SL`.

Use the validation-selected probability threshold to skip weak setups.
Default suggestion after training is printed as `val_threshold` / `test_at_val_threshold`.
