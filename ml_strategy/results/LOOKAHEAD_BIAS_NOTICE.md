# CRITICAL: Lookahead Bias – Previous Results INVALID

## What was wrong

`build_multi_tf_frame()` used `merge_asof(direction="backward")` on **bar open** timestamps.

Timestamps in the data are bar **open** times. An H1 bar that opens at T is only complete at T+1h. Joining that bar's close/RSI/body onto M15 bars during [T, T+1h) leaks up to ~45 minutes of future H1 information.

Same issue for M5 (bar only complete at open+5m).

## Evidence

| Setup | Test Accuracy | Test AUC |
|-------|---------------|----------|
| **Bugged** (open-time merge) | **70.97%** | **0.787** |
| **Fixed** (availability = open + period) | **51.5%** | **0.515** |

After the fix, performance collapses to near-random (baseline ~50.6% positive rate).

`h1_ret_1` dominated feature importance under the bug — classic symptom of leaking the still-forming H1 bar.

## What is invalid

- All numbers in the old `OOS_SUMMARY.txt` (strategies A–D, PnL figures)
- Claims of ~71% OOS accuracy / high profit factors
- Any model trained with the old merge logic

## Fix

For H1/M5 features, set availability time to **bar open + bar duration**, then `merge_asof` on that:

```python
h1_align["available_at"] = h1_align["timestamp"] + 60 * 60 * 1000  # +1h
m5_align["available_at"] = m5_align["timestamp"] + 5 * 60 * 1000   # +5m

base = pd.merge_asof(
    base.sort_values("timestamp"),
    h1_align.sort_values("available_at"),
    left_on="timestamp", right_on="available_at",
    direction="backward",
)
```

Only **completed** higher/lower TF bars are used at decision time.

## Next steps

1. Use corrected `train_model.py` / `strategy_backtest.py` (availability-time merge).
2. Retrain; do not use old models for live trading.
3. Expect ~50–53% direction accuracy unless a real edge is found with other features/horizons.

Apologies for the invalid earlier results.
