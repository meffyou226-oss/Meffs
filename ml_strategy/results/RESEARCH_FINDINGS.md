# Research Findings (post look-ahead fix)

## 1. M15 next-bar direction ML — dead

After joining H1/M5 only when bars are **complete** (`available_at = open + period`):

| Target | Test AUC |
|--------|----------|
| Next M15 direction | ~0.52 |
| Meta-label (trend + TP before SL) | ~0.52–0.54 |
| Big move >1 ATR in 8 bars | ~0.52 |
| Long TP vs SL path | ~0.50 |

Previous ~71% accuracy was **lookahead leakage**, not edge.

## 2. What works better (modestly)

### H1 Trend rule (no ML)

- **Long:** EMA21 > EMA50 and ADX ≥ 25  
- **Short:** EMA21 < EMA50 and ADX ≥ 25  
- Session filter: 07–20 UTC  
- Entry: next H1 **open** after signal bar closes  
- Exit: TP 2×ATR / SL 1×ATR / max 24 bars  
- Cost: spread $0.40  

**OOS (2025-07 → data end, 0.05 lot scale in R terms):**

- Winrate ~ **37.5%** (expected with 2R:1R payoff)
- Profit factor ~ **1.18**
- Net ~ **+0.08 R per trade** after spread
- Max DD order of **~80 R** on full OOS path — size carefully

This is a **thin** edge, not a money printer. Useful as honest baseline.

### Meta-labeling on same primary

Walk-forward AUC ~0.51–0.54; filtering rarely improved net R after costs in our folds.
Not worth complexity yet on this feature set.

## 3. Recommendations

1. Prefer **H1 (or higher)** decisions over M15 next-bar classification.
2. Use **path-dependent exits** (TP/SL), not fixed-horizon direction labels.
3. Always **availability-time** merge for multi-TF features.
4. Validate with **walk-forward**; single OOS window is weak evidence.
5. Start from the **rule baseline** (`strategy_h1_trend.py`) before adding ML filters.
6. Risk small: edge is ~0.05–0.1 R/trade after costs.

## 4. Files

- `code/strategy_h1_trend.py` — runnable baseline  
- `code/train_model.py` — leak-fixed training (expect ~50–53% if used for direction)  
- `results/LOOKAHEAD_BIAS_NOTICE.md` — why old results are invalid  
