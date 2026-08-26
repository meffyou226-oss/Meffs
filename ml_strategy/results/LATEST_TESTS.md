# Latest tests (leak-free, cost $0.40 spread)

## Sources tried
- ATR fixed-dollar risk sizing (industry standard for gold)
- Daily EMA50/200 alignment as HTF filter
- DI+/DI- + ADX + EMA200
- Session filters, vol sizing (previous)

## OOS from 2025-07 (and 2024+ stability)

| Setup | n | WR | PF | $ PnL | Max DD | Notes |
|-------|---|-----|-----|-------|--------|-------|
| v2 fixed 0.05 lot | 618 | 43% | 1.14 | 8.5k | -11.3k | baseline |
| v2 + **$50 ATR risk** | 618 | 43% | **1.35** | 6.8k | **-3.2k** | same signals, better risk |
| **v2 + $50 + daily align** | **316** | **47.5%** | **1.64** | **5.5k** | **-2.5k** | **best risk-adjusted** |
| DI+ADX+EMA200 lon $50 | 698 | 42% | 1.31 | 6.8k | -2.2k | solid alternative |
| DI both sessions $50 | 1619 | 40% | 1.23 | 11.8k | more $ more DD |

### 2024+ check (stability)
- v2 $50 + daily: n=803, WR 44%, **PF 1.47**, sum **$10.8k**, DD **-$2.5k**

## Recommended production hypothesis

1. **Signal:** EMA21 > EMA50 > EMA100, ADX >= 28, London 07-12 UTC  
2. **HTF:** Daily EMA50 vs EMA200 same direction (previous completed day only)  
3. **Exit:** TP 3.0 ATR / SL 1.5 ATR / 36 bars  
4. **Size:** fixed **$ risk** at SL (e.g. $30–50), lot = risk / (1.5 * ATR * 100)  

Much lower DD than fixed 0.05 lot; PF improves. Still not risk-free — paper trade.

## What did not help
- M15 next-bar ML (lookahead or random after fix)
- Trend meta-label ML (val 0.67 → test 0.50 overfit)
- Heavy MR on H1 (too few quality setups)
