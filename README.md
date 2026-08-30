# Meffs

XAUUSD multi-timeframe data + ML strategy package.

## Data

OHLCV CSVs under `data/`:
- `xauusd_m1/`
- `xauusd_m5/`
- `xauusd_m15/`
- `xauusd_h1/`
- `btcusd_m1/`
- `btcusd_m5/`

Format: `timestamp,open,high,low,close` (Unix ms).

### XAUUSD M1 (Dukascopy)

Monatliche Dateien: `data/xauusd_m1/XAUUSD_M1_YYYY_MM.csv`

Lokal herunterladen:

```bash
pip install requests
python scripts/download_xauusd_m1.py --out data/xauusd_m1 --start 2022-01
```

Oder GitHub Action **Download XAUUSD M1** (Actions → Run workflow) starten.
Quelle: Dukascopy BID `candles_min_1` (`datafeed.dukascopy.com`).

### BTCUSD M1 (Dukascopy)

Monatliche Dateien: `data/btcusd_m1/BTCUSD_M1_YYYY_MM.csv`
Zeitraum: 2022-01 bis 2026-08, BID, M1.

Lokal herunterladen:

```bash
bash scripts/download_btcusd_m1.sh data/btcusd_m1 2022-01 2026-08
```

Oder GitHub Action **Download BTCUSD M1** (Actions → Run workflow) starten.
Quelle: Dukascopy via `dukascopy-node` (`btcusd`, timeframe `m1`, price `bid`).

### BTCUSD M5 (Dukascopy)

Monatliche Dateien: `data/btcusd_m5/BTCUSD_M5_YYYY_MM.csv`
Zeitraum: 2022-01 bis 2026-08, BID, M5.

Lokal herunterladen:

```bash
bash scripts/download_btcusd_m5.sh data/btcusd_m5 2022-01 2026-08
```

Oder GitHub Action **Download BTCUSD M5** (Actions → Run workflow) starten.
Quelle: Dukascopy via `dukascopy-node` (`btcusd`, timeframe `m5`, price `bid`).

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

## Zone strategy (Demand / Supply)

Look-ahead-free port of the TradingView `XAU Zone Pro` setup plus a LightGBM filter.
See `ml_strategy/ZONE_STRATEGY.md`.

```bash
cd ml_strategy/code
python zone_backtest.py --tf m15
python train_zone_filter.py --tf m15 --train-end 2025-06-30 --val-end 2025-12-31
python zone_filter_backtest.py --tf m15 --threshold 0.74 --oos-start 2026-01-01
```

**Disclaimer:** Research only. Not financial advice. Paper-trade first.
