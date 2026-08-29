# macro_gold — XAUUSD Daily + Gold-Makro, point-in-time

Stand Abruf: 2026-08-29.

Panel aus Gold-Tagesschlusskursen und Makroserien. Jede Makrospalte ist
**as-of** auf den Handelstag gemerged (kein Lookahead).

## Coverage

| | |
|---|---|
| Zeitraum | **2016-01-04 bis 2026-08-26** |
| Zeilen | **2669 Handelstage** |
| 2016-2021 | 1508 Tage, `xau_source=comex_gc_f` (Yahoo GC=F) |
| 2022-2026 | 1161 Tage, `xau_source=dukascopy_h1` (Repo `data/xauusd_h1/`) |

Im Repo gibt es keine Dukascopy-H1-Dateien vor 2022. Deshalb ist der
Kalender 2016-2021 COMEX-Gold-Future, ab 2022-01-03 Dukascopy-Spot.

Dateien:

- `out/xauusd_macro_gold_pit_daily.csv` — volles Panel
- `out/xauusd_macro_gold_pit_daily_YYYY.csv` — Jahresdateien
- `out/coverage.csv`

| Spalte | Inhalt |
|---|---|
| date | Handelstag |
| xauusd_open/high/low/close | Tages-OHLC |
| h1_bars | nicht-flache H1-Kerzen (leer bei GC=F) |
| xau_source | `comex_gc_f` oder `dukascopy_h1` |
| real_yield_10y | FRED DFII10 |
| be_inflation_10y | FRED T10YIE |
| usd_broad_tw | FRED DTWEXBGS |
| vix | FRED VIXCLS |
| hy_oas | FRED BAMLH0A0HYM2 (erst ab 2023-08-30) |
| gld_holdings_tonnes / _oz | SPDR GLD Bestände |
| cot_mm_net / _long / _short | CFTC COMEX Gold 088691 Managed Money |
| *_obs_date | Beobachtungsdatum der zuletzt verfuegbaren Zahl |

Voll ab 2016: Realzins, Breakeven, Dollar, VIX, GLD, Goldpreis.
COT leer nur 2016-01-04 bis 2016-01-07 (erster Report 2016-01-05, verfuegbar 2016-01-08).
HY-OAS: FRED liefert BAMLH0A0HYM2 oeffentlich erst ab 2023-08-29.

## Point-in-time

| Quelle | Verzug | available_date |
|---|---|---|
| FRED | 1 Kalendertag | observation_date + 1 |
| SPDR GLD | keiner | observation_date |
| CFTC COT | Freitag fuer Dienstagsstand | report_date + 3 |

`merge_asof(..., direction="backward")` auf `available_date <= date`.
Assert: keine `*_obs_date` nach `date`.

## Quellen

- FRED graph.csv bzw. DFII10-Fallback ivo-welch fredwrap
- SPDR GLD historical-archive API
- CFTC `fut_disagg_txt_YYYY.zip` 2016-2020 + Cache ab 2021
- Dukascopy H1 ab 2022, Yahoo `GC=F` 2016-2021

## Rebuild

```bash
pip install -r macro_gold/requirements.txt
XAUUSD_H1_DIR=data/xauusd_h1 python macro_gold/build_macro_gold.py
```
