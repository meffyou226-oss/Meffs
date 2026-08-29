# macro_gold — XAUUSD Daily + Gold-Makro, point-in-time

Stand Abruf: 2026-08-29.

Panel aus Dukascopy-XAUUSD-Tagesschlusskursen (H1 aus `data/xauusd_h1/`)
und den unten genannten Makroserien. Jede Makrospalte ist **as-of** auf
den Handelstag gemerged: es fliesst nur Information ein, die an diesem
Tag bereits veröffentlicht war (kein Lookahead).

## Ergebnisdatei

Nach `python build_macro_gold.py`:

`out/xauusd_macro_gold_pit_daily.csv`

Coverage des Abrufs 2026-08-29 steht in `out/coverage.csv`.

- Zeitraum: 2022-01-03 bis 2026-08-26
- 1.161 Handelstage (UTC-Kalendertag mit mindestens 4 echten H1-Kerzen)
- XAUUSD = Dukascopy-Spot (nicht COMEX-Future)

| Spalte | Inhalt |
|---|---|
| date | UTC-Handelstag (Basis-Kalender) |
| xauusd_open/high/low/close | aus H1 aggregiert |
| h1_bars | Anzahl nicht-flacher H1-Kerzen |
| real_yield_10y | FRED DFII10, 10Y TIPS-Realzins (%) |
| be_inflation_10y | FRED T10YIE, 10Y Inflations-Breakeven (%) |
| usd_broad_tw | FRED DTWEXBGS, nominaler Broad-Dollar |
| vix | FRED VIXCLS |
| hy_oas | FRED BAMLH0A0HYM2, ICE BofA US HY OAS (%) |
| gld_holdings_tonnes / _oz | SPDR GLD physische Bestände |
| cot_mm_net / _long / _short | CFTC Disagg. Futures, COMEX Gold 088691, Managed Money |
| *_obs_date | Beobachtungsdatum der zuletzt verfügbaren Zahl |

Leere `hy_oas` vor 2023-08-30: FRED liefert die Serie aktuell erst ab
2023-08-29 (ICE-Fenster auf fredgraph.csv).

## Point-in-time Regeln

| Quelle | Verzug | available_date |
|---|---|---|
| FRED DFII10, T10YIE, DTWEXBGS, VIXCLS, BAMLH0A0HYM2 | ~1 Kalendertag | observation_date + 1 Tag |
| SPDR GLD Holdings | keiner | observation_date |
| CFTC COT | Freitag für Dienstagsstand | report_date (Di) + 3 Tage |

Merge: `pandas.merge_asof(..., direction="backward")` auf `available_date <= date`.
`*_obs_date` ist nie nach `date` (Assert im Build-Skript).

Beispiel letzter COT-Stand im Panel: Report **2026-08-18**
(verfügbar ab 2026-08-21). Der Report vom 2026-08-25 erscheint erst
am Freitag 2026-08-28 und darf am 2026-08-26 noch nicht verwendet werden.

DTWEXBGS letzter FRED-Stand im Abruf: 2026-08-21.

## Quellen

- FRED: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIE`
- GLD Holdings: `https://api.spdrgoldshares.com/api/v1/historical-archive?product=gld&exchange=NYSE&lang=en`
  (die alte URL `.../GLD_US_archive_EN.csv` leitet auf eine Barlist-PDF um)
- CFTC via PyPI `cot_reports` (`disaggregated_fut`), Contract-Code `088691`
- XAUUSD: `data/xauusd_h1/XAUUSD_H1_YYYY_MM.csv` (Unix-ms, UTC)

## Rebuild

```bash
pip install -r macro_gold/requirements.txt
XAUUSD_H1_DIR=data/xauusd_h1 python macro_gold/build_macro_gold.py
```

Das Skript erwartet die FRED-CSVs und die GLD-XLSX unter `macro_gold/raw/`
(oder `gld_holdings_tonnes.csv` als Fallback). COT wird von cftc.gov geholt
und als `raw/cftc_gold_managed_money_net.csv` gecacht.
