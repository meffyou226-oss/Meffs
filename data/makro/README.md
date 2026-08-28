# XAUUSD Makro-Daten (komplette Historie 2022-08 bis 2026-08)

Stand Abruf: 2026-08-28.

Quellen: FRED (St. Louis Fed) + Yahoo Finance (COMEX GC=F, ICE DXY, SI=F, GLD, GDX).

## Daily – so liegen die Dateien im Repo

Die komplette Daily-Historie ist als Quartalsdateien committed (volle 34 Spalten).
Jahresdateien werden lokal mit `bash combine_daily.sh` zusammengesetzt.

| Datei | Zeitraum | Zeilen (ohne Header) |
|---|---|---|
| `xauusd_macro_daily_2022_q3.csv` | 2022-08-01 – 2022-09-30 | 45 |
| `xauusd_macro_daily_2022_q4.csv` | 2022-10-03 – 2022-12-30 | 64 |
| `xauusd_macro_daily_2023_q1.csv` … `_q4.csv` | 2023 | 258 |
| `xauusd_macro_daily_2024_q1.csv` … `_q4.csv` | 2024 | 259 |
| `xauusd_macro_daily_2025_q1.csv` … `_q4.csv` | 2025 | 258 |
| `xauusd_macro_daily_2026_q1.csv` | 2026-01-02 – 2026-03-31 | 63 |
| `xauusd_macro_daily_2026_q2.csv` | 2026-04-01 – 2026-06-30 | 65 |
| `xauusd_macro_daily_2026_q3.csv` / `_h2.csv` | 2026-07-01 – 2026-08-28 | 43 |

Nach `bash combine_daily.sh` existieren lokal:

- `xauusd_macro_daily_2022.csv` … `_2026.csv`
- `xauusd_macro_daily_2026_h1.csv` (Jan–Jun 2026, 128 Tage)

Gesamt Daily: 2022-08-01 bis 2026-08-28 = 1.055 Handelstage, 34 Spalten.

## Weitere Dateien

| Datei | Inhalt |
|---|---|
| `xauusd_macro_weekly.csv` | Initial Claims + WALCL |
| `xauusd_macro_weekly_claims.csv` | nur ICSA |
| `xauusd_macro_weekly_walcl.csv` | nur Fed-Bilanz WALCL |
| `xauusd_macro_monthly.csv` | CPI/PCE YoY, NFP, UNRATE, M2 + Monatsschluss |
| `xauusd_macro_dashboard.csv` | letzter Wert je Serie |
| `xauusd_macro_katalog.csv` | Serien-IDs und Quellen |

Daily-Spalten: Date, XAUUSD, DXY, XAGUSD, Gold_Silver_Ratio, GLD, GDX, VIX, GVZ_GoldVol, DGS1/2/5/10/30, DFII5/10, T10YIE, T5YIE, T10Y2Y, T10Y3M, DFF_EffFF, FF_Upper, SOFR, USD_Broad_TW, USD_AFE_TW, EURUSD, USDCNY, USDJPY, SP500, WTI_FRED, XAU_Miners, EPU_US, HY_OAS, RRP.

Gold-Proxy ist COMEX Front-Month (`GC=F`), nicht LBMA-Fix.
Leere Zellen = kein Handels-/Release-Tag fuer diese Serie.
