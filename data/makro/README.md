# XAUUSD Makro-Daten (komplette Historie 2022-08 bis 2026-08)

Stand Abruf: 2026-08-28.

Quellen: FRED (St. Louis Fed) + Yahoo Finance (COMEX GC=F, ICE DXY, SI=F, GLD, GDX).

## Dateien

| Datei | Inhalt | Zeitraum |
|---|---|---|
| `xauusd_macro_daily_2022.csv` | Tagesdaten Gold/DXY/VIX/Treasuries/TIPS/USD/Risiko | 2022-08-01 – 2022-12-30 |
| `xauusd_macro_daily_2023.csv` | dito | 2023 |
| `xauusd_macro_daily_2024.csv` | dito | 2024 |
| `xauusd_macro_daily_2025.csv` | dito | 2025 |
| `xauusd_macro_daily_2026.csv` | dito | 2026-01-01 – 2026-08-28 |
| `xauusd_macro_weekly.csv` | Initial Claims + Fed-Bilanz WALCL | 2022-08 – 2026-08 |
| `xauusd_macro_weekly_claims.csv` | nur Initial Claims (ICSA) | wochenweise |
| `xauusd_macro_weekly_walcl.csv` | nur Fed-Bilanz (WALCL, Mio USD) | wochenweise |
| `xauusd_macro_monthly.csv` | CPI/PCE YoY, NFP, UNRATE, M2 + Monatsschluss | Aug 2022 – Aug 2026 |
| `xauusd_macro_dashboard.csv` | letzter Wert je Serie | Snapshot 2026-08-28 |
| `xauusd_macro_katalog.csv` | Serien-IDs und Quellenlinks | — |

Daily-Spalten: Date, XAUUSD, DXY, XAGUSD, Gold_Silver_Ratio, GLD, GDX, VIX, GVZ_GoldVol, DGS1/2/5/10/30, DFII5/10, T10YIE, T5YIE, T10Y2Y, T10Y3M, DFF, FF_Upper, SOFR, USD_Broad_TW, USD_AFE_TW, EURUSD, USDCNY, USDJPY, SP500, WTI_FRED, XAU_Miners, EPU_US, HY_OAS, RRP.

Gold-Proxy ist COMEX Front-Month (`GC=F`), nicht LBMA-Fix.
Leere Zellen = kein Handels-/Release-Tag fuer diese Serie.
