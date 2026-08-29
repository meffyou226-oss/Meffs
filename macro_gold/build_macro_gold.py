#!/usr/bin/env python3
"""Build a point-in-time XAUUSD + gold-macro daily panel.

No lookahead:
  FRED daily series  -> observation_date + 1 calendar day (publication lag)
  GLD holdings       -> same-day (source has no lag)
  CFTC COT MM net    -> report Tuesday + 3 calendar days (Friday release)

Base calendar:
  2016-2021: COMEX GC=F daily (raw/gc_f_daily_2016_2021.csv) if present
  2022+:     Dukascopy XAUUSD H1 aggregated to UTC trading days
"""
