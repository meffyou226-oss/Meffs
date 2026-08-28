#!/usr/bin/env bash
# Baut die Jahresdateien aus den Quartals-CSVs.
# Aufruf im Ordner data/makro/:  bash combine_daily.sh
set -euo pipefail
cd "$(dirname "$0")"

combine_year() {
  local year="$1"
  local out="xauusd_macro_daily_${year}.csv"
  local first=1
  shopt -s nullglob
  local parts=(xauusd_macro_daily_${year}_q*.csv)
  if [ ${#parts[@]} -eq 0 ]; then
    echo "skip $year (keine Quartalsdateien)"
    return
  fi
  : > "$out"
  for p in "${parts[@]}"; do
    if [ "$first" -eq 1 ]; then
      cat "$p" >> "$out"
      first=0
    else
      tail -n +2 "$p" >> "$out"
    fi
  done
  echo "wrote $out ($(wc -l < "$out") Zeilen inkl. Header)"
}

combine_year 2022
combine_year 2023
combine_year 2024
combine_year 2025
combine_year 2026

# explizit H1 2026 (Q1+Q2) fuer den Walk-Forward-Backtest
if [ -f xauusd_macro_daily_2026_q1.csv ] && [ -f xauusd_macro_daily_2026_q2.csv ]; then
  head -1 xauusd_macro_daily_2026_q1.csv > xauusd_macro_daily_2026_h1.csv
  tail -n +2 xauusd_macro_daily_2026_q1.csv >> xauusd_macro_daily_2026_h1.csv
  tail -n +2 xauusd_macro_daily_2026_q2.csv >> xauusd_macro_daily_2026_h1.csv
  echo "wrote xauusd_macro_daily_2026_h1.csv ($(wc -l < xauusd_macro_daily_2026_h1.csv) Zeilen inkl. Header)"
fi
