#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
out=xauusd_macro_gold_pit_daily.csv
first=1
: > "$out"
for y in 2022 2023 2024 2025 2026; do
  f=xauusd_macro_gold_pit_daily_${y}.csv
  [ -f "$f" ] || continue
  if [ "$first" -eq 1 ]; then cat "$f" >> "$out"; first=0; else tail -n +2 "$f" >> "$out"; fi
done
echo "wrote $out ($(wc -l < "$out") lines)"
