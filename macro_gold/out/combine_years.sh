#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
out=xauusd_macro_gold_pit_daily.csv
first=1
: > "$out"
shopt -s nullglob
for f in xauusd_macro_gold_pit_daily_????_q?.csv; do
  if [ "$first" -eq 1 ]; then cat "$f" >> "$out"; first=0; else tail -n +2 "$f" >> "$out"; fi
done
echo "wrote $out ($(wc -l < "$out") lines)"
