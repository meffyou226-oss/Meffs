#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-data/dxy_m1}"
START="${2:-2022-01}"
END="${3:-2026-12}"
mkdir -p "$OUT_DIR"
y=${START%-*}; m=$((10#${START#*-}))
ey=${END%-*}; em=$((10#${END#*-}))
while [ "$y" -lt "$ey" ] || { [ "$y" -eq "$ey" ] && [ "$m" -le "$em" ]; }; do
  from=$(printf '%04d-%02d-01' "$y" "$m")
  ny=$y; nm=$((m + 1)); if [ "$nm" -eq 13 ]; then nm=1; ny=$((y + 1)); fi
  to=$(printf '%04d-%02d-01' "$ny" "$nm")
  dest=$(printf '%s/DXY_M1_%04d_%02d.csv' "$OUT_DIR" "$y" "$m")
  if [ -f "$dest" ] && [ "$(wc -c < "$dest")" -gt 1000 ]; then
    echo "skip $(basename "$dest")"
  else
    echo "download $from -> $to"
    tmp=$(mktemp -d)
    npx --yes dukascopy-node -i dollaridxusd -from "$from" -to "$to" -t m1 -f csv -dir "$tmp" -fl
    src=$(ls "$tmp"/*.csv | head -1)
    mv "$src" "$dest"
    echo "wrote $(basename "$dest") rows=$(wc -l < "$dest")"
    rm -rf "$tmp"
  fi
  m=$((m + 1)); if [ "$m" -eq 13 ]; then m=1; y=$((y + 1)); fi
done
