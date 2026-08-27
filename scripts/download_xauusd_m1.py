#!/usr/bin/env python3
"""Download Dukascopy XAUUSD M1 BID candles as monthly CSVs.

Format matches data/xauusd_m5: timestamp,open,high,low,close (Unix ms).
"""
from __future__ import annotations

import argparse
import calendar
import lzma
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

POINT = 1000.0
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Referer": "https://www.dukascopy.com/",
}


def month_days(year: int, month: int):
    last = calendar.monthrange(year, month)[1]
    today = datetime.now(timezone.utc).date()
    for d in range(1, last + 1):
        dt = datetime(year, month, d, tzinfo=timezone.utc).date()
        if dt > today:
            break
        yield d


def fetch_day(session: requests.Session, year: int, month: int, day: int, retries: int = 6):
    url = (
        f"https://datafeed.dukascopy.com/datafeed/XAUUSD/"
        f"{year}/{month-1:02d}/{day:02d}/BID_candles_min_1.bi5"
    )
    delay = 1.0
    for _ in range(retries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 404:
                return day, []
            if r.status_code != 200 or not r.content:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raw = lzma.decompress(r.content)
            base = datetime(year, month, day, tzinfo=timezone.utc)
            rows = []
            for i in range(len(raw) // 24):
                t, o, c, lo, h, _v = struct.unpack(">5if", raw[i * 24 : (i + 1) * 24])
                ts = int((base + timedelta(seconds=t)).timestamp() * 1000)
                rows.append((ts, o / POINT, h / POINT, lo / POINT, c / POINT))
            return day, rows
        except Exception:
            time.sleep(delay)
            delay = min(delay * 2, 30)
    return day, None


def download_month(out_dir: Path, session: requests.Session, year: int, month: int) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"XAUUSD_M1_{year}_{month:02d}.csv"
    days = list(month_days(year, month))
    if not days:
        return None
    if out.exists() and out.stat().st_size > 1000:
        print(f"skip {out.name}")
        return out
    all_rows = []
    failed = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fetch_day, session, year, month, d): d for d in days}
        for fut in as_completed(futs):
            day, rows = fut.result()
            if rows is None:
                failed.append(day)
            elif rows:
                all_rows.extend(rows)
    for d in failed:
        day, rows = fetch_day(session, year, month, d, retries=10)
        if rows:
            all_rows.extend(rows)
        else:
            print(f"missing {year}-{month:02d}-{d:02d}")
    all_rows.sort(key=lambda r: r[0])
    with out.open("w", encoding="utf-8") as f:
        f.write("timestamp,open,high,low,close\n")
        for ts, o, h, lo, c in all_rows:
            f.write(f"{ts},{o:.3f},{h:.3f},{lo:.3f},{c:.3f}\n")
    print(f"wrote {out.name} rows={len(all_rows)} bytes={out.stat().st_size}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/xauusd_m1")
    p.add_argument("--start", default="2022-01", help="YYYY-MM")
    p.add_argument("--end", default=None, help="YYYY-MM inclusive (default: current month UTC)")
    args = p.parse_args()
    sy, sm = map(int, args.start.split("-"))
    now = datetime.now(timezone.utc)
    if args.end:
        ey, em = map(int, args.end.split("-"))
    else:
        ey, em = now.year, now.month
    session = requests.Session()
    session.headers.update(HEADERS)
    out_dir = Path(args.out)
    y, m = sy, sm
    while (y, m) <= (ey, em):
        print(f"=== {y}-{m:02d} ===")
        download_month(out_dir, session, y, m)
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
        time.sleep(0.2)


if __name__ == "__main__":
    main()
