#!/usr/bin/env python3
"""Download Dukascopy XAUUSD M1 BID candles as monthly CSVs using stdlib only."""
from __future__ import annotations

import calendar
import lzma
import random
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

OUT_DIR = Path("/workspace/415d80d1-e105-410b-896d-38f27e4c9898/sessions/agent_50742d8d-8d25-4388-882c-5c27593ef8ee/data/xauusd_m1")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Encoding": "identity",
    "Referer": "https://www.dukascopy.com/",
    "Connection": "keep-alive",
}
POINT = 1000.0


def month_days(year: int, month: int):
    last = calendar.monthrange(year, month)[1]
    today = datetime.now(timezone.utc).date()
    for d in range(1, last + 1):
        dt = datetime(year, month, d, tzinfo=timezone.utc).date()
        if dt > today:
            break
        yield d


def fetch_day(year: int, month: int, day: int, retries: int = 8):
    url = (
        f"https://datafeed.dukascopy.com/datafeed/XAUUSD/"
        f"{year}/{month-1:02d}/{day:02d}/BID_candles_min_1.bi5"
    )
    delay = 0.25
    for _ in range(retries):
        try:
            req = Request(url, headers=HEADERS)
            with urlopen(req, timeout=20) as r:
                if r.status == 404:
                    return day, []
                if r.status == 429:
                    time.sleep(min(delay + random.random(), 8))
                    delay = min(delay * 1.6, 8)
                    continue
                data = r.read()
            if not data:
                time.sleep(delay)
                delay = min(delay * 1.6, 8)
                continue
            raw = lzma.decompress(data)
            base = datetime(year, month, day, tzinfo=timezone.utc)
            rows = []
            for i in range(len(raw) // 24):
                t, o, c, lo, h, _v = struct.unpack(">5if", raw[i * 24 : (i + 1) * 24])
                ts = int((base + timedelta(seconds=t)).timestamp() * 1000)
                rows.append((ts, o / POINT, h / POINT, lo / POINT, c / POINT))
            return day, rows
        except HTTPError as e:
            if e.code == 404:
                return day, []
            time.sleep(delay + random.random() * 0.2)
            delay = min(delay * 1.6, 8)
        except Exception:
            time.sleep(delay + random.random() * 0.2)
            delay = min(delay * 1.6, 8)
    return day, None


def download_month(out_dir: Path, year: int, month: int, day_workers: int) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"XAUUSD_M1_{year}_{month:02d}.csv"
    days = list(month_days(year, month))
    if not days:
        return None
    if out.exists() and out.stat().st_size > 1000:
        print(f"skip {out.name}", flush=True)
        return out
    t0 = time.time()
    all_rows = []
    failed = []
    with ThreadPoolExecutor(max_workers=day_workers) as ex:
        futs = [ex.submit(fetch_day, year, month, d) for d in days]
        for fut in as_completed(futs):
            day, rows = fut.result()
            if rows is None:
                failed.append(day)
            elif rows:
                all_rows.extend(rows)
    for d in failed:
        day, rows = fetch_day(year, month, d, retries=12)
        if rows:
            all_rows.extend(rows)
        else:
            print(f"missing {year}-{month:02d}-{d:02d}", flush=True)
    all_rows.sort(key=lambda r: r[0])
    with out.open("w", encoding="utf-8") as f:
        f.write("timestamp,open,high,low,close\n")
        for ts, o, h, lo, c in all_rows:
            f.write(f"{ts},{o:.3f},{h:.3f},{lo:.3f},{c:.3f}\n")
    print(f"wrote {out.name} rows={len(all_rows)} sec={time.time()-t0:.1f}", flush=True)
    return out


def month_range(start: str, end: str):
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    y, m = sy, sm
    out = []
    while (y, m) <= (ey, em):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def main():
    start = "2023-05"
    end = "2026-08"
    month_jobs = 4
    day_workers = 6
    months = month_range(start, end)
    print(f"months={len(months)} month_jobs={month_jobs}", flush=True)
    if month_jobs == 1:
        for y, m in months:
            download_month(OUT_DIR, y, m, day_workers)
        return
    with ThreadPoolExecutor(max_workers=month_jobs) as ex:
        futs = {
            ex.submit(download_month, OUT_DIR, y, m, day_workers): (y, m)
            for y, m in months
        }
        for fut in as_completed(futs):
            y, m = futs[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"FAIL {y}-{m:02d}: {e}", flush=True)


if __name__ == "__main__":
    main()
