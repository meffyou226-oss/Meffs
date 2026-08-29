#!/usr/bin/env python3
"""Build a point-in-time XAUUSD + gold-macro daily panel.

No lookahead:
  FRED daily series  -> observation_date + 1 calendar day (publication lag)
  GLD holdings       -> same-day (source has no lag)
  CFTC COT MM net    -> report Tuesday + 3 calendar days (Friday release)

Base calendar: Dukascopy XAUUSD H1 aggregated to UTC trading days with
actual range (placeholder flat weekend/holiday bars dropped).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "out"
H1_DIR = Path(os.environ.get("XAUUSD_H1_DIR", "/tmp/meffs_h1"))

FRED_LAG_DAYS = 1
COT_LAG_DAYS = 3
CFTC_CODE = "088691"


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace(".", pd.NA), errors="coerce")


def load_fred(path: Path, col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    val_col = col if col in df.columns else df.columns[1]
    out = pd.DataFrame(
        {
            "obs_date": pd.to_datetime(df[date_col], errors="coerce"),
            col: _to_num(df[val_col]),
        }
    ).dropna(subset=["obs_date"])
    out = out.dropna(subset=[col])
    out["available_date"] = out["obs_date"] + pd.Timedelta(days=FRED_LAG_DAYS)
    return out.sort_values("available_date")


def load_gld(path: Path) -> pd.DataFrame:
    csv_fallback = RAW / "gld_holdings_tonnes.csv"
    if not path.exists() and csv_fallback.exists():
        out = pd.read_csv(csv_fallback, parse_dates=["obs_date"])
        if "available_date" not in out.columns:
            out["available_date"] = out["obs_date"]
        else:
            out["available_date"] = pd.to_datetime(out["available_date"])
        return out.sort_values("available_date")
    raw = pd.read_excel(path, sheet_name="US GLD Historical Archive", header=0)
    raw.columns = [str(c).strip() for c in raw.columns]
    date_col = [c for c in raw.columns if c.lower() == "date"][0]
    tonnes_col = next(c for c in raw.columns if "tonne" in c.lower())
    oz_col = next(
        c
        for c in raw.columns
        if "total ounces" in c.lower() or ("ounces of gold in the trust" in c.lower())
    )
    out = pd.DataFrame(
        {
            "obs_date": pd.to_datetime(raw[date_col], errors="coerce", dayfirst=True),
            "gld_holdings_tonnes": _to_num(raw[tonnes_col]),
            "gld_holdings_oz": _to_num(raw[oz_col]),
        }
    )
    out = out.dropna(subset=["obs_date", "gld_holdings_tonnes"])
    out["available_date"] = out["obs_date"]
    return out.sort_values("available_date")


def load_xauusd_daily(h1_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(h1_dir / "XAUUSD_H1_*.csv")))
    if not files:
        raise FileNotFoundError(f"No H1 files in {h1_dir}")
    parts = [pd.read_csv(f) for f in files]
    h1 = pd.concat(parts, ignore_index=True)
    h1["ts"] = pd.to_datetime(h1["timestamp"], unit="ms", utc=True)
    h1["date"] = h1["ts"].dt.tz_convert("UTC").dt.normalize()
    rng = (h1["high"] - h1["low"]).abs()
    live = h1.loc[rng > 1e-8].copy()
    g = live.groupby("date", sort=True)
    daily = pd.DataFrame(
        {
            "date": g["date"].first().dt.tz_localize(None),
            "xauusd_open": g["open"].first().values,
            "xauusd_high": g["high"].max().values,
            "xauusd_low": g["low"].min().values,
            "xauusd_close": g["close"].last().values,
            "h1_bars": g["close"].size().values,
        }
    ).reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily[daily["h1_bars"] >= 4].copy()
    return daily.sort_values("date")


def load_cot() -> pd.DataFrame:
    import tempfile
    import cot_reports as cot

    cached = RAW / "cftc_gold_managed_money_net.csv"
    if cached.exists():
        out = pd.read_csv(cached, parse_dates=["obs_date", "available_date"])
        return out.sort_values("available_date")

    years = list(range(2021, 2027))
    frames = []
    cwd = Path.cwd()
    for y in years:
        try:
            with tempfile.TemporaryDirectory() as td:
                os.chdir(td)
                df = cot.cot_year(
                    year=y,
                    cot_report_type="disaggregated_fut",
                    store_txt=True,
                    verbose=False,
                )
            frames.append(df)
            print(f"COT {y}: {len(df)} rows")
        except Exception as e:
            print(f"COT {y} failed: {e}")
        finally:
            os.chdir(cwd)
    if not frames:
        raise RuntimeError("No COT data downloaded")
    cot_df = pd.concat(frames, ignore_index=True)
    cols = {c.lower().strip(): c for c in cot_df.columns}
    code_col = cols.get("cftc_contract_market_code") or cols.get("cftc_contract_market_code_quotes")
    name_col = cols.get("market_and_exchange_names") or cols.get("market_and_exchange_name")
    date_col = None
    for key in ("report_date_as_yyyy_mm_dd", "report_date_as_yyyy-mm-dd", "as_of_date_in_form_yymmdd"):
        if key in cols:
            date_col = cols[key]
            break
    if date_col is None:
        for c in cot_df.columns:
            if "date" in c.lower() and "report" in c.lower():
                date_col = c
                break
    mm_long = mm_short = None
    for c in cot_df.columns:
        cl = c.lower()
        if "m_money" in cl and "positions_long_all" in cl and "old" not in cl and "other" not in cl:
            mm_long = c
        if "m_money" in cl and "positions_short_all" in cl and "old" not in cl and "other" not in cl:
            mm_short = c
    gold = cot_df.copy()
    if code_col is not None:
        code = gold[code_col].astype(str).str.replace(r"\D", "", regex=True)
        gold = gold[code.str.endswith("088691") | (code == CFTC_CODE)]
    elif name_col is not None:
        gold = gold[gold[name_col].astype(str).str.contains("GOLD", case=False)
                    & gold[name_col].astype(str).str.contains("COMMODITY EXCHANGE", case=False)]
        gold = gold[~gold[name_col].astype(str).str.contains("MICRO", case=False)]
    else:
        raise RuntimeError(f"Cannot locate gold contract. Columns: {list(cot_df.columns)}")
    out = pd.DataFrame(
        {
            "obs_date": pd.to_datetime(gold[date_col], errors="coerce"),
            "cot_mm_long": _to_num(gold[mm_long]),
            "cot_mm_short": _to_num(gold[mm_short]),
        }
    ).dropna(subset=["obs_date"])
    out["cot_mm_net"] = out["cot_mm_long"] - out["cot_mm_short"]
    out["available_date"] = out["obs_date"] + pd.Timedelta(days=COT_LAG_DAYS)
    out = out.drop_duplicates("obs_date").sort_values("available_date")
    out.to_csv(cached, index=False)
    print("COT gold weeks:", len(out), out["obs_date"].min(), out["obs_date"].max())
    return out


def _naive_day(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.dt.normalize().astype("datetime64[ns]")


def asof_attach(base: pd.DataFrame, feat: pd.DataFrame, value_cols: list[str], prefix_obs: str) -> pd.DataFrame:
    left = pd.DataFrame({"available_date": _naive_day(base["date"])}).sort_values("available_date")
    right = feat.copy()
    right["available_date"] = _naive_day(right["available_date"])
    right["obs_date"] = _naive_day(right["obs_date"])
    right = right.sort_values("available_date")[["available_date", "obs_date"] + value_cols]
    merged = pd.merge_asof(left, right, on="available_date", direction="backward")
    return merged.rename(columns={"obs_date": f"{prefix_obs}_obs_date", "available_date": "date"})


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading XAUUSD H1 → daily …")
    xau = load_xauusd_daily(H1_DIR)
    print("XAUUSD days:", len(xau), xau["date"].min().date(), xau["date"].max().date())

    print("Loading FRED …")
    fred_map = {
        "DFII10": "real_yield_10y",
        "T10YIE": "be_inflation_10y",
        "DTWEXBGS": "usd_broad_tw",
        "VIXCLS": "vix",
        "BAMLH0A0HYM2": "hy_oas",
    }
    fred_frames = {}
    for sid, name in fred_map.items():
        src = RAW / f"{sid}.csv"
        f = load_fred(src, sid).rename(columns={sid: name})
        fred_frames[name] = f
        print(f"  {sid}: {len(f)} {f['obs_date'].min().date()} → {f['obs_date'].max().date()}")

    print("Loading GLD holdings …")
    gld = load_gld(RAW / "GLD_historical_archive.xlsx")
    print("  GLD tonnes:", len(gld), gld["obs_date"].min().date(), gld["obs_date"].max().date())

    print("Loading CFTC COT via cot_reports …")
    cot = load_cot()

    panel = xau.copy()
    panel["date"] = _naive_day(panel["date"])
    attaches = [
        (fred_frames["real_yield_10y"], ["real_yield_10y"], "real_yield_10y"),
        (fred_frames["be_inflation_10y"], ["be_inflation_10y"], "be_inflation_10y"),
        (fred_frames["usd_broad_tw"], ["usd_broad_tw"], "usd_broad_tw"),
        (fred_frames["vix"], ["vix"], "vix"),
        (fred_frames["hy_oas"], ["hy_oas"], "hy_oas"),
        (gld, [c for c in gld.columns if c.startswith("gld_")], "gld"),
        (cot, ["cot_mm_net", "cot_mm_long", "cot_mm_short"], "cot"),
    ]
    for feat, cols, prefix in attaches:
        extra = asof_attach(panel, feat, cols, prefix)
        panel = panel.merge(extra, on="date", how="left")

    for c in panel.columns:
        if c.endswith("_obs_date"):
            bad = panel[c].notna() & (panel[c] > panel["date"])
            if bad.any():
                raise AssertionError(f"Lookahead leak in {c}: {int(bad.sum())} rows")

    panel["date"] = panel["date"].dt.strftime("%Y-%m-%d")
    for c in panel.columns:
        if c.endswith("_obs_date"):
            panel[c] = pd.to_datetime(panel[c]).dt.strftime("%Y-%m-%d")

    out_path = OUT / "xauusd_macro_gold_pit_daily.csv"
    panel.to_csv(out_path, index=False)
    print("Wrote", out_path, "rows", len(panel), "cols", len(panel.columns))

    cov = []
    for c in ["real_yield_10y", "be_inflation_10y", "usd_broad_tw", "vix", "hy_oas", "gld_holdings_tonnes", "cot_mm_net"]:
        s = panel[c]
        cov.append({
            "column": c,
            "non_null": int(s.notna().sum()),
            "null": int(s.isna().sum()),
            "first_non_null": panel.loc[s.notna(), "date"].iloc[0] if s.notna().any() else "",
            "last_non_null": panel.loc[s.notna(), "date"].iloc[-1] if s.notna().any() else "",
            "last_value": s.dropna().iloc[-1] if s.notna().any() else "",
        })
    pd.DataFrame(cov).to_csv(OUT / "coverage.csv", index=False)


if __name__ == "__main__":
    main()
