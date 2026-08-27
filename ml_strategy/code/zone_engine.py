#!/usr/bin/env python3
"""Look-ahead-free XAU demand/supply zone engine (Pine 'XAU Zone Pro' port).

Bias rules enforced here:
- A swing pivot is only emitted on the confirmation bar (index = pivot + swing_len).
- Zone geometry uses only the confirmed pivot candle.
- Features / pending levels use only data available at the current closed bar.
- Trade exits start on the bar AFTER entry (same as the Pine script).
- Same-bar SL vs TP conflict is resolved against the trader (SL first).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from zone_indicators import ema, rsi, wilder_atr


@dataclass
class ZoneConfig:
    swing_len: int = 5
    impulse_atr: float = 0.6
    rr1: float = 1.0
    rr2: float = 2.0
    sl_buffer_atr: float = 0.15
    entry_frac: float = 0.25
    max_active_zones: int = 40
    max_zone_age: int = 80
    one_trade_at_a_time: bool = True


@dataclass
class Zone:
    kind: str
    born_idx: int
    pivot_idx: int
    top: float
    bot: float
    impulse: float
    atr: float
    active: bool = True
    first_touch_idx: Optional[int] = None


@dataclass
class Trade:
    entry_idx: int
    exit_idx: Optional[int]
    is_long: bool
    entry: float
    sl: float
    tp1: float
    tp2: float
    status: int
    tp2_hit: bool = False
    zone_kind: str = ""
    zone_age: int = 0
    zone_height: float = 0.0
    zone_impulse: float = 0.0
    zone_atr: float = 0.0
    first_touch: int = 1
    features: dict = field(default_factory=dict)

    @property
    def risk(self) -> float:
        return abs(self.entry - self.sl)


def _is_unique_extreme(arr, center, left, right, mode):
    sl = slice(center - left, center + right + 1)
    w = arr[sl]
    v = arr[center]
    if mode == "low":
        return v <= w.min() and int(np.sum(w == v)) == 1
    return v >= w.max() and int(np.sum(w == v)) == 1


def _levels(zone, atr_now, cfg):
    height = zone.top - zone.bot
    if zone.kind == "demand":
        entry = zone.top - height * cfg.entry_frac
        sl = zone.bot - atr_now * cfg.sl_buffer_atr
        risk = entry - sl
        if risk <= 0:
            return None
        return entry, sl, entry + risk * cfg.rr1, entry + risk * cfg.rr2, True
    entry = zone.bot + height * cfg.entry_frac
    sl = zone.top + atr_now * cfg.sl_buffer_atr
    risk = sl - entry
    if risk <= 0:
        return None
    return entry, sl, entry - risk * cfg.rr1, entry - risk * cfg.rr2, False


def _pick_zone(demand, supply, close):
    d = next((z for z in reversed(demand) if z.active), None)
    s = next((z for z in reversed(supply) if z.active), None)
    if d and s:
        return d if abs(close - d.top) <= abs(close - s.bot) else s
    return d or s


def _prune(zones, cfg):
    actives = [z for z in zones if z.active]
    if len(actives) > cfg.max_active_zones:
        for z in actives[: len(actives) - cfg.max_active_zones]:
            z.active = False
    return zones


def run_zones(df, cfg=None):
    cfg = cfg or ZoneConfig()
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    n = len(df)
    slen = cfg.swing_len
    atr = wilder_atr(h, l, c, 14)
    ema21 = ema(c, 21)
    ema50 = ema(c, 50)
    rsi14 = rsi(c, 14)
    demand, supply, trades = [], [], []
    open_trade = None
    trade_taken_this_zone = False
    pending = None
    for i in range(n):
        if i >= slen * 2:
            p = i - slen
            atr_i = atr[i]
            if np.isfinite(atr_i) and atr_i > 0:
                if _is_unique_extreme(l, p, slen, slen, "low"):
                    impulse = h[p + 1] - l[p - 1]
                    if impulse > atr_i * cfg.impulse_atr:
                        demand.append(Zone("demand", i, p, max(h[p], c[p]), min(l[p], o[p]), impulse, atr_i))
                        trade_taken_this_zone = False
                        pending = None
                if _is_unique_extreme(h, p, slen, slen, "high"):
                    impulse = h[p - 1] - l[p + 1]
                    if impulse > atr_i * cfg.impulse_atr:
                        supply.append(Zone("supply", i, p, max(h[p], o[p]), min(l[p], c[p]), impulse, atr_i))
                        trade_taken_this_zone = False
                        pending = None
        for z in demand:
            if z.active and (c[i] < z.bot or (i - z.born_idx) > cfg.max_zone_age):
                z.active = False
        for z in supply:
            if z.active and (c[i] > z.top or (i - z.born_idx) > cfg.max_zone_age):
                z.active = False
        demand = _prune(demand, cfg)
        supply = _prune(supply, cfg)
        atr_i = atr[i]
        zone = _pick_zone(demand, supply, c[i]) if np.isfinite(atr_i) and atr_i > 0 else None
        pending = None
        if zone is not None:
            lv = _levels(zone, atr_i, cfg)
            if lv is not None:
                pending = (*lv, zone)
        if open_trade is not None and i > open_trade.entry_idx:
            t = open_trade
            if t.is_long:
                if l[i] <= t.sl:
                    t.status, t.exit_idx, open_trade = 2, i, None
                elif h[i] >= t.tp1:
                    t.status, t.exit_idx = 1, i
                    t.tp2_hit = h[i] >= t.tp2
                    open_trade = None
            else:
                if h[i] >= t.sl:
                    t.status, t.exit_idx, open_trade = 2, i, None
                elif l[i] <= t.tp1:
                    t.status, t.exit_idx = 1, i
                    t.tp2_hit = l[i] <= t.tp2
                    open_trade = None
        if pending is not None and open_trade is None and not trade_taken_this_zone and np.isfinite(atr_i):
            entry, sl, tp1, tp2, is_long, zone = pending
            if (l[i] <= entry <= h[i]) and ((c[i] > o[i]) if is_long else (c[i] < o[i])):
                if zone.first_touch_idx is None:
                    zone.first_touch_idx = i
                    first_touch = 1
                else:
                    first_touch = 0
                age = i - zone.born_idx
                height = zone.top - zone.bot
                ema_spread = ((ema21[i] - ema50[i]) / atr_i) if np.isfinite(ema21[i]) and np.isfinite(ema50[i]) else 0.0
                ts = df["timestamp"].iloc[i]
                feats = {
                    "is_long": int(is_long),
                    "zone_height_atr": height / atr_i,
                    "zone_impulse_atr": zone.impulse / atr_i,
                    "zone_age": age,
                    "first_touch": first_touch,
                    "atr": atr_i,
                    "rsi14": rsi14[i] if np.isfinite(rsi14[i]) else 50.0,
                    "ema_spread_atr": ema_spread,
                    "dist_entry_atr": (c[i] - entry) / atr_i,
                    "close_in_zone": (c[i] - zone.bot) / height if height > 0 else 0.5,
                    "hour": int(getattr(ts, "hour", 0)),
                    "dow": int(getattr(ts, "dayofweek", 0)),
                    "body_atr": abs(c[i] - o[i]) / atr_i,
                    "wick_lower_atr": (min(o[i], c[i]) - l[i]) / atr_i,
                    "wick_upper_atr": (h[i] - max(o[i], c[i])) / atr_i,
                    "n_active_demand": sum(1 for z in demand if z.active),
                    "n_active_supply": sum(1 for z in supply if z.active),
                    "trend_with_zone": int((is_long and ema_spread > 0) or ((not is_long) and ema_spread < 0)),
                }
                t = Trade(i, None, is_long, float(entry), float(sl), float(tp1), float(tp2), 0, False, zone.kind, age, height, zone.impulse, atr_i, first_touch, feats)
                trades.append(t)
                open_trade = t
                trade_taken_this_zone = True
        if open_trade is None and trades:
            last = trades[-1]
            if last.status == 1 and not last.tp2_hit and last.exit_idx is not None and i > last.exit_idx:
                if last.is_long and h[i] >= last.tp2:
                    last.tp2_hit = True
                elif (not last.is_long) and l[i] <= last.tp2:
                    last.tp2_hit = True
    return trades, _summarize(trades, cfg)


def _summarize(trades, cfg):
    closed = [t for t in trades if t.status in (1, 2)]
    wins = [t for t in closed if t.status == 1]
    losses = [t for t in closed if t.status == 2]
    gross_win = sum(abs(t.tp2 - t.entry) if t.tp2_hit else abs(t.tp1 - t.entry) for t in wins)
    gross_loss = sum(t.risk for t in losses)
    n_closed = len(closed)
    wr = (len(wins) / n_closed * 100.0) if n_closed else float("nan")
    pf = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else float("nan"))
    return {
        "n_trades": len(trades), "n_closed": n_closed,
        "n_open": sum(1 for t in trades if t.status == 0),
        "wins": len(wins), "losses": len(losses),
        "tp2": sum(1 for t in wins if t.tp2_hit),
        "winrate": wr, "profit_factor": pf,
        "avg_win": (gross_win / len(wins)) if wins else float("nan"),
        "avg_loss": (gross_loss / len(losses)) if losses else float("nan"),
        "rr1": cfg.rr1, "rr2": cfg.rr2,
    }


def trades_to_frame(trades, df):
    rows = []
    for t in trades:
        row = dict(t.features)
        r = -1.0 if t.status == 2 else ((abs(t.tp2 - t.entry) / t.risk if t.tp2_hit else abs(t.tp1 - t.entry) / t.risk) if t.status == 1 and t.risk else 0.0)
        row.update({
            "entry_idx": t.entry_idx, "exit_idx": t.exit_idx,
            "entry_time": df["timestamp"].iloc[t.entry_idx],
            "exit_time": df["timestamp"].iloc[t.exit_idx] if t.exit_idx is not None else pd.NaT,
            "is_long": int(t.is_long), "entry": t.entry, "sl": t.sl, "tp1": t.tp1, "tp2": t.tp2,
            "status": t.status, "tp2_hit": int(t.tp2_hit), "label_win": int(t.status == 1),
            "r_multiple": r, "zone_kind": t.zone_kind,
        })
        rows.append(row)
    return pd.DataFrame(rows)
