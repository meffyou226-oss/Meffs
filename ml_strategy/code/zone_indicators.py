#!/usr/bin/env python3
"""Look-ahead-safe indicators (only past bars)."""
from __future__ import annotations

import numpy as np


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    prev_c = close[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - prev_c), abs(low[i] - prev_c))
        prev_c = close[i]
    atr = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return atr
    atr[window - 1] = tr[:window].mean()
    for i in range(window, n):
        atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window
    return atr


def ema(arr: np.ndarray, span: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=np.float64)
    if len(arr) == 0:
        return out
    alpha = 2.0 / (span + 1.0)
    start = 0
    while start < len(arr) and not np.isfinite(arr[start]):
        start += 1
    if start >= len(arr):
        return out
    out[start] = arr[start]
    for i in range(start + 1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def rsi(close: np.ndarray, window: int = 14) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan)
    if n < window + 1:
        return out
    delta = np.diff(close, prepend=close[0])
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)
    avg_g = gain[1 : window + 1].mean()
    avg_l = loss[1 : window + 1].mean()
    out[window] = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    for i in range(window + 1, n):
        avg_g = (avg_g * (window - 1) + gain[i]) / window
        avg_l = (avg_l * (window - 1) + loss[i]) / window
        out[i] = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    return out
