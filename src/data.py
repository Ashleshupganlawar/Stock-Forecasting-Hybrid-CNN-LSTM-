from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
from pathlib import Path
import time

import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries


@dataclass(frozen=True)
class AVConfig:
    api_key: str
    symbol: str
    outputsize: str = "compact"
    adjusted_close_key: str = "5. adjusted close"
    close_key: str = "4. close"
    cache_dir: str = "cache"
    cache_ttl_seconds: int = 24 * 60 * 60  # 24 hours


class Normalizer:
    def __init__(self) -> None:
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        self.mu = x.mean(axis=0, keepdims=True)
        self.sd = x.std(axis=0, keepdims=True) + 1e-8
        return (x - self.mu) / self.sd

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mu is None or self.sd is None:
            raise ValueError("Call fit_transform() first.")
        x = np.asarray(x, dtype=np.float32)
        return (x * self.sd) + self.mu


def _cache_path(cfg: AVConfig) -> Path:
    d = Path(cfg.cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    # cache key includes symbol + outputsize
    return d / f"{cfg.symbol.upper()}_{cfg.outputsize}.csv"


def _read_cache(cfg: AVConfig) -> Optional[pd.Series]:
    p = _cache_path(cfg)
    if not p.exists():
        return None
    age = time.time() - p.stat().st_mtime
    if age > cfg.cache_ttl_seconds:
        return None
    df = pd.read_csv(p, parse_dates=["date"])
    df = df.sort_values("date")
    s = pd.Series(df["close"].values, index=pd.to_datetime(df["date"]))
    s.name = "close"
    return s


def _write_cache(cfg: AVConfig, close: pd.Series) -> None:
    p = _cache_path(cfg)
    df = pd.DataFrame({"date": close.index.astype("datetime64[ns]"), "close": close.values})
    df.to_csv(p, index=False)


def _download_from_stooq(symbol: str) -> pd.Series:
    # Stooq uses tickers like aapl.us for US stocks
    s = symbol.strip().lower()
    if "." not in s:
        s = f"{s}.us"
    url = f"https://stooq.com/q/d/l/?s={s}&i=d"
    df = pd.read_csv(url)
    if df.empty or "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("Stooq returned empty/invalid data. Try a US ticker like AAPL/MSFT/GOOG.")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    close = pd.Series(df["Close"].astype(float).values, index=df["Date"])
    close.name = "close"
    return close


def download_adjusted_close(cfg: AVConfig) -> Tuple[pd.Series, str]:
    """
    Order:
    1) local cache
    2) AlphaVantage (daily_adjusted -> daily)
    3) Stooq (free CSV)
    """
    cached = _read_cache(cfg)
    if cached is not None and len(cached) > 0:
        return cached, f"{cached.index.min().date()} -> {cached.index.max().date()} (cached)"

    # Try Alpha Vantage if key exists
    if cfg.api_key:
        ts = TimeSeries(key=cfg.api_key, output_format="pandas")

        # Try adjusted (may be premium)
        try:
            data, _ = ts.get_daily_adjusted(symbol=cfg.symbol, outputsize=cfg.outputsize)
            if cfg.adjusted_close_key in data.columns:
                data = data.sort_index()
                close = data[cfg.adjusted_close_key].astype(float)
                _write_cache(cfg, close)
                return close, f"{close.index.min().date()} -> {close.index.max().date()} (alpha adjusted)"
        except ValueError as e:
            # premium / rate limit / etc -> fallback
            pass

        # Try free daily close
        try:
            data, _ = ts.get_daily(symbol=cfg.symbol, outputsize=cfg.outputsize)
            if cfg.close_key in data.columns:
                data = data.sort_index()
                close = data[cfg.close_key].astype(float)
                _write_cache(cfg, close)
                return close, f"{close.index.min().date()} -> {close.index.max().date()} (alpha close)"
        except ValueError:
            # rate-limited -> fallback
            pass

    # Final fallback: Stooq
    close = _download_from_stooq(cfg.symbol)
    _write_cache(cfg, close)
    return close, f"{close.index.min().date()} -> {close.index.max().date()} (stooq)"


def make_windows(x: np.ndarray, window_size: int):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    if len(x) <= window_size:
        raise ValueError("Not enough data points for the chosen window_size")

    X, y = [], []
    for i in range(window_size, len(x)):
        X.append(x[i - window_size:i])
        y.append(x[i])

    X = np.array(X, dtype=np.float32)[:, :, None]
    y = np.array(y, dtype=np.float32)[:, None]
    last_window = x[-window_size:].astype(np.float32)[:, None]
    return X, y, last_window


def split_train_val(X: np.ndarray, y: np.ndarray, train_split: float = 0.7):
    n = len(X)
    split_idx = int(n * train_split)
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
