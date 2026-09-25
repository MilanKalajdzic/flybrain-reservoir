"""Prices -> causal features and forward-return targets.

Row t only uses information available at the close of day t. The target at row t is the return
from close t to close t+h, which is only known at t+h; readout.walk_forward makes sure a model
fitted at time t never trains on targets that weren't known yet.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = {
    "r1": "1-day log return",
    "r5": "5-day log return",
    "r20": "20-day log return (1-month momentum)",
    "r60": "60-day log return (3-month momentum)",
    "vol20": "20-day realized volatility",
    "vol_ratio": "log(5-day vol / 60-day vol): is vol spiking?",
    "dd252": "log distance from the 1-year high",
}
DEFAULT_FEATURES = ("r1", "r5", "r20", "vol20", "vol_ratio")


def download_prices(tickers, start: str = "2003-01-01", end: str | None = None, cache_dir: str | Path = "data/cache",
                    refresh: bool = False) -> pd.DataFrame:
    """Adjusted close prices from Yahoo Finance (cached as parquet). Columns = tickers.

    `end` is inclusive (yfinance's own `end` isn't, so it gets the day after). With end=None the data
    runs to the latest close and every rerun shifts slightly; the configs pin an end date instead.
    """
    tickers = [tickers] if isinstance(tickers, str) else list(tickers)
    cache = Path(cache_dir) / f"prices_{'_'.join(tickers)}_{start}_{end or 'latest'}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache).loc[start:end]
    import yfinance as yf

    day_after = None if end is None else (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(tickers, start=start, end=day_after, auto_adjust=True, progress=False, threads=False)
    close = close_from_yfinance(raw, tickers).loc[start:end]
    if close.empty:
        raise RuntimeError(f"yfinance returned no data for {tickers}. Retry later, or use market.source: csv")
    cache.parent.mkdir(parents=True, exist_ok=True)
    close.to_parquet(cache)
    return close


def close_from_yfinance(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Pull the Close columns out of yf.download output (handles both column layouts)."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        level = next((i for i in range(raw.columns.nlevels) if "Close" in raw.columns.get_level_values(i)), None)
        if level is None:
            raise ValueError(f"no Close column in yfinance output: {list(raw.columns)[:10]}")
        close = raw.xs("Close", axis=1, level=level)
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close = close[[t for t in tickers if t in close.columns]].dropna(how="all", axis=1)
    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close.index.name = "date"
    return close.sort_index()


def load_prices_csv(path: str | Path) -> pd.DataFrame:
    """CSV with a date column first and one column of prices per ticker."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df.sort_index()


@dataclass
class Dataset:
    """Aligned arrays for one asset. Row t = close of day t."""

    ticker: str
    dates: pd.DatetimeIndex
    X: np.ndarray          # (T, d) standardized features -> reservoir input
    X_ar: np.ndarray       # (T, lags) standardized lagged daily returns -> AR baseline
    target: np.ndarray     # (T,) forward h-day log return / trailing vol: what the readout fits
    fwd_ret: np.ndarray    # (T,) forward h-day log return (NaN for the last h rows)
    next_ret: np.ndarray   # (T,) simple return close t -> close t+1: what a position earns
    feature_names: list[str]
    horizon: int

    def __len__(self) -> int:
        return len(self.dates)

    def slice(self, start: int) -> "Dataset":
        """Drop the first `start` rows (used to line baselines up with the post-washout reservoir states)."""
        return replace(self, dates=self.dates[start:], X=self.X[start:], X_ar=self.X_ar[start:],
                       target=self.target[start:], fwd_ret=self.fwd_ret[start:], next_ret=self.next_ret[start:])


def _rolling_z(df: pd.DataFrame, window: int, clip: float) -> pd.DataFrame:
    """z-score against the trailing window only (causal), clipped so crashes don't blow up the reservoir."""
    mu = df.rolling(window, min_periods=window // 2).mean()
    sd = df.rolling(window, min_periods=window // 2).std()
    return ((df - mu) / sd).clip(-clip, clip)


def make_dataset(close: pd.Series, features=DEFAULT_FEATURES, horizon: int = 1, z_window: int = 252,
                 ar_lags: int = 10, clip: float = 5.0, ticker: str | None = None) -> Dataset:
    close = close.dropna().astype(float)
    unknown = [f for f in features if f not in FEATURES]
    if unknown:
        raise ValueError(f"unknown features {unknown}; available: {list(FEATURES)}")
    logp = np.log(close)
    r = logp.diff()
    raw = {
        "r1": r,
        "r5": r.rolling(5).sum(),
        "r20": r.rolling(20).sum(),
        "r60": r.rolling(60).sum(),
        "vol20": r.rolling(20).std(),
        "vol_ratio": np.log(r.rolling(5).std() / r.rolling(60).std()),
        "dd252": logp - logp.rolling(252).max(),
    }
    feats = pd.DataFrame({f: raw[f] for f in features}).replace([np.inf, -np.inf], np.nan)
    X = _rolling_z(feats, z_window, clip)
    zr = _rolling_z(r.to_frame("r"), z_window, clip)["r"]
    X_ar = pd.concat({f"lag{k}": zr.shift(k) for k in range(ar_lags)}, axis=1)

    vol = r.rolling(20).std().replace(0.0, np.nan)
    fwd = logp.shift(-horizon) - logp
    target = fwd / (vol * np.sqrt(horizon))
    next_ret = np.expm1(r.shift(-1))

    # keep rows from the first day where every input is defined
    ok = X.notna().all(axis=1) & X_ar.notna().all(axis=1) & vol.notna()
    if not ok.any():
        raise ValueError("not enough price history for the requested features")
    start = int(np.argmax(ok.to_numpy()))
    sl = slice(start, None)
    # A NaN later on (stale or missing prices) would propagate through the recurrent matrix and wreck every
    # state after it, so set those inputs to 0 (= "average" on the z-score scale). Uses nothing from the future.
    gaps = int(X.iloc[sl].isna().to_numpy().sum() + X_ar.iloc[sl].isna().to_numpy().sum())
    if gaps:
        warnings.warn(f"{ticker or close.name}: {gaps} missing feature values after the warm-up, set to 0")
    X, X_ar = X.fillna(0.0), X_ar.fillna(0.0)
    return Dataset(
        ticker=ticker or str(close.name or "asset"),
        dates=close.index[sl],
        X=X.to_numpy(np.float32)[sl],
        X_ar=X_ar.to_numpy(np.float32)[sl],
        target=target.to_numpy(np.float64)[sl],
        fwd_ret=fwd.to_numpy(np.float64)[sl],
        next_ret=next_ret.to_numpy(np.float64)[sl],
        feature_names=list(features),
        horizon=horizon,
    )
