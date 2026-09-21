"""Forecast and trading metrics, plus a block bootstrap for "is A really better than B?"."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def positions(pred: np.ndarray, mode: str = "sign") -> np.ndarray:
    """Prediction -> position in [-1, 1]. NaN predictions (before the first fit) mean flat.

    sign:   +1 / -1 on the predicted direction.
    linear: prediction divided by its own trailing std (past predictions only), clipped to [-1, 1].
    """
    pred = np.asarray(pred, dtype=np.float64)
    if mode == "sign":
        pos = np.sign(pred)
    elif mode == "linear":
        scale = pd.Series(pred).expanding(min_periods=20).std().shift(1).to_numpy()
        pos = np.clip(pred / scale, -1.0, 1.0)
    else:
        raise ValueError("position mode must be 'sign' or 'linear'")
    return np.where(np.isfinite(pos), pos, 0.0)


def strategy_returns(pred: np.ndarray, next_ret: np.ndarray, mode: str = "sign", cost_bps: float = 1.0):
    """Daily net returns (NaN outside the test period) and turnover of trading the predictions."""
    pred = np.asarray(pred, dtype=np.float64)
    next_ret = np.asarray(next_ret, dtype=np.float64)
    test = np.isfinite(pred) & np.isfinite(next_ret)
    net = np.full(len(pred), np.nan)
    gross = np.full(len(pred), np.nan)
    turnover = np.full(len(pred), np.nan)
    if not test.any():
        return net, gross, turnover
    idx = np.flatnonzero(test)
    sl = slice(idx[0], idx[-1] + 1)
    pos = positions(pred[sl], mode)
    r = np.nan_to_num(next_ret[sl])
    to = np.abs(np.diff(pos, prepend=0.0))
    gross[sl] = pos * r
    net[sl] = pos * r - cost_bps * 1e-4 * to
    turnover[sl] = to
    return net, gross, turnover


def sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    wealth = np.cumprod(1.0 + r)
    return float((wealth / np.maximum.accumulate(wealth) - 1.0).min())


def hit_rate(pred: np.ndarray, realized: np.ndarray) -> float:
    ok = np.isfinite(pred) & np.isfinite(realized) & (pred != 0) & (realized != 0)
    return float((np.sign(pred[ok]) == np.sign(realized[ok])).mean()) if ok.any() else float("nan")


def information_coefficient(pred: np.ndarray, realized: np.ndarray) -> float:
    """Spearman rank correlation between prediction and realized return."""
    ok = np.isfinite(pred) & np.isfinite(realized)
    if ok.sum() < 3 or np.std(pred[ok]) == 0:
        return float("nan")
    return float(stats.spearmanr(pred[ok], realized[ok])[0])


def evaluate(pred: np.ndarray, fwd_ret: np.ndarray, next_ret: np.ndarray, mode: str = "sign",
             cost_bps: float = 1.0) -> dict:
    net, gross, turnover = strategy_returns(pred, next_ret, mode, cost_bps)
    return {
        "n_test": int(np.isfinite(net).sum()),
        "hit_rate": hit_rate(pred, fwd_ret),
        "ic": information_coefficient(pred, fwd_ret),
        "sharpe_gross": sharpe(gross),
        "sharpe_net": sharpe(net),
        "ann_return_net": float(np.nanmean(net) * TRADING_DAYS) if np.isfinite(net).any() else float("nan"),
        "max_drawdown_net": max_drawdown(net),
        "turnover": float(np.nanmean(turnover)) if np.isfinite(turnover).any() else float("nan"),
    }


def _sharpe_rows(R: np.ndarray) -> np.ndarray:
    sd = R.std(axis=1, ddof=1)
    return np.where(sd > 0, R.mean(axis=1) / np.where(sd > 0, sd, 1.0) * np.sqrt(TRADING_DAYS), np.nan)


def block_bootstrap_sharpe_diff(ra: np.ndarray, rb: np.ndarray, block: int = 20, n_boot: int = 2000,
                                seed: int = 0) -> dict:
    """Sharpe(a) - Sharpe(b) with a moving-block bootstrap CI and two-sided p-value.

    Resampling blocks of consecutive days (same days for both strategies) keeps volatility
    clustering and the correlation between the two return streams intact.
    """
    ra, rb = np.asarray(ra, dtype=np.float64), np.asarray(rb, dtype=np.float64)
    ok = np.isfinite(ra) & np.isfinite(rb)
    ra, rb = ra[ok], rb[ok]
    n = len(ra)
    obs = sharpe(ra) - sharpe(rb)
    if n < 2 * block:
        return {"diff": obs, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = []
    for start in range(0, n_boot, 250):  # chunks keep memory small
        size = min(250, n_boot - start)
        starts = rng.integers(0, n - block + 1, size=(size, n_blocks))
        idx = (starts[:, :, None] + np.arange(block)).reshape(size, -1)[:, :n]
        diffs.append(_sharpe_rows(ra[idx]) - _sharpe_rows(rb[idx]))
    diffs = np.concatenate(diffs)
    diffs = diffs[np.isfinite(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = float(np.mean(np.abs(diffs - diffs.mean()) >= abs(obs)))
    return {"diff": float(obs), "ci_low": float(lo), "ci_high": float(hi), "p_value": p}
