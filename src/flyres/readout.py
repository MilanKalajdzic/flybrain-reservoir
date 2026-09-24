"""Linear readout: ridge regression on reservoir states, refit walk-forward.

The only trained part of the whole model. Same code is used for the baselines (raw features, AR
lags), so every model gets exactly the same fitting procedure and test period.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ALPHAS = np.logspace(-4, 3, 15)


@dataclass
class RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    alpha: float
    val_mse: float = float("nan")  # error on the held-out end of the training window (in-sample if too short)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, dtype=np.float64) - self.mean) / self.scale) @ self.coef + self.intercept


def _standardize(X: np.ndarray):
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-8] = 1.0  # dead neurons: leave them at zero instead of dividing by ~0
    return mean, scale


def ridge_path(X: np.ndarray, y: np.ndarray, alphas=ALPHAS, penalty_factor=None):
    """Coefficients for every alpha from one eigendecomposition.

    Solves min_b (1/n)||y - Zb||^2 + alpha * sum_j f_j b_j^2 on standardized Z, where f is
    `penalty_factor` (default all ones). A tiny f_j leaves feature j practically unpenalized.
    Uses the n x n (dual) problem when there are more features than samples.
    Returns (coefs[len(alphas), p] in standardized units, intercept, mean, scale).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    alphas = np.atleast_1d(np.asarray(alphas, dtype=np.float64))
    mean, scale = _standardize(X)
    w = np.ones(X.shape[1]) if penalty_factor is None else 1.0 / np.sqrt(np.asarray(penalty_factor, dtype=np.float64))
    Z = (X - mean) / scale * w  # rescaling column j by w_j turns a plain ridge penalty into alpha * f_j
    intercept = y.mean()
    yc = y - intercept
    n, p = Z.shape
    if n >= p:
        evals, V = np.linalg.eigh(Z.T @ Z / n)
        b = V.T @ (Z.T @ yc / n)
        coefs = (V @ (b[:, None] / (evals[:, None] + alphas[None, :]))).T
    else:
        evals, U = np.linalg.eigh(Z @ Z.T / n)
        c = U.T @ yc
        coefs = ((Z.T @ U) @ (c[:, None] / (evals[:, None] + alphas[None, :])) / n).T
    return coefs * w, intercept, mean, scale


def fit_ridge(X: np.ndarray, y: np.ndarray, alphas=ALPHAS, val_frac: float = 0.2, gap: int = 1,
              penalty_factor=None) -> RidgeModel:
    """Pick alpha on the last `val_frac` of the training window (time-ordered, with a gap), then refit on all of it."""
    n = len(y)
    n_val = max(int(n * val_frac), 20)
    n_fit = n - n_val - gap
    alphas = np.atleast_1d(alphas)
    val_mse = float("nan")
    if n_fit >= 50 and len(alphas) > 1:
        coefs, icpt, mean, scale = ridge_path(X[:n_fit], y[:n_fit], alphas, penalty_factor)
        pred = ((X[n_fit + gap:] - mean) / scale) @ coefs.T + icpt
        mse = ((pred - y[n_fit + gap:, None]) ** 2).mean(axis=0)
        best = int(np.argmin(mse))
        alpha, val_mse = float(alphas[best]), float(mse[best])
    else:
        alpha = float(np.median(alphas))
    coefs, icpt, mean, scale = ridge_path(X, y, [alpha], penalty_factor)
    model = RidgeModel(mean, scale, coefs[0], icpt, alpha, val_mse)
    if not np.isfinite(model.val_mse):  # too little data for a validation split: fall back to in-sample
        model.val_mse = float(((model.predict(X) - y) ** 2).mean())
    return model


def walk_forward(X: np.ndarray, y: np.ndarray, horizon: int = 1, train_min: int = 756, refit_every: int = 126,
                 window: int | None = None, alphas=ALPHAS, val_frac: float = 0.2, penalty_factor=None):
    """Expanding (or rolling) window refits. Returns (predictions, splits).

    A model fitted at the close of day t0 predicts rows t0 .. t0+refit_every-1. It trains only on
    rows s with s + horizon <= t0, because the target of row s is the return up to s + horizon.
    predictions[t] is NaN before the first fit.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    if train_min + horizon >= T:
        raise ValueError(f"series too short: {T} rows for train_min={train_min}")
    valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
    pred = np.full(T, np.nan)
    splits = []
    for t0 in range(train_min, T, refit_every):
        t1 = min(t0 + refit_every, T)
        train_end = t0 - horizon + 1  # exclusive: targets of rows < train_end are known at t0
        train_start = 0 if window is None else max(0, train_end - window)
        tr = np.arange(train_start, train_end)
        tr = tr[valid[tr]]
        model = fit_ridge(X[tr], y[tr], alphas, val_frac, gap=horizon, penalty_factor=penalty_factor)
        pred[t0:t1] = model.predict(X[t0:t1])
        splits.append({"t0": t0, "t1": t1, "train_start": train_start, "train_end": train_end,
                       "n_train": len(tr), "alpha": model.alpha, "val_mse": model.val_mse})
    return pred, pd.DataFrame(splits)


def ridge_multi(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, alpha: float = 1e-4) -> np.ndarray:
    """Ridge with several targets and a fixed alpha (used by the memory-capacity benchmark)."""
    X_train = np.asarray(X_train, dtype=np.float64)
    mean, scale = _standardize(X_train)
    Z = (X_train - mean) / scale
    ym = Y_train.mean(axis=0)
    n, p = Z.shape
    B = np.linalg.solve(Z.T @ Z / n + alpha * np.eye(p), Z.T @ (Y_train - ym) / n)
    return ((np.asarray(X_test, dtype=np.float64) - mean) / scale) @ B + ym
