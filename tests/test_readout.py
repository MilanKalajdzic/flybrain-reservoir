import numpy as np
import pytest

from flyres.metrics import information_coefficient
from flyres.readout import fit_ridge, ridge_path, walk_forward


def _explicit_ridge(X, y, alpha, pf=None):
    mean, sd = X.mean(0), X.std(0)
    Z = (X - mean) / sd
    pf = np.ones(X.shape[1]) if pf is None else np.asarray(pf)
    n = len(y)
    return np.linalg.solve(Z.T @ Z / n + alpha * np.diag(pf), Z.T @ (y - y.mean()) / n)


@pytest.mark.parametrize("n, p", [(200, 10), (30, 80)])  # primal and dual branch
def test_ridge_path_matches_closed_form(n, p):
    rng = np.random.default_rng(0)
    X, y = rng.standard_normal((n, p)), rng.standard_normal(n)
    coefs, *_ = ridge_path(X, y, [0.1, 1.0])
    np.testing.assert_allclose(coefs[0], _explicit_ridge(X, y, 0.1), atol=1e-8)
    np.testing.assert_allclose(coefs[1], _explicit_ridge(X, y, 1.0), atol=1e-8)


def test_penalty_factor_is_per_feature_penalty():
    rng = np.random.default_rng(1)
    X, y = rng.standard_normal((150, 6)), rng.standard_normal(150)
    pf = [0.001, 0.001, 1, 1, 1, 1]
    coefs, *_ = ridge_path(X, y, [0.5], penalty_factor=pf)
    np.testing.assert_allclose(coefs[0], _explicit_ridge(X, y, 0.5, pf), atol=1e-8)


def test_ridge_recovers_known_signal():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((2000, 5))
    y = X @ np.array([1.0, -2.0, 0.0, 0.5, 0.0]) + 0.1 * rng.standard_normal(2000)
    model = fit_ridge(X, y)
    Xt = rng.standard_normal((500, 5))
    assert np.corrcoef(model.predict(Xt), Xt @ np.array([1.0, -2.0, 0.0, 0.5, 0.0]))[0, 1] > 0.999


@pytest.mark.parametrize("horizon", [1, 5])
def test_walk_forward_never_trains_on_unknown_targets(horizon):
    rng = np.random.default_rng(3)
    X, y = rng.standard_normal((1200, 4)), rng.standard_normal(1200)
    _, splits = walk_forward(X, y, horizon=horizon, train_min=300, refit_every=100)
    # last training row + horizon must be <= the fit date
    assert ((splits["train_end"] - 1 + horizon) <= splits["t0"]).all()


@pytest.mark.parametrize("horizon", [1, 5])
def test_walk_forward_ignores_the_future(horizon):
    """Scrambling every target that isn't known yet at the first fit must not change the first block."""
    rng = np.random.default_rng(4)
    X, y = rng.standard_normal((1000, 4)), rng.standard_normal(1000)
    p1, _ = walk_forward(X, y, horizon=horizon, train_min=300, refit_every=100)
    y2 = y.copy()
    y2[300 - horizon + 1:] = rng.standard_normal(len(y2) - (300 - horizon + 1)) * 100
    p2, _ = walk_forward(X, y2, horizon=horizon, train_min=300, refit_every=100)
    np.testing.assert_array_equal(p1[300:400], p2[300:400])
    assert np.isnan(p1[:300]).all()


def test_noise_gives_no_skill():
    rng = np.random.default_rng(5)
    X, y = rng.standard_normal((3000, 20)), rng.standard_normal(3000)
    pred, _ = walk_forward(X, y, train_min=500, refit_every=250)
    assert abs(information_coefficient(pred, y)) < 0.06


def test_rolling_window():
    rng = np.random.default_rng(6)
    X, y = rng.standard_normal((1000, 3)), rng.standard_normal(1000)
    _, splits = walk_forward(X, y, train_min=300, refit_every=200, window=250)
    assert (splits["n_train"] <= 250).all()
