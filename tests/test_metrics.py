import numpy as np
import pytest

from flyres.metrics import (block_bootstrap_sharpe_diff, evaluate, hit_rate, max_drawdown, positions, sharpe,
                            strategy_returns)


def test_sharpe():
    r = np.array([0.01, -0.005, 0.02, 0.0])
    assert sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert np.isnan(sharpe(np.array([0.01, 0.01])))


def test_max_drawdown():
    assert max_drawdown(np.array([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
    # measured from the starting capital: a loss on the first day counts
    assert max_drawdown(np.array([-0.5, 0.1])) == pytest.approx(-0.5)
    assert max_drawdown(np.array([-0.2, -0.1, 0.05])) == pytest.approx(0.8 * 0.9 - 1.0)


def test_costs_charged_on_turnover():
    pred = np.array([np.nan, 1.0, 2.0, -1.0])
    nxt = np.array([0.05, 0.01, 0.02, 0.03])
    net, gross, turnover = strategy_returns(pred, nxt, "sign", cost_bps=10)
    assert np.isnan(net[0])  # no forecast yet -> not in the test period
    np.testing.assert_allclose(turnover[1:], [1, 0, 2])
    np.testing.assert_allclose(net[1:], [0.01 - 0.001, 0.02, -0.03 - 0.002])
    np.testing.assert_allclose(gross[1:], [0.01, 0.02, -0.03])


def test_linear_positions_are_causal_and_bounded():
    pred = np.random.default_rng(0).standard_normal(300)
    pos = positions(pred, "linear")
    assert np.all(np.abs(pos) <= 1)
    changed = pred.copy()
    changed[200:] *= 50  # future predictions must not affect earlier positions
    np.testing.assert_array_equal(positions(changed, "linear")[:200], pos[:200])


def test_hit_rate_ignores_zeros_and_nans():
    assert hit_rate(np.array([1, -1, 1, np.nan, 1.0]), np.array([1, 1, 0, 1, 2.0])) == pytest.approx(2 / 3)


def test_evaluate_keys():
    rng = np.random.default_rng(1)
    m = evaluate(rng.standard_normal(500), rng.standard_normal(500) * 0.01, rng.standard_normal(500) * 0.01)
    assert {"ic", "hit_rate", "sharpe_net", "sharpe_gross", "max_drawdown_net", "turnover"} <= set(m)


def test_bootstrap_detects_a_real_difference_and_not_a_fake_one():
    rng = np.random.default_rng(2)
    base = rng.standard_normal(3000) * 0.01
    same = block_bootstrap_sharpe_diff(base, base.copy(), n_boot=500)
    assert same["diff"] == pytest.approx(0.0) and same["ci_low"] <= 0 <= same["ci_high"]
    better = block_bootstrap_sharpe_diff(base + 0.002, base, n_boot=500)
    assert better["diff"] > 0 and better["ci_low"] > 0 and better["p_value"] < 0.01
    # p and the interval come from the same draws, so they never contradict each other
    for shift in (0.0, 0.0002, 0.0004, 0.0006, 0.0008):
        r = block_bootstrap_sharpe_diff(base + shift, base + rng.standard_normal(3000) * 0.004, n_boot=500)
        assert (r["p_value"] < 0.05) == (r["ci_low"] > 0 or r["ci_high"] < 0)
