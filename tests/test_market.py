import numpy as np
import pandas as pd
import pytest

from flyres.market import close_from_yfinance, make_dataset
from flyres.synthetic import synthetic_prices


@pytest.fixture(scope="module")
def close():
    return synthetic_prices(1500, seed=1)


def test_dataset_is_clean_and_aligned(close):
    ds = make_dataset(close, horizon=5)
    assert np.isfinite(ds.X).all() and np.isfinite(ds.X_ar).all()
    c = close.loc[ds.dates].to_numpy()
    i = 100
    assert ds.fwd_ret[i] == pytest.approx(np.log(c[i + 5] / c[i]))
    assert ds.next_ret[i] == pytest.approx(c[i + 1] / c[i] - 1)
    assert np.isnan(ds.fwd_ret[-5:]).all() and np.isfinite(ds.fwd_ret[:-5]).all()


def test_features_are_causal(close):
    """Changing prices after day t must not change any feature at or before t."""
    ds = make_dataset(close)
    cut = ds.dates[400]
    shocked = close.copy()
    shocked[shocked.index > cut] *= np.linspace(1, 3, int((shocked.index > cut).sum()))
    ds2 = make_dataset(shocked)
    np.testing.assert_array_equal(ds.X[:401], ds2.X[:401])
    np.testing.assert_array_equal(ds.X_ar[:401], ds2.X_ar[:401])


def test_slice(close):
    ds = make_dataset(close)
    s = ds.slice(100)
    assert len(s) == len(ds) - 100 and s.dates[0] == ds.dates[100]
    np.testing.assert_array_equal(s.X, ds.X[100:])


def test_stale_prices_do_not_produce_nans(close):
    stale = close.copy()
    stale.iloc[600:606] = stale.iloc[600]  # a week of identical prices: 5-day vol = 0, log(0) = -inf
    with pytest.warns(UserWarning, match="missing feature values"):
        ds = make_dataset(stale)
    assert np.isfinite(ds.X).all() and np.isfinite(ds.X_ar).all()


def test_unknown_feature(close):
    with pytest.raises(ValueError, match="unknown features"):
        make_dataset(close, features=["r1", "moon_phase"])


def test_yfinance_layouts():
    idx = pd.date_range("2020-01-01", periods=3, tz="America/New_York")
    multi = pd.DataFrame(np.arange(12.0).reshape(3, 4), index=idx,
                         columns=pd.MultiIndex.from_product([["Close", "Open"], ["SPY", "QQQ"]]))
    out = close_from_yfinance(multi, ["SPY", "QQQ"])
    assert list(out.columns) == ["SPY", "QQQ"] and out.index.tz is None
    flat = pd.DataFrame({"Close": [1.0, 2.0, 3.0], "Open": [1.0, 1.0, 1.0]}, index=idx)
    assert list(close_from_yfinance(flat, ["SPY"]).columns) == ["SPY"]
    assert close_from_yfinance(pd.DataFrame(), ["SPY"]).empty
