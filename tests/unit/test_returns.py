# --- FILE: tests/unit/test_returns.py ---
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.returns import log_returns, standardize, winsorize


def test_log_returns_basic():
    p = pd.Series([100, 101, 102, 100, 105], name="price")
    r = log_returns(p, drop_na=True)
    assert len(r) == 4
    assert np.isclose(r.iloc[0], np.log(101 / 100))


def test_log_returns_negative_prices_raise():
    p = pd.Series([100, -1.0, 102])
    with pytest.raises(ValueError):
        log_returns(p)


def test_standardize_global():
    s = pd.Series(np.random.default_rng(0).normal(0, 2.0, 1000))
    z = standardize(s, method="global")
    assert abs(z.mean()) < 1e-12
    assert abs(z.std(ddof=1) - 1.0) < 1e-12


def test_winsorize_clips_tails():
    s = pd.Series(np.concatenate([[-100.0], np.zeros(998), [100.0]]))
    w = winsorize(s, quantiles=(0.01, 0.99))
    assert w.min() > -10
    assert w.max() < 10
