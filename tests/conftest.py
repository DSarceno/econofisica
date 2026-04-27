# --- FILE: tests/conftest.py ---
"""Fixtures compartidas para los tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260426)


@pytest.fixture()
def synthetic_ou_series(rng: np.random.Generator) -> pd.Series:
    """Genera una serie OU sintetica con parametros conocidos.

    SDE: dx = -theta*(x - mu) dt + sigma dW, theta=0.05, mu=0, sigma=1.
    Solucion exacta para dt=1:
        x_{t+1} = mu + (x_t - mu) * exp(-theta) + sigma * sqrt((1 - exp(-2 theta)) / (2 theta)) * eps
    """
    n = 5000
    theta = 0.05
    mu = 0.0
    sigma = 1.0
    a = np.exp(-theta)
    b = sigma * np.sqrt((1 - a * a) / (2 * theta))
    x = np.zeros(n)
    eps = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = mu + (x[t - 1] - mu) * a + b * eps[t]
    idx = pd.date_range("2000-01-03", periods=n, freq="B")
    return pd.Series(x, index=idx, name="ou_synthetic")


@pytest.fixture()
def synthetic_gaussian_returns(rng: np.random.Generator) -> pd.Series:
    n = 4000
    x = rng.normal(loc=0.0, scale=0.01, size=n)
    idx = pd.date_range("2000-01-03", periods=n, freq="B")
    return pd.Series(x, index=idx, name="gauss_iid")
