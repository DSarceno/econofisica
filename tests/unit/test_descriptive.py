# --- FILE: tests/unit/test_descriptive.py ---
import numpy as np
import pandas as pd

from src.statistics.descriptive import (
    autocorrelation,
    describe,
    empirical_pdf,
    hill_tail_index,
    squared_autocorrelation,
)


def test_describe_returns_expected_fields():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 1, 5000))
    d = describe(s)
    assert d.n == 5000
    assert abs(d.mean) < 0.05
    assert abs(d.std - 1.0) < 0.05
    assert "kurtosis_excess" in d.to_dict()


def test_empirical_pdf_shape():
    s = pd.Series(np.random.default_rng(0).normal(0, 1, 2000))
    centers, density = empirical_pdf(s, bins=50)
    assert centers.shape == (50,)
    assert density.shape == (50,)


def test_acf_at_lag0_is_one():
    s = pd.Series(np.random.default_rng(0).normal(0, 1, 1000))
    a = autocorrelation(s, nlags=10)
    assert abs(a[0] - 1.0) < 1e-9
    a2 = squared_autocorrelation(s, nlags=10)
    assert abs(a2[0] - 1.0) < 1e-9


def test_hill_tail_index_pareto():
    # Pareto con alpha=3 -> Hill ~ 3
    rng = np.random.default_rng(42)
    x = rng.pareto(3.0, 5000) + 1.0
    s = pd.Series(x)
    alpha = hill_tail_index(s, k=200)
    assert 2.5 < alpha < 3.5
