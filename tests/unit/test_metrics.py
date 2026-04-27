# --- FILE: tests/unit/test_metrics.py ---
import numpy as np

from src.validation.metrics import (
    compare_all,
    js_divergence,
    kl_divergence,
    ks_distance,
    moment_errors,
)


def test_metrics_identical_distributions(rng):
    x = rng.normal(0, 1, 5000)
    y = rng.normal(0, 1, 5000)
    res = compare_all(x, y, bins=80)
    assert res["ks_stat"] < 0.05
    assert res["jensen_shannon"] < 0.02


def test_metrics_different_distributions(rng):
    x = rng.normal(0, 1, 5000)
    y = rng.standard_t(df=3, size=5000)  # cola pesada
    res = compare_all(x, y, bins=80)
    assert res["ks_stat"] > 0.02


def test_kl_nonnegative(rng):
    x = rng.normal(0, 1, 1000)
    y = rng.normal(0.1, 1.05, 1000)
    assert kl_divergence(x, y, bins=50) >= 0


def test_js_symmetric_within_tol(rng):
    x = rng.normal(0, 1, 2000)
    y = rng.normal(0.1, 1.0, 2000)
    a = js_divergence(x, y, bins=50)
    b = js_divergence(y, x, bins=50)
    assert abs(a - b) < 1e-9
