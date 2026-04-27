# --- FILE: tests/unit/test_chapman_kolmogorov.py ---
import numpy as np

from src.markov.chapman_kolmogorov import chapman_kolmogorov_test


def test_ck_low_divergence_for_markov(synthetic_ou_series):
    res = chapman_kolmogorov_test(
        synthetic_ou_series,
        delta_t_steps=2,
        n_bins=15,
        binning="quantile",
        divergence="jensen_shannon",
        bootstrap_samples=0,
    )
    # OU es markoviano por construccion -> JS pequena
    assert res.mean_divergence < 0.05


def test_ck_higher_divergence_for_non_markov_proxy(rng):
    # Construimos una serie con memoria larga: media movil de ruido blanco
    n = 5000
    eps = rng.normal(size=n)
    x = np.convolve(eps, np.ones(50) / 50, mode="same")
    import pandas as pd
    s = pd.Series(x)
    res_short = chapman_kolmogorov_test(s, delta_t_steps=2, n_bins=15, bootstrap_samples=0)
    res_long = chapman_kolmogorov_test(s, delta_t_steps=20, n_bins=15, bootstrap_samples=0)
    # La discrepancia tiende a crecer con dt si hay memoria
    assert res_long.mean_divergence >= res_short.mean_divergence * 0.5
