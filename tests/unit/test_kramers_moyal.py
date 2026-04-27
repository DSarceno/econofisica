# --- FILE: tests/unit/test_kramers_moyal.py ---
"""
Tests cientificos: si el proceso es OU con parametros conocidos, los
coeficientes K-M deberian recuperar (asintoticamente) drift lineal y
difusion constante.
"""
import numpy as np

from src.kramers_moyal.estimators import estimate_km_coefficients


def test_km_recovers_ou_drift_slope(synthetic_ou_series):
    km = estimate_km_coefficients(
        synthetic_ou_series,
        orders=[1, 2],
        n_bins=20,
        binning="quantile",
        delta_t_step=1,
        min_obs_per_bin=30,
    )
    xc = km.x_centers
    d1 = km.coefficients[1]
    mask = np.isfinite(d1)
    slope = np.polyfit(xc[mask], d1[mask], 1)[0]
    # OU: D^(1)(x) = -theta * x con theta=0.05 -> pendiente ~ -0.05
    assert -0.10 < slope < -0.01


def test_km_recovers_ou_diffusion_positive(synthetic_ou_series):
    km = estimate_km_coefficients(
        synthetic_ou_series,
        orders=[1, 2],
        n_bins=20,
        delta_t_step=1,
        min_obs_per_bin=30,
    )
    d2 = km.coefficients[2]
    mask = np.isfinite(d2)
    assert (d2[mask] > 0).all()
    # difusion casi constante: coeficiente de variacion bajo
    cv = np.nanstd(d2[mask]) / max(np.nanmean(d2[mask]), 1e-12)
    assert cv < 0.6  # tolerancia amplia por finite-sample


def test_km_min_obs_filters_sparse_bins(synthetic_ou_series):
    km = estimate_km_coefficients(
        synthetic_ou_series,
        orders=[1],
        n_bins=200,                    # muchos bins -> algunos sin obs
        delta_t_step=1,
        min_obs_per_bin=200,
    )
    # con umbral alto, esperamos varios NaN
    assert np.isnan(km.coefficients[1]).any()
