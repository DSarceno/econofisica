# --- FILE: tests/unit/test_pawula.py ---
import numpy as np

from src.kramers_moyal.estimators import estimate_km_coefficients
from src.kramers_moyal.pawula import pawula_ratios


def test_pawula_ratios_finite_for_ou(synthetic_ou_series):
    km = estimate_km_coefficients(
        synthetic_ou_series, orders=[1, 2, 4], n_bins=20, min_obs_per_bin=30
    )
    res = pawula_ratios(km.x_centers, km.coefficients, higher_orders=[4])
    # Para OU exacto (gaussiano), R_2 deberia ser bajo (ideal 0).
    assert 4 in res.summary
    assert np.isfinite(res.summary[4]["mean_abs_ratio"])
