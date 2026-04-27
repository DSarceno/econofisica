# --- FILE: tests/integration/test_pipeline_smoke.py ---
"""Smoke test end-to-end usando series sinteticas (sin red)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.kramers_moyal.estimators import estimate_km_coefficients
from src.markov.chapman_kolmogorov import chapman_kolmogorov_test
from src.models.langevin_reconstructed import LangevinReconstructed
from src.models.ornstein_uhlenbeck import OrnsteinUhlenbeck
from src.preprocessing.returns import standardize
from src.simulation.euler_maruyama import simulate_paths
from src.statistics.descriptive import describe
from src.validation.metrics import compare_all


def test_full_pipeline_synthetic(synthetic_ou_series):
    s = synthetic_ou_series
    z = standardize(s, method="global")

    d = describe(z)
    assert d.n > 0

    ck = chapman_kolmogorov_test(z, delta_t_steps=2, n_bins=15, bootstrap_samples=0)
    assert ck.mean_divergence < 0.1

    km = estimate_km_coefficients(z, orders=[1, 2], n_bins=20, min_obs_per_bin=30)

    m_ou = OrnsteinUhlenbeck.fit_mle(z)
    m_lan = LangevinReconstructed.from_km_estimate(km)

    sim_ou = simulate_paths(
        m_ou, n_paths=50, n_steps=2000, dt=1.0, burn_in=500, parent_seed=1
    )
    sim_lan = simulate_paths(
        m_lan, n_paths=50, n_steps=2000, dt=1.0, burn_in=500, parent_seed=2,
        initial_condition="data_first", data_first=float(z.iloc[0]),
    )

    res_ou = compare_all(z.to_numpy(), sim_ou[-1])
    res_lan = compare_all(z.to_numpy(), sim_lan[-1])
    assert np.isfinite(res_ou["jensen_shannon"])
    assert np.isfinite(res_lan["jensen_shannon"])
