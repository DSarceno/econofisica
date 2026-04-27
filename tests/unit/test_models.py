# --- FILE: tests/unit/test_models.py ---
import numpy as np

from src.kramers_moyal.estimators import estimate_km_coefficients
from src.models.baseline_gaussian import GaussianBaseline
from src.models.langevin_reconstructed import LangevinReconstructed
from src.models.ornstein_uhlenbeck import OrnsteinUhlenbeck


def test_gaussian_fit_recovers_params(synthetic_gaussian_returns):
    m = GaussianBaseline.fit(synthetic_gaussian_returns)
    assert abs(m.mu) < 0.005
    assert abs(m.sigma - 0.01) < 0.002


def test_ou_mle_recovers_theta(synthetic_ou_series):
    m = OrnsteinUhlenbeck.fit_mle(synthetic_ou_series, dt=1.0)
    # theta verdadero: 0.05; permitimos rango amplio finite-sample
    assert 0.01 < m.theta < 0.15
    assert abs(m.mu) < 0.5


def test_langevin_from_km_smoke(synthetic_ou_series):
    km = estimate_km_coefficients(
        synthetic_ou_series, orders=[1, 2], n_bins=20, min_obs_per_bin=30
    )
    m = LangevinReconstructed.from_km_estimate(km, diffusion_floor=1e-6)
    xs = np.linspace(-2, 2, 50)
    d1 = m.drift(xs, 0.0)
    d2 = m.diffusion(xs, 0.0)
    assert d1.shape == xs.shape
    assert (d2 > 0).all()
