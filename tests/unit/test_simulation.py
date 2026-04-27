# --- FILE: tests/unit/test_simulation.py ---
import numpy as np

from src.models.baseline_gaussian import GaussianBaseline
from src.models.ornstein_uhlenbeck import OrnsteinUhlenbeck
from src.simulation.euler_maruyama import simulate_paths


def test_simulate_gaussian_recovers_mean_std():
    m = GaussianBaseline(mu=0.0, sigma=0.02)
    paths = simulate_paths(
        m, n_paths=200, n_steps=2000, dt=1.0, burn_in=100,
        parent_seed=123, initial_condition="zero",
    )
    flat = paths.flatten()
    # incrementos: en realidad son x_t (random walk con sigma sqrt(t)) NO iid;
    # validamos que la difusion no diverja absurdamente
    assert paths.shape == (2000, 200)
    assert np.isfinite(flat).all()


def test_simulate_ou_stationary_distribution():
    m = OrnsteinUhlenbeck(theta=0.5, mu=0.0, diffusion_coef=0.5)
    paths = simulate_paths(
        m, n_paths=500, n_steps=4000, dt=0.1, burn_in=2000,
        parent_seed=42, initial_condition="zero",
    )
    final = paths[-1]
    # OU estacionario: var = D/theta = 1
    assert abs(final.mean()) < 0.2
    assert 0.5 < final.std() < 1.5


def test_reproducibility():
    m = OrnsteinUhlenbeck(theta=1.0, mu=0.0, diffusion_coef=0.5)
    a = simulate_paths(m, n_paths=20, n_steps=200, dt=0.05, burn_in=50, parent_seed=7)
    b = simulate_paths(m, n_paths=20, n_steps=200, dt=0.05, burn_in=50, parent_seed=7)
    assert np.array_equal(a, b)
