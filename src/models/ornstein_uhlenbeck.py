# --- FILE: src/models/ornstein_uhlenbeck.py ---
"""
Modelo M1: proceso de Ornstein-Uhlenbeck (reversion a la media, ruido aditivo).

SDE:    dx = -theta (x - mu) dt + sqrt(2 D) dW

Trazabilidad tesis -> codigo:
    - tch1 (Langevin con potencial cuadratico): U(x) = (theta/2)(x-mu)^2 -> drift lineal.
    - protocolo seccion 4.1.11.2: M1 OU.

Estimacion (MLE para series equiespaciadas dt=1):
    Si y_t = x_{t+1}, x_t -> y = a + b*x + eps con b = e^{-theta dt}, eps ~ N(0, var_eps),
    se obtiene:
        theta = -ln(b)/dt
        mu    = a / (1 - b)
        D     = var_eps * theta / (1 - b^2)   (siendo D el coef. de difusion D^(2))
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import StochasticModel


@dataclass
class OrnsteinUhlenbeck(StochasticModel):
    theta: float = 1.0
    mu: float = 0.0
    diffusion_coef: float = 0.5  # D^(2)
    name: str = "M1_OU"

    @classmethod
    def fit_mle(cls, series: pd.Series, dt: float = 1.0) -> "OrnsteinUhlenbeck":
        x = series.dropna().to_numpy()
        x_t = x[:-1]
        x_t1 = x[1:]
        # Regresion OLS y_t = a + b*x_t
        X = np.vstack([np.ones_like(x_t), x_t]).T
        coefs, *_ = np.linalg.lstsq(X, x_t1, rcond=None)
        a, b = float(coefs[0]), float(coefs[1])
        b = float(np.clip(b, -0.999, 0.999))
        residuals = x_t1 - (a + b * x_t)
        var_eps = float(np.var(residuals, ddof=1))
        theta = -np.log(b) / dt if b > 0 else 1.0 / dt
        mu = a / (1.0 - b) if abs(1.0 - b) > 1e-9 else 0.0
        denom = (1.0 - b * b)
        D = (var_eps * theta) / (2.0 * denom) if denom > 1e-9 else var_eps / 2.0
        return cls(theta=float(theta), mu=float(mu), diffusion_coef=float(max(D, 1e-12)))

    @classmethod
    def fit_km(
        cls,
        x_centers: np.ndarray,
        d1: np.ndarray,
        d2: np.ndarray,
    ) -> "OrnsteinUhlenbeck":
        """Ajuste alternativo desde coeficientes K-M empiricos.

        D^(1)(x) ~ -theta(x - mu)  ->  regresion lineal.
        D^(2)(x) ~ const           ->  promedio.
        """
        mask = np.isfinite(d1) & np.isfinite(d2)
        xc = x_centers[mask]
        d1m = d1[mask]
        d2m = d2[mask]
        A = np.vstack([np.ones_like(xc), xc]).T
        beta, *_ = np.linalg.lstsq(A, d1m, rcond=None)
        a, slope = float(beta[0]), float(beta[1])
        theta = -slope
        mu = a / (-slope) if abs(slope) > 1e-12 else 0.0
        D = float(np.nanmean(d2m))
        return cls(theta=float(max(theta, 1e-9)), mu=float(mu), diffusion_coef=float(max(D, 1e-12)))

    def drift(self, x: np.ndarray, t: float) -> np.ndarray:
        return -self.theta * (x - self.mu)

    def diffusion(self, x: np.ndarray, t: float) -> np.ndarray:
        return np.full_like(x, self.diffusion_coef, dtype=np.float64)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "theta": self.theta,
            "mu": self.mu,
            "diffusion_coef": self.diffusion_coef,
        }
