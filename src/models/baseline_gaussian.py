# --- FILE: src/models/baseline_gaussian.py ---
"""
Modelo M0: ruido blanco gaussiano (Bachelier puro, drift cero, difusion constante).

Trazabilidad tesis -> codigo:
    - tch3 (Bachelier): r_t ~ N(mu, sigma^2) iid.
    - protocolo seccion 4.1.11.1: M0 baseline.

Forma SDE equivalente:
    dx = mu dt + sqrt(2 * (sigma^2 / 2)) dW = mu dt + sigma dW
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import StochasticModel


@dataclass
class GaussianBaseline(StochasticModel):
    mu: float = 0.0
    sigma: float = 1.0
    name: str = "M0_gaussian"

    @classmethod
    def fit(cls, series: pd.Series) -> "GaussianBaseline":
        x = series.dropna().to_numpy()
        return cls(mu=float(np.mean(x)), sigma=float(np.std(x, ddof=1)))

    def drift(self, x: np.ndarray, t: float) -> np.ndarray:
        return np.full_like(x, self.mu, dtype=np.float64)

    def diffusion(self, x: np.ndarray, t: float) -> np.ndarray:
        # En la convencion dx = mu dt + sqrt(2 D^(2)) dW, sigma^2 = 2 D^(2)
        return np.full_like(x, 0.5 * self.sigma * self.sigma, dtype=np.float64)

    def to_dict(self) -> dict:
        return {"name": self.name, "mu": self.mu, "sigma": self.sigma}
