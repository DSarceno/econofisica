# --- FILE: src/models/potential.py ---
"""
Reconstruccion de potencial efectivo a partir de la distribucion estacionaria.

Trazabilidad tesis -> codigo:
    - tch1 (Potencial efectivo): cuando D^(2) ~ const,
          U(x) = -log(P_st(x))   (modulo constante)
    - apendice (Interpretacion del potencial efectivo):
          P_st(x) ∝ exp(-U(x))
          D^(1)(x) = -dU/dx + dD^(2)/dx

Esta utilidad provee dos rutas alternativas a la del Langevin reconstruido:
    1. Potencial 'estadistico' desde la PDF empirica.
    2. Potencial 'dinamico' desde D^(1) integrado.
"""
from __future__ import annotations

import numpy as np

from src.statistics.descriptive import empirical_pdf


def potential_from_pdf(series, *, bins: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """U(x) = -log P_emp(x), normalizado para U_min = 0."""
    centers, density = empirical_pdf(series, bins=bins)
    eps = 1e-12
    U = -np.log(density + eps)
    U -= U.min()
    return centers, U


def potential_from_drift(x_grid: np.ndarray, drift: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """U(x) = -integral D^(1)(x) dx (suponiendo difusion ~ const)."""
    dx = np.diff(x_grid, prepend=x_grid[0])
    U = -np.cumsum(drift * dx)
    U -= U.min()
    return x_grid, U
