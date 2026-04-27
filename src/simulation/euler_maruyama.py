# --- FILE: src/simulation/euler_maruyama.py ---
"""
Integrador Euler-Maruyama para SDE escalares en interpretacion de Ito.

Trazabilidad tesis -> codigo:
    - tch1 (Metodo numerico) y protocolo seccion 4.6 / 5.9:
          x_{t+dt} = x_t + D^(1)(x_t) dt + sqrt(2 D^(2)(x_t) dt) * eta,
          eta ~ N(0,1).

Implementacion:
    - Vectorizado por paths (n_paths integraciones simultaneas).
    - 'burn_in' descarta los primeros pasos para mitigar dependencia de la
      condicion inicial (relajacion al estado pseudo-estacionario).
    - RNG explicito por path para reproducibilidad bit-exacta (utils.seeding).
"""
from __future__ import annotations

import logging

import numpy as np

from src.models.base import StochasticModel
from src.utils.seeding import derived_generator

logger = logging.getLogger(__name__)


def _initial_state(
    kind: str,
    n_paths: int,
    rng: np.random.Generator,
    *,
    data_first: float | None = None,
    sample_pool: np.ndarray | None = None,
) -> np.ndarray:
    if kind == "zero":
        return np.zeros(n_paths)
    if kind == "data_first":
        if data_first is None:
            raise ValueError("data_first requiere proveer 'data_first'.")
        return np.full(n_paths, float(data_first))
    if kind == "sample":
        if sample_pool is None:
            raise ValueError("sample requiere 'sample_pool'.")
        return rng.choice(sample_pool, size=n_paths, replace=True)
    raise ValueError(f"Tipo de condicion inicial desconocido: {kind}")


def simulate_paths(
    model: StochasticModel,
    *,
    n_paths: int,
    n_steps: int,
    dt: float = 1.0,
    burn_in: int = 0,
    parent_seed: int = 0,
    initial_condition: str = "zero",
    data_first: float | None = None,
    sample_pool: np.ndarray | None = None,
) -> np.ndarray:
    """Simula 'n_paths' trayectorias de longitud 'n_steps' (post burn-in).

    Returns
    -------
    np.ndarray of shape (n_steps, n_paths)
    """
    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths y n_steps deben ser positivos.")
    rng_init = derived_generator(parent_seed, stream_id=0)
    x = _initial_state(
        initial_condition,
        n_paths,
        rng_init,
        data_first=data_first,
        sample_pool=sample_pool,
    )

    total_steps = burn_in + n_steps
    out = np.empty((n_steps, n_paths), dtype=np.float64)

    rng = derived_generator(parent_seed, stream_id=1)
    sqrt_dt = np.sqrt(dt)
    for step in range(total_steps):
        d1 = model.drift(x, step * dt)
        d2 = np.maximum(model.diffusion(x, step * dt), 0.0)
        eta = rng.standard_normal(n_paths)
        x = x + d1 * dt + np.sqrt(2.0 * d2) * sqrt_dt * eta
        if step >= burn_in:
            out[step - burn_in] = x

    logger.info(
        "EulerMaruyama: model=%s n_paths=%d n_steps=%d burn=%d dt=%.4f",
        getattr(model, "name", "unknown"), n_paths, n_steps, burn_in, dt,
    )
    return out
