# --- FILE: src/kramers_moyal/estimators.py ---
"""
Estimacion no parametrica de coeficientes de Kramers-Moyal D^(k)(x).

Trazabilidad tesis -> codigo:
    - tch2 (Expansion de Kramers-Moyal):
          D^(k)(x) = (1/k!) lim_{dt->0} (1/dt) <[X(t+dt)-x]^k | X(t)=x>
    - tch4 (Estimacion empirica): version finita
          D^(k)(x) ~ (1/k!dt) <[r(t+dt)-r]^k | r(t)=r>
    - protocolo seccion 4.5 (formulas 2 y 3): casos D^(1) y D^(2).

Aproximacion empirica: en cada bin del espacio de estados (cuantil-uniforme),
se estima el momento condicional de orden k de los incrementos a paso dt.

Outputs:
    - centros de bins (x_centers)
    - D^(k) por bin
    - errores estadisticos por bin (bootstrap si habilitado)
    - poblacion por bin (para diagnostico)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class KMEstimate:
    x_centers: np.ndarray
    bin_edges: np.ndarray
    counts: np.ndarray
    coefficients: dict[int, np.ndarray] = field(default_factory=dict)
    coefficients_se: dict[int, np.ndarray] = field(default_factory=dict)
    delta_t: float = 1.0
    binning: str = "quantile"
    n_used: int = 0


def _build_edges(x: np.ndarray, n_bins: int, scheme: str) -> np.ndarray:
    if scheme == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(x, qs)
        edges[0] -= 1e-12
        edges[-1] += 1e-12
        return edges
    if scheme == "uniform":
        return np.linspace(x.min() - 1e-12, x.max() + 1e-12, n_bins + 1)
    raise ValueError(scheme)


def _km_single(
    x_t: np.ndarray,
    incr: np.ndarray,
    edges: np.ndarray,
    orders: list[int],
    delta_t: float,
    min_obs: int,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Estima los momentos condicionales por bin para los ordenes pedidos."""
    n_bins = edges.size - 1
    bin_idx = np.digitize(x_t, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    counts = np.bincount(bin_idx, minlength=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out: dict[int, np.ndarray] = {}
    for k in orders:
        powered = incr ** k
        # promedio condicional por bin (vectorial)
        sums = np.bincount(bin_idx, weights=powered, minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(counts >= min_obs, sums / np.maximum(counts, 1), np.nan)
        from math import factorial
        out[k] = means / (factorial(k) * delta_t)
    return out, centers, counts


def estimate_km_coefficients(
    series: pd.Series,
    *,
    orders: list[int] | None = None,
    n_bins: int = 50,
    binning: str = "quantile",
    delta_t_step: int = 1,
    min_obs_per_bin: int = 30,
    bootstrap_n: int = 0,
    rng: np.random.Generator | None = None,
) -> KMEstimate:
    """Estima D^(k) para los ordenes solicitados.

    Parameters
    ----------
    series :
        Serie temporal (log-retornos, posiblemente estandarizados).
    orders :
        Lista de ordenes a estimar. Por defecto [1, 2, 4].
    delta_t_step :
        Numero de pasos discretos para el incremento. Para datos diarios,
        delta_t_step=1 implica dt = 1 dia.
    min_obs_per_bin :
        Si un bin tiene menos observaciones que este umbral -> NaN (descartado).
    bootstrap_n :
        Si > 0, ejecuta bootstrap por bloques para errores estandar por bin.
    """
    orders = list(orders or [1, 2, 4])
    x = series.dropna().to_numpy()
    if x.size <= delta_t_step + 10:
        raise ValueError("Serie demasiado corta para el delta_t solicitado.")

    x_t = x[:-delta_t_step]
    incr = x[delta_t_step:] - x_t
    edges = _build_edges(x_t, n_bins=n_bins, scheme=binning)

    coeffs, centers, counts = _km_single(
        x_t, incr, edges, orders, delta_t=float(delta_t_step), min_obs=min_obs_per_bin
    )

    se: dict[int, np.ndarray] = {}
    if bootstrap_n > 0:
        rng = rng or np.random.default_rng()
        n = x_t.size
        block = max(50, n // 100)
        boot_acc: dict[int, list[np.ndarray]] = {k: [] for k in orders}
        for _ in range(bootstrap_n):
            n_blocks = n // block + 1
            starts = rng.integers(0, n - block, size=n_blocks)
            idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
            xb_t = x_t[idx]
            ib = incr[idx]
            cb, _, _ = _km_single(xb_t, ib, edges, orders, float(delta_t_step), min_obs_per_bin)
            for k in orders:
                boot_acc[k].append(cb[k])
        for k in orders:
            stacked = np.vstack(boot_acc[k])
            se[k] = np.nanstd(stacked, axis=0, ddof=1)

    logger.info(
        "KM: dt=%d nbins=%d ordenes=%s n_used=%d",
        delta_t_step, n_bins, orders, x_t.size
    )
    return KMEstimate(
        x_centers=centers,
        bin_edges=edges,
        counts=counts,
        coefficients=coeffs,
        coefficients_se=se,
        delta_t=float(delta_t_step),
        binning=binning,
        n_used=int(x_t.size),
    )
