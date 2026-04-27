# --- FILE: src/kramers_moyal/pawula.py ---
"""
Test de Pawula: validez del truncamiento difusivo de la expansion de KM.

Trazabilidad tesis -> codigo:
    - tch2 (Teorema de Pawula): si la expansion de KM se trunca en orden finito
      distinto de 2, no se preserva la positividad de la densidad. Por tanto,
      idealmente D^(k) = 0 para k >= 3. En la practica empirica esto se evalua
      mediante:
            R_k(x) = D^(2k)(x) / [(2k-1)!! * (D^(2)(x))^k]
      Si R_k -> 0 para k >= 2, la aproximacion difusiva es razonable.

Convencion: usamos los ordenes pares (4, 6) para escalas comparables.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np


def _double_factorial(n: int) -> int:
    if n <= 0:
        return 1
    res = 1
    for k in range(n, 0, -2):
        res *= k
    return res


@dataclass
class PawulaResult:
    x_centers: np.ndarray
    ratios: dict[int, np.ndarray]  # k -> R_k(x), k >= 2 (orden par 2k)
    summary: dict[int, dict[str, float]]


def pawula_ratios(
    x_centers: np.ndarray,
    coefficients: dict[int, np.ndarray],
    higher_orders: list[int],
) -> PawulaResult:
    """Calcula R_k(x) = D^(2k)(x) / [(2k-1)!! * D^(2)(x)^k] para los ordenes 2k.

    Notas de implementacion:
        - 'higher_orders' contiene ordenes pares (>=4): por ejemplo [4, 6].
        - Se ignoran bins con D^(2) <= 0 o NaN.
    """
    if 2 not in coefficients:
        raise ValueError("Se requiere D^(2) en 'coefficients'.")
    d2 = coefficients[2]
    ratios: dict[int, np.ndarray] = {}
    summary: dict[int, dict[str, float]] = {}
    for order in higher_orders:
        if order % 2 != 0 or order < 4:
            continue
        if order not in coefficients:
            continue
        d_high = coefficients[order]
        k = order // 2
        denom = _double_factorial(2 * k - 1) * np.power(d2, k)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = d_high / denom
        r[~np.isfinite(r)] = np.nan
        ratios[order] = r
        summary[order] = {
            "mean_abs_ratio": float(np.nanmean(np.abs(r))),
            "median_abs_ratio": float(np.nanmedian(np.abs(r))),
            "max_abs_ratio": float(np.nanmax(np.abs(r))),
        }
    return PawulaResult(x_centers=x_centers, ratios=ratios, summary=summary)
