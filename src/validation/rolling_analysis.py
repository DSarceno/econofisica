# --- FILE: src/validation/rolling_analysis.py ---
"""
Analisis por ventanas temporales: evidencia de no-estacionariedad / no-equilibrio.

Trazabilidad tesis -> codigo:
    - protocolo seccion 4.1.14 (Evaluacion temporal) y 4.1.15 (No-equilibrio).
    - tch3 (No estacionariedad y cambios de regimen).

Idea: deslizar una ventana de tamano W con paso S; en cada ventana reestimar
los coeficientes K-M y reportar como cambian los descriptores agregados:
    - mean_drift_slope (signo y magnitud de la pendiente local de D^(1))
    - mean_diffusion (nivel de difusion)
    - asymmetry_drift (asimetria del drift respecto al origen)
    - probability_current_proxy (cuando se solicite)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.kramers_moyal.estimators import estimate_km_coefficients


@dataclass
class RollingResult:
    centers_time: list[pd.Timestamp]
    metrics: pd.DataFrame


def _summarize_km(x_centers: np.ndarray, d1: np.ndarray, d2: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(d1) & np.isfinite(d2)
    if mask.sum() < 5:
        return {"slope_d1": np.nan, "mean_d2": np.nan, "asym_d1": np.nan}
    xc = x_centers[mask]
    d1m = d1[mask]
    d2m = d2[mask]
    slope = float(np.polyfit(xc, d1m, 1)[0])
    mean_d2 = float(np.mean(d2m))
    asym = float(np.mean(d1m * np.sign(xc)))
    return {"slope_d1": slope, "mean_d2": mean_d2, "asym_d1": asym}


def rolling_km(
    series: pd.Series,
    *,
    window_size: int,
    step: int,
    n_bins: int = 30,
    min_obs_per_bin: int = 20,
    delta_t_step: int = 1,
) -> RollingResult:
    """Estima KM en ventanas deslizantes y resume cada estimacion."""
    if window_size > len(series):
        raise ValueError("window_size > len(series)")
    rows: list[dict] = []
    centers: list[pd.Timestamp] = []
    for start in range(0, len(series) - window_size + 1, step):
        end = start + window_size
        sub = series.iloc[start:end]
        try:
            km = estimate_km_coefficients(
                sub,
                orders=[1, 2],
                n_bins=n_bins,
                binning="quantile",
                delta_t_step=delta_t_step,
                min_obs_per_bin=min_obs_per_bin,
            )
        except ValueError:
            continue
        summary = _summarize_km(km.x_centers, km.coefficients[1], km.coefficients[2])
        summary["t_start"] = sub.index[0]
        summary["t_end"] = sub.index[-1]
        rows.append(summary)
        centers.append(sub.index[len(sub) // 2])

    df = pd.DataFrame(rows)
    return RollingResult(centers_time=centers, metrics=df)


def probability_current_proxy(
    x_centers: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    pdf: np.ndarray,
) -> np.ndarray:
    """
    Estima J(x) = D^(1)(x) P(x) - d/dx [D^(2)(x) P(x)] como proxy.

    Trazabilidad: tch1 / apendice (Corriente de probabilidad).
    En estado estacionario verdadero J = 0; desviaciones sistematicas evidencian
    flujos no nulos -> sistema fuera del equilibrio detallado.
    """
    flux_diff = np.gradient(d2 * pdf, x_centers)
    return d1 * pdf - flux_diff
