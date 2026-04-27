# --- FILE: src/preprocessing/returns.py ---
"""
Construccion de log-retornos y transformaciones derivadas.

Trazabilidad tesis -> codigo:
    - tch3 (Variables financieras): r_t = ln P_t - ln P_{t-1}.
    - tch3 (Hechos estilizados): justifica preferencia del log-retorno por
      aditividad temporal y aproximacion mesoscopica.
    - tch4 (Discretizacion): el delta_t corresponde a 1 paso (diario).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def log_returns(prices: pd.Series, *, drop_na: bool = True) -> pd.Series:
    """Calcula r_t = ln(P_t/P_{t-1}). Mantiene el indice temporal."""
    if (prices <= 0).any():
        raise ValueError("Precios no positivos no son admisibles para log-retornos.")
    r = np.log(prices).diff()
    r.name = "log_return"
    if drop_na:
        r = r.dropna()
    logger.info("log_returns: n=%d, media=%.6e, std=%.6e", len(r), r.mean(), r.std())
    return r


def standardize(
    series: pd.Series,
    *,
    method: str = "global",
    rolling_window: int = 252,
) -> pd.Series:
    """Estandariza la serie. 'global' usa media y std muestrales totales;
    'rolling' usa estadisticos moviles para mitigar heterocedasticidad lenta.
    """
    if method == "global":
        mu = series.mean()
        sigma = series.std(ddof=1)
        if sigma == 0 or np.isnan(sigma):
            raise ValueError("Desviacion estandar nula; serie degenerada.")
        return ((series - mu) / sigma).rename(f"{series.name}_std_global")
    if method == "rolling":
        mu = series.rolling(rolling_window, min_periods=rolling_window // 2).mean()
        sigma = series.rolling(rolling_window, min_periods=rolling_window // 2).std(ddof=1)
        z = (series - mu) / sigma
        return z.dropna().rename(f"{series.name}_std_roll{rolling_window}")
    raise ValueError(f"method desconocido: {method}")


def winsorize(series: pd.Series, quantiles: tuple[float, float] = (0.001, 0.999)) -> pd.Series:
    """Recorta los valores extremos a los cuantiles especificados.

    Atencion: aplicar con cautela; puede ocultar la fenomenologia de colas
    pesadas (tch3 seccion "Colas pesadas y eventos extremos"). Mantener
    DESACTIVADO por defecto y solo usar para diagnosticos secundarios.
    """
    lo, hi = series.quantile(quantiles[0]), series.quantile(quantiles[1])
    return series.clip(lower=lo, upper=hi).rename(f"{series.name}_winz")
