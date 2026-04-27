# --- FILE: src/statistics/descriptive.py ---
"""
Estadistica descriptiva: PDF empirica, momentos, ACF, indice de cola (Hill).

Trazabilidad tesis -> codigo:
    - tch3 (Hechos estilizados): no-gaussianidad, colas pesadas, asimetria,
      exceso de curtosis, clustering de volatilidad, dependencia temporal.
    - protocolo seccion 4.3: PDF, momentos, ACF.
    - protocolo seccion 6.4 / 6.6 (cola): indice tail Hill.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf as sm_acf


@dataclass
class DescriptiveStats:
    n: int
    mean: float
    std: float
    skewness: float
    kurtosis_excess: float  # exceso (Fisher)
    min: float
    max: float
    quantiles: dict[float, float] = field(default_factory=dict)
    jarque_bera_stat: float = float("nan")
    jarque_bera_p: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "skewness": self.skewness,
            "kurtosis_excess": self.kurtosis_excess,
            "min": self.min,
            "max": self.max,
            "quantiles": self.quantiles,
            "jarque_bera_stat": self.jarque_bera_stat,
            "jarque_bera_p": self.jarque_bera_p,
        }


def describe(series: pd.Series) -> DescriptiveStats:
    """Estadisticos resumen de una serie temporal de retornos."""
    x = series.dropna().to_numpy()
    qs = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    jb_stat, jb_p = stats.jarque_bera(x)
    return DescriptiveStats(
        n=int(x.size),
        mean=float(np.mean(x)),
        std=float(np.std(x, ddof=1)),
        skewness=float(stats.skew(x)),
        kurtosis_excess=float(stats.kurtosis(x, fisher=True)),
        min=float(np.min(x)),
        max=float(np.max(x)),
        quantiles={q: float(np.quantile(x, q)) for q in qs},
        jarque_bera_stat=float(jb_stat),
        jarque_bera_p=float(jb_p),
    )


def empirical_pdf(
    series: pd.Series,
    bins: int = 100,
    range_: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Histograma normalizado: retorna (centros_bin, densidad)."""
    x = series.dropna().to_numpy()
    counts, edges = np.histogram(x, bins=bins, range=range_, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def autocorrelation(series: pd.Series, nlags: int = 60) -> np.ndarray:
    """ACF de la serie cruda. Para clustering de volatilidad usar squared_acf."""
    x = series.dropna().to_numpy()
    return sm_acf(x, nlags=nlags, fft=True)


def squared_autocorrelation(series: pd.Series, nlags: int = 60) -> np.ndarray:
    """ACF de r_t^2: detecta clustering de volatilidad (tch3 hecho estilizado)."""
    x = series.dropna().to_numpy() ** 2
    return sm_acf(x, nlags=nlags, fft=True)


def hill_tail_index(series: pd.Series, k: int) -> float:
    """Estimador de Hill del exponente de cola (asume cola derecha, |r|).

    Para una distribucion P(|r|>x) ~ x^{-alpha}, el estimador es:
        alpha_hat = 1 / [ (1/k) sum_{i=1}^{k} log(X_(n-i+1)/X_(n-k)) ]
    donde X_(.) son los estadisticos de orden ascendentes de |r|.
    Referencia tesis: tch3 seccion "Colas pesadas y eventos extremos".
    """
    x = np.sort(np.abs(series.dropna().to_numpy()))
    n = x.size
    if k <= 1 or k >= n:
        raise ValueError(f"k={k} fuera de rango (1, n)={n}")
    threshold = x[n - k - 1]
    if threshold <= 0:
        raise ValueError("Umbral no positivo; verifique los datos.")
    tail = x[n - k:]
    return float(1.0 / np.mean(np.log(tail / threshold)))
