# --- FILE: src/validation/metrics.py ---
"""
Metricas de comparacion datos vs simulacion.

Trazabilidad tesis -> codigo:
    - protocolo seccion 4.7 (Comparacion de modelos) y 4.1.13.
    - Metricas: KS, Jensen-Shannon, KL, MSE de ACF^2, momentos, indice de Hill.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import acf as sm_acf

from src.statistics.descriptive import hill_tail_index


def ks_distance(empirical: np.ndarray, simulated: np.ndarray) -> dict[str, float]:
    stat, pvalue = stats.ks_2samp(empirical, simulated)
    return {"ks_stat": float(stat), "ks_pvalue": float(pvalue)}


def _normalize_hist(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(x, bins=edges, density=False)
    h = h.astype(np.float64) + 1e-12
    h /= h.sum()
    return h


def js_divergence(empirical: np.ndarray, simulated: np.ndarray, *, bins: int = 100) -> float:
    lo = min(np.min(empirical), np.min(simulated))
    hi = max(np.max(empirical), np.max(simulated))
    edges = np.linspace(lo, hi, bins + 1)
    p = _normalize_hist(empirical, edges)
    q = _normalize_hist(simulated, edges)
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


def kl_divergence(empirical: np.ndarray, simulated: np.ndarray, *, bins: int = 100) -> float:
    lo = min(np.min(empirical), np.min(simulated))
    hi = max(np.max(empirical), np.max(simulated))
    edges = np.linspace(lo, hi, bins + 1)
    p = _normalize_hist(empirical, edges)
    q = _normalize_hist(simulated, edges)
    return float(np.sum(p * np.log(p / q)))


def moment_errors(empirical: np.ndarray, simulated: np.ndarray) -> dict[str, float]:
    return {
        "mean_abs_err": float(abs(np.mean(empirical) - np.mean(simulated))),
        "std_abs_err": float(abs(np.std(empirical, ddof=1) - np.std(simulated, ddof=1))),
        "skew_abs_err": float(abs(stats.skew(empirical) - stats.skew(simulated))),
        "kurt_abs_err": float(
            abs(stats.kurtosis(empirical, fisher=True) - stats.kurtosis(simulated, fisher=True))
        ),
    }


def acf_squared_mse(empirical: np.ndarray, simulated: np.ndarray, nlags: int = 30) -> float:
    a = sm_acf(empirical ** 2, nlags=nlags, fft=True)
    b = sm_acf(simulated ** 2, nlags=nlags, fft=True)
    return float(np.mean((a - b) ** 2))


def hill_diff(empirical: np.ndarray, simulated: np.ndarray, *, k_frac: float = 0.05) -> float:
    import pandas as pd
    n_e, n_s = empirical.size, simulated.size
    k_e = max(20, int(k_frac * n_e))
    k_s = max(20, int(k_frac * n_s))
    he = hill_tail_index(pd.Series(empirical), k=k_e)
    hs = hill_tail_index(pd.Series(simulated), k=k_s)
    return float(abs(he - hs))


def compare_all(empirical: np.ndarray, simulated: np.ndarray, *, bins: int = 100) -> dict:
    res: dict[str, float | dict] = {}
    res.update(ks_distance(empirical, simulated))
    res["jensen_shannon"] = js_divergence(empirical, simulated, bins=bins)
    res["kl_divergence"] = kl_divergence(empirical, simulated, bins=bins)
    res["moments"] = moment_errors(empirical, simulated)
    res["acf_squared_mse"] = acf_squared_mse(empirical, simulated)
    try:
        res["hill_diff"] = hill_diff(empirical, simulated)
    except ValueError:
        res["hill_diff"] = float("nan")
    return res
