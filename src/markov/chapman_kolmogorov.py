# --- FILE: src/markov/chapman_kolmogorov.py ---
"""
Test empirico de Chapman-Kolmogorov para validar la hipotesis markoviana.

Trazabilidad tesis -> codigo:
    - tch2 (Procesos de Markov, Chapman-Kolmogorov):
          P(x3,t3|x1,t1) = integral dx2 P(x3,t3|x2,t2) P(x2,t2|x1,t1)
    - tch4 (Verificacion Markovianidad): se evalua comparando la PDF
      condicional empirica directa a 2 pasos con la composicion de pasos
      consecutivos.
    - apendice (Chapman-Kolmogorov): forma integral.

Procedimiento:
    1. Discretizar el espacio de estados en N bins (cuantiles o uniforme).
    2. Estimar matriz de transicion T_1 a paso 1 y T_2 a paso 2 via histogramas.
    3. Comparar T_2 vs T_1 @ T_1 con divergencia (KL o Jensen-Shannon).
    4. Bootstrap por bloques para construir intervalos de confianza.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CKTestResult:
    delta_t_steps: int
    n_bins: int
    bin_edges: np.ndarray
    transition_direct: np.ndarray         # T_{2*dt} estimada directamente
    transition_composed: np.ndarray       # T_dt @ T_dt
    divergence_per_row: np.ndarray        # divergencia por estado origen
    mean_divergence: float
    divergence_kind: str
    bootstrap_ci: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        return {
            "delta_t_steps": self.delta_t_steps,
            "n_bins": self.n_bins,
            "mean_divergence": self.mean_divergence,
            "divergence_kind": self.divergence_kind,
            "bootstrap_ci": self.bootstrap_ci,
        }


def _make_edges(x: np.ndarray, n_bins: int, *, scheme: str = "quantile") -> np.ndarray:
    if scheme == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(x, qs)
        edges[0] -= 1e-12
        edges[-1] += 1e-12
        return edges
    if scheme == "uniform":
        return np.linspace(x.min() - 1e-12, x.max() + 1e-12, n_bins + 1)
    raise ValueError(scheme)


def _transition_matrix(x: np.ndarray, edges: np.ndarray, lag: int) -> np.ndarray:
    """Estima la matriz de transicion P(x_{t+lag} | x_t) por conteo en bins."""
    n_bins = edges.size - 1
    src = np.digitize(x[:-lag], edges) - 1
    dst = np.digitize(x[lag:], edges) - 1
    src = np.clip(src, 0, n_bins - 1)
    dst = np.clip(dst, 0, n_bins - 1)
    counts = np.zeros((n_bins, n_bins), dtype=np.float64)
    np.add.at(counts, (src, dst), 1.0)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def _js(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (_kl(p, m) + _kl(q, m)))


def chapman_kolmogorov_test(
    series: pd.Series,
    *,
    delta_t_steps: int = 2,
    n_bins: int = 25,
    binning: str = "quantile",
    divergence: str = "jensen_shannon",
    bootstrap_samples: int = 0,
    bootstrap_confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> CKTestResult:
    """Ejecuta el test de Chapman-Kolmogorov para 1 paso vs delta_t_steps pasos.

    El parametro delta_t_steps representa el lag total a evaluar (ej. 2 -> compara
    T_2 directa con T_1 @ T_1; con 4 -> T_4 vs T_2 @ T_2 si lo deseas adaptar).
    """
    if delta_t_steps < 2:
        raise ValueError("delta_t_steps debe ser >= 2")
    x = series.dropna().to_numpy()
    edges = _make_edges(x, n_bins=n_bins, scheme=binning)

    half = delta_t_steps // 2
    T_short = _transition_matrix(x, edges, lag=half)
    T_direct = _transition_matrix(x, edges, lag=delta_t_steps)
    T_composed = np.linalg.matrix_power(T_short, 2)

    divfn = _js if divergence == "jensen_shannon" else _kl
    div_per_row = np.array(
        [divfn(T_direct[i], T_composed[i]) for i in range(T_direct.shape[0])]
    )
    mean_div = float(np.mean(div_per_row))

    ci: tuple[float, float] | None = None
    if bootstrap_samples > 0:
        rng = rng or np.random.default_rng()
        boot = np.empty(bootstrap_samples)
        n = x.size
        block = max(50, n // 100)
        for b in range(bootstrap_samples):
            # Bootstrap de bloques moviles (estaciona la dependencia temporal)
            n_blocks = n // block + 1
            starts = rng.integers(0, n - block, size=n_blocks)
            xb = np.concatenate([x[s : s + block] for s in starts])[:n]
            T_s = _transition_matrix(xb, edges, lag=half)
            T_d = _transition_matrix(xb, edges, lag=delta_t_steps)
            T_c = np.linalg.matrix_power(T_s, 2)
            boot[b] = np.mean([divfn(T_d[i], T_c[i]) for i in range(T_d.shape[0])])
        alpha = (1.0 - bootstrap_confidence) / 2.0
        ci = (float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha)))

    logger.info(
        "CK-test: dt=%d nbins=%d div=%s mean=%.4e CI=%s",
        delta_t_steps, n_bins, divergence, mean_div, ci,
    )
    return CKTestResult(
        delta_t_steps=delta_t_steps,
        n_bins=n_bins,
        bin_edges=edges,
        transition_direct=T_direct,
        transition_composed=T_composed,
        divergence_per_row=div_per_row,
        mean_divergence=mean_div,
        divergence_kind=divergence,
        bootstrap_ci=ci,
    )
