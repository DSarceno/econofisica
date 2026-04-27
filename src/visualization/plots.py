# --- FILE: src/visualization/plots.py ---
"""
Figuras estandar usadas en la tesis.

Cada funcion guarda y retorna la ruta del archivo. La paleta y estilo provienen
de visualization.style.apply_thesis_style().
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf as sm_acf

from src.utils.io import ensure_dir


def _save(fig, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_price_and_returns(
    prices: pd.Series,
    returns: pd.Series,
    out: str | Path,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    axes[0].plot(prices.index, prices.values)
    axes[0].set_ylabel("Precio (Adj. Close)")
    axes[0].set_title("S&P 500 — Precio y log-retornos")
    axes[1].plot(returns.index, returns.values, lw=0.6)
    axes[1].set_ylabel(r"$r_t = \ln P_t - \ln P_{t-1}$")
    axes[1].set_xlabel("Fecha")
    fig.tight_layout()
    return _save(fig, out)


def plot_pdf_vs_normal(
    returns: pd.Series,
    out: str | Path,
    *,
    bins: int = 100,
) -> Path:
    x = returns.dropna().to_numpy()
    fig, ax = plt.subplots()
    ax.hist(x, bins=bins, density=True, alpha=0.45, label="Empirica")
    grid = np.linspace(x.min(), x.max(), 400)
    pdf = stats.norm.pdf(grid, loc=x.mean(), scale=x.std(ddof=1))
    ax.plot(grid, pdf, lw=2.0, label="N(mu, sigma^2)")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r_t$")
    ax.set_ylabel("Densidad (log)")
    ax.set_title("PDF empirica vs gaussiana — colas pesadas")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out)


def plot_acf(
    returns: pd.Series,
    out: str | Path,
    *,
    nlags: int = 60,
) -> Path:
    a1 = sm_acf(returns.dropna().to_numpy(), nlags=nlags, fft=True)
    a2 = sm_acf(returns.dropna().to_numpy() ** 2, nlags=nlags, fft=True)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    axes[0].vlines(range(nlags + 1), 0, a1)
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].set_ylabel(r"ACF $r_t$")
    axes[0].set_title("Autocorrelacion: retornos vs retornos al cuadrado")
    axes[1].vlines(range(nlags + 1), 0, a2)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_ylabel(r"ACF $r_t^2$")
    axes[1].set_xlabel("lag")
    fig.tight_layout()
    return _save(fig, out)


def plot_km_coefficients(
    x_centers: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    out: str | Path,
    *,
    d1_se: np.ndarray | None = None,
    d2_se: np.ndarray | None = None,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(x_centers, d1, marker="o", ms=3)
    if d1_se is not None:
        axes[0].fill_between(x_centers, d1 - d1_se, d1 + d1_se, alpha=0.2)
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_xlabel("x (log-retorno)")
    axes[0].set_ylabel(r"$D^{(1)}(x)$")
    axes[0].set_title("Drift")

    axes[1].plot(x_centers, d2, marker="o", ms=3, color="#c0392b")
    if d2_se is not None:
        axes[1].fill_between(x_centers, d2 - d2_se, d2 + d2_se, alpha=0.2, color="#c0392b")
    axes[1].set_xlabel("x (log-retorno)")
    axes[1].set_ylabel(r"$D^{(2)}(x)$")
    axes[1].set_title("Difusion")
    fig.tight_layout()
    return _save(fig, out)


def plot_potential(
    x_grid: np.ndarray,
    U: np.ndarray,
    out: str | Path,
    *,
    title: str = "Potencial efectivo",
) -> Path:
    fig, ax = plt.subplots()
    ax.plot(x_grid, U)
    ax.set_xlabel("x")
    ax.set_ylabel("U(x)")
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, out)


def plot_simulated_vs_empirical(
    empirical: np.ndarray,
    simulated: np.ndarray,
    out: str | Path,
    *,
    bins: int = 100,
    label_sim: str = "Simulado",
) -> Path:
    fig, ax = plt.subplots()
    ax.hist(empirical, bins=bins, density=True, alpha=0.45, label="Empirico")
    ax.hist(simulated, bins=bins, density=True, alpha=0.45, label=label_sim)
    ax.set_yscale("log")
    ax.set_xlabel(r"$r_t$")
    ax.set_ylabel("Densidad (log)")
    ax.set_title("Comparacion: empirico vs simulado")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out)


def plot_chapman_kolmogorov(
    bin_edges: np.ndarray,
    direct: np.ndarray,
    composed: np.ndarray,
    out: str | Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(direct, origin="lower", aspect="auto", cmap="viridis")
    axes[0].set_title("T directa")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(composed, origin="lower", aspect="auto", cmap="viridis")
    axes[1].set_title("T compuesta (T_dt @ T_dt)")
    plt.colorbar(im1, ax=axes[1])
    fig.suptitle("Test Chapman-Kolmogorov: matrices de transicion")
    fig.tight_layout()
    return _save(fig, out)


def plot_rolling_metrics(rolling_df: pd.DataFrame, out: str | Path) -> Path:
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    if "t_start" in rolling_df.columns:
        t = pd.to_datetime(rolling_df["t_start"])
    else:
        t = rolling_df.index
    axes[0].plot(t, rolling_df["slope_d1"])
    axes[0].set_ylabel(r"slope $D^{(1)}$")
    axes[0].set_title("Diagnostico por ventanas")
    axes[1].plot(t, rolling_df["mean_d2"], color="#c0392b")
    axes[1].set_ylabel(r"$\langle D^{(2)} \rangle$")
    axes[2].plot(t, rolling_df["asym_d1"], color="#16a085")
    axes[2].set_ylabel("Asimetria drift")
    axes[2].set_xlabel("Tiempo")
    fig.tight_layout()
    return _save(fig, out)
