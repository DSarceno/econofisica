# --- FILE: src/visualization/style.py ---
"""Estilo grafico unificado para todas las figuras."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_thesis_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (7.0, 4.5),
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "lines.linewidth": 1.4,
        }
    )
    # Color cycle accesible
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=["#1f3a93", "#c0392b", "#16a085", "#8e44ad", "#d35400", "#2c3e50"]
    )
