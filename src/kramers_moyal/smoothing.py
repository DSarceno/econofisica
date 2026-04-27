# --- FILE: src/kramers_moyal/smoothing.py ---
"""Suavizado de coeficientes de KM (Savitzky-Golay y media movil)."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def smooth(values: np.ndarray, *, method: str, window: int, polyorder: int) -> np.ndarray:
    """Suaviza ignorando NaN: rellena con interpolacion lineal -> aplica filtro -> reintroduce NaN."""
    if method == "none":
        return values
    nan_mask = np.isnan(values)
    if nan_mask.all():
        return values
    idx = np.arange(values.size)
    valid = ~nan_mask
    interp = np.interp(idx, idx[valid], values[valid])
    if method == "savgol":
        w = max(3, window if window % 2 == 1 else window + 1)
        w = min(w, interp.size - 1 if interp.size % 2 == 0 else interp.size)
        if w < 3:
            return values
        out = savgol_filter(interp, window_length=w, polyorder=min(polyorder, w - 1))
    elif method == "moving_average":
        kernel = np.ones(window) / window
        out = np.convolve(interp, kernel, mode="same")
    else:
        raise ValueError(f"Metodo de suavizado desconocido: {method}")
    out[nan_mask] = np.nan
    return out
