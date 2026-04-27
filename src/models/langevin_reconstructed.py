# --- FILE: src/models/langevin_reconstructed.py ---
"""
Modelo M2: Langevin reconstruido empiricamente desde coeficientes de KM.

Trazabilidad tesis -> codigo:
    - tch1 / tch2: Langevin con D^(1)(x), D^(2)(x) dependientes del estado.
    - tch4: estimacion no parametrica.
    - protocolo seccion 4.1.11.3: M2.

Construccion:
    - Recibe el resultado KMEstimate y construye interpoladores 1D para
      D^(1)(x) y D^(2)(x).
    - Aplica un piso 'diffusion_floor' para evitar D^(2) <= 0 en regiones
      con poca poblacion.
    - Fuera del soporte observado, extrapola con valores extremos (clipping).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline, interp1d

from src.kramers_moyal.estimators import KMEstimate

from .base import StochasticModel


@dataclass
class LangevinReconstructed(StochasticModel):
    x_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    d1_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    d2_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    diffusion_floor: float = 1e-8
    drift_kind: str = "cubic"
    diffusion_kind: str = "cubic"
    name: str = "M2_langevin_reconstructed"
    _drift_fn: object = field(default=None, repr=False)
    _diff_fn: object = field(default=None, repr=False)
    _x_min: float = 0.0
    _x_max: float = 0.0

    def __post_init__(self) -> None:
        if self.x_grid.size == 0:
            return
        self._build_interpolators()

    def _build_interpolators(self) -> None:
        # Eliminar NaN antes de interpolar
        mask = (
            np.isfinite(self.x_grid)
            & np.isfinite(self.d1_grid)
            & np.isfinite(self.d2_grid)
        )
        xg = self.x_grid[mask]
        d1 = self.d1_grid[mask]
        d2 = np.maximum(self.d2_grid[mask], self.diffusion_floor)
        order = np.argsort(xg)
        xg, d1, d2 = xg[order], d1[order], d2[order]
        if xg.size < 4:
            kind = "linear"
        else:
            kind = self.drift_kind
        if kind == "cubic" and xg.size >= 4:
            self._drift_fn = CubicSpline(xg, d1, extrapolate=True)
        else:
            self._drift_fn = interp1d(xg, d1, kind="linear", fill_value="extrapolate")
        if self.diffusion_kind == "cubic" and xg.size >= 4:
            self._diff_fn = CubicSpline(xg, d2, extrapolate=True)
        else:
            self._diff_fn = interp1d(xg, d2, kind="linear", fill_value="extrapolate")
        self._x_min = float(xg.min())
        self._x_max = float(xg.max())

    @classmethod
    def from_km_estimate(
        cls,
        km: KMEstimate,
        *,
        diffusion_floor: float = 1e-8,
        drift_kind: str = "cubic",
        diffusion_kind: str = "cubic",
    ) -> "LangevinReconstructed":
        if 1 not in km.coefficients or 2 not in km.coefficients:
            raise ValueError("KMEstimate debe contener D^(1) y D^(2).")
        return cls(
            x_grid=km.x_centers,
            d1_grid=km.coefficients[1],
            d2_grid=km.coefficients[2],
            diffusion_floor=diffusion_floor,
            drift_kind=drift_kind,
            diffusion_kind=diffusion_kind,
        )

    def _eval(self, fn, x: np.ndarray) -> np.ndarray:
        x_clip = np.clip(x, self._x_min, self._x_max)
        return np.asarray(fn(x_clip), dtype=np.float64)

    def drift(self, x: np.ndarray, t: float) -> np.ndarray:
        return self._eval(self._drift_fn, x)

    def diffusion(self, x: np.ndarray, t: float) -> np.ndarray:
        d2 = self._eval(self._diff_fn, x)
        return np.maximum(d2, self.diffusion_floor)

    def potential(self, x_grid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruye U(x) por integracion: dU/dx = -D^(1)(x).

        Atencion (apendice): bajo difusion dependiente del estado,
        la relacion exacta es D^(1) = -dU/dx + dD^(2)/dx. Aqui retornamos la
        version sin correccion como diagnostico visual; usar 'effective_potential'
        para la correccion completa.
        """
        if x_grid is None:
            x_grid = np.linspace(self._x_min, self._x_max, 200)
        d1 = self.drift(x_grid, 0.0)
        # Integracion trapezoidal acumulativa
        dx = np.diff(x_grid, prepend=x_grid[0])
        U = -np.cumsum(d1 * dx)
        U -= U.min()
        return x_grid, U

    def effective_potential(self, x_grid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """U_eff(x) usando la correccion por difusion dependiente del estado."""
        if x_grid is None:
            x_grid = np.linspace(self._x_min, self._x_max, 400)
        d1 = self.drift(x_grid, 0.0)
        d2 = self.diffusion(x_grid, 0.0)
        dD2_dx = np.gradient(d2, x_grid)
        force_eff = d1 - dD2_dx
        dx = np.diff(x_grid, prepend=x_grid[0])
        U = -np.cumsum(force_eff * dx)
        U -= U.min()
        return x_grid, U

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "x_min": self._x_min,
            "x_max": self._x_max,
            "n_grid": int(self.x_grid.size),
            "diffusion_floor": self.diffusion_floor,
            "drift_kind": self.drift_kind,
            "diffusion_kind": self.diffusion_kind,
        }
