# --- FILE: src/models/base.py ---
"""Interfaz comun para modelos estocasticos compatibles con Euler-Maruyama.

Trazabilidad tesis -> codigo:
    - tch1 (Langevin): dx = D^(1)(x,t) dt + sqrt(2 D^(2)(x,t)) dW_t
    - tch protocolo seccion 4.1.11: M0 (gaussiano), M1 (OU), M2 (reconstruido).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class StochasticModel(ABC):
    """Interfaz minima para SDE escalares dx = drift(x,t)dt + sqrt(2*diff(x,t)) dW.

    Subclases deben implementar:
        - drift(x, t): coeficiente D^(1)
        - diffusion(x, t): coeficiente D^(2) (>= 0)
    Opcionalmente:
        - fit(series): ajuste a datos
        - sample_initial(n, rng): condicion inicial natural
    """

    name: str = "base"

    @abstractmethod
    def drift(self, x: np.ndarray, t: float) -> np.ndarray:
        ...

    @abstractmethod
    def diffusion(self, x: np.ndarray, t: float) -> np.ndarray:
        ...

    def stationary_sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Por defecto: condicion inicial gaussiana (puede ser sobreescrita)."""
        return rng.standard_normal(n)

    def to_dict(self) -> dict:
        return {"name": self.name}
