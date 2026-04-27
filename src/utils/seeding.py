# --- FILE: src/utils/seeding.py ---
"""Sembrado reproducible para numpy y RNGs derivados."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Fija las semillas globales y retorna un Generator dedicado."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def derived_generator(parent_seed: int, stream_id: int) -> np.random.Generator:
    """Genera un RNG independiente derivado de (parent_seed, stream_id).

    Util para paralelizar simulaciones con trazabilidad: cada path obtiene su
    propia secuencia, garantizando reproducibilidad bit-exacta.
    """
    ss = np.random.SeedSequence((parent_seed, stream_id))
    return np.random.default_rng(ss)
