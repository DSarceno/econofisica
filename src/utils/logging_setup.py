# --- FILE: src/utils/logging_setup.py ---
"""Configuracion centralizada de logging usando logging.config + YAML."""
from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml


def setup_logging(
    config_path: str | Path = "configs/logging.yaml",
    default_level: int = logging.INFO,
) -> logging.Logger:
    """Inicializa el sistema de logging desde un YAML; fallback basico si falta."""
    cfg_path = Path(config_path)
    if cfg_path.exists():
        # Asegurar que el directorio de logs exista
        Path("logs").mkdir(parents=True, exist_ok=True)
        with cfg_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        logging.config.dictConfig(cfg)
    else:
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    return logging.getLogger("econophys")
