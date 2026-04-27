# --- FILE: src/utils/io.py ---
"""IO con metadatos: cada artefacto persistido lleva snapshot de config + hash."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def hash_array(arr: np.ndarray | pd.Series | pd.DataFrame) -> str:
    """SHA-256 del contenido binario; estable entre corridas."""
    if isinstance(arr, (pd.Series, pd.DataFrame)):
        arr = arr.to_numpy()
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    return h.hexdigest()


def _coerce_for_json(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (pd.Timestamp, dt.date, dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"No serializable: {type(obj)}")


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_coerce_for_json)
    return p


def save_dataframe(df: pd.DataFrame, path: str | Path, *, index: bool = True) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    if p.suffix == ".parquet":
        df.to_parquet(p, index=index)
    elif p.suffix == ".csv":
        df.to_csv(p, index=index)
    else:
        raise ValueError(f"Extension no soportada: {p.suffix}")
    return p


def save_array(arr: np.ndarray, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    np.save(p, arr)
    return p


def snapshot_config(config: BaseModel, output_dir: str | Path) -> Path:
    """Persiste copia inmutable del config junto al output."""
    out = ensure_dir(output_dir)
    return save_json(
        {
            "snapshot_at": dt.datetime.utcnow().isoformat() + "Z",
            "config": config.model_dump(),
        },
        out / "config_snapshot.json",
    )
