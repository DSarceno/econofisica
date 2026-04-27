# --- FILE: src/reporting/tables.py ---
"""Generacion de tablas en CSV y LaTeX (booktabs)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir


def save_table(df: pd.DataFrame, csv_path: str | Path, *, latex_path: str | Path | None = None) -> None:
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)
    df.to_csv(csv_path, index=False)
    if latex_path is not None:
        lp = Path(latex_path)
        ensure_dir(lp.parent)
        with lp.open("w", encoding="utf-8") as fh:
            fh.write(
                df.to_latex(
                    index=False,
                    float_format=lambda v: f"{v: .4g}",
                    escape=True,
                )
            )


def descriptive_table(stats_dict: dict) -> pd.DataFrame:
    rows = [
        ("n", stats_dict["n"]),
        ("mean", stats_dict["mean"]),
        ("std", stats_dict["std"]),
        ("skewness", stats_dict["skewness"]),
        ("kurtosis_excess", stats_dict["kurtosis_excess"]),
        ("min", stats_dict["min"]),
        ("max", stats_dict["max"]),
        ("JarqueBera", stats_dict["jarque_bera_stat"]),
        ("JB p-value", stats_dict["jarque_bera_p"]),
    ]
    return pd.DataFrame(rows, columns=["Estadistico", "Valor"])


def comparison_table(per_model_metrics: dict[str, dict]) -> pd.DataFrame:
    """Convierte {model: metrics_dict} en DataFrame ordenado para tesis."""
    rows = []
    for model_name, metrics in per_model_metrics.items():
        flat = {"model": model_name}
        for key, val in metrics.items():
            if isinstance(val, dict):
                for k2, v2 in val.items():
                    flat[f"{key}.{k2}"] = v2
            else:
                flat[key] = val
        rows.append(flat)
    return pd.DataFrame(rows)
