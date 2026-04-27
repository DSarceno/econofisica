# --- FILE: src/reporting/markdown_report.py ---
"""Reporte Markdown ejecutivo de la corrida del pipeline."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from src.utils.io import ensure_dir


def render_report(
    *,
    out_path: str | Path,
    config_snapshot: dict,
    descriptive: dict,
    ck_results: list[dict],
    km_summary: dict,
    pawula_summary: dict,
    comparison: dict,
    figure_paths: dict[str, str],
) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    now = dt.datetime.utcnow().isoformat() + "Z"

    md: list[str] = []
    md.append(f"# Reporte de pipeline — {config_snapshot['project']['name']}\n")
    md.append(f"_Generado:_ `{now}`\n")
    md.append("## 1. Configuracion\n")
    md.append(f"- Ticker: `{config_snapshot['data']['ticker']}`")
    md.append(f"- Periodo: {config_snapshot['data']['start']} — {config_snapshot['data']['end']}")
    md.append(f"- Semilla: {config_snapshot['project']['random_seed']}\n")

    md.append("## 2. Estadistica descriptiva\n")
    md.append("| Estadistico | Valor |\n|---|---|")
    for k in ["n", "mean", "std", "skewness", "kurtosis_excess", "min", "max"]:
        md.append(f"| {k} | {descriptive[k]:.6g} |")
    md.append(f"| Jarque-Bera p | {descriptive['jarque_bera_p']:.3e} |\n")

    md.append("## 3. Test de Markov (Chapman-Kolmogorov)\n")
    md.append("| dt | divergencia media | bootstrap CI |\n|---|---|---|")
    for r in ck_results:
        ci = r.get("bootstrap_ci")
        ci_str = f"[{ci[0]:.3e}, {ci[1]:.3e}]" if ci else "n/a"
        md.append(f"| {r['delta_t_steps']} | {r['mean_divergence']:.3e} | {ci_str} |")

    md.append("\n## 4. Coeficientes de Kramers-Moyal (resumen)\n")
    md.append(f"- delta_t: {km_summary['delta_t']}, n_bins: {km_summary['n_bins']}")
    md.append(f"- n_used: {km_summary['n_used']}\n")

    md.append("## 5. Test de Pawula\n")
    if pawula_summary:
        md.append("| Orden | mean(|R_k|) | median | max |\n|---|---|---|---|")
        for order, s in pawula_summary.items():
            md.append(
                f"| {order} | {s['mean_abs_ratio']:.3e} | {s['median_abs_ratio']:.3e} | {s['max_abs_ratio']:.3e} |"
            )
    md.append("")

    md.append("## 6. Comparacion de modelos (vs empirico)\n")
    md.append("| Modelo | KS | JS | err mean | err std | err skew | err kurt |\n|---|---|---|---|---|---|---|")
    for name, m in comparison.items():
        moments = m.get("moments", {})
        md.append(
            f"| {name} | {m.get('ks_stat', float('nan')):.3e} | {m.get('jensen_shannon', float('nan')):.3e} | "
            f"{moments.get('mean_abs_err', float('nan')):.3e} | {moments.get('std_abs_err', float('nan')):.3e} | "
            f"{moments.get('skew_abs_err', float('nan')):.3e} | {moments.get('kurt_abs_err', float('nan')):.3e} |"
        )

    md.append("\n## 7. Figuras\n")
    for k, p in figure_paths.items():
        md.append(f"- **{k}**: `{p}`")

    out_path.write_text("\n".join(md), encoding="utf-8")
    return out_path
