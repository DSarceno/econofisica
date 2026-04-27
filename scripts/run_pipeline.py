# --- FILE: scripts/run_pipeline.py ---
"""
Orquestador end-to-end del pipeline de reconstruccion estocastica.

Stages:
    1. Cargar configuracion y semillas.
    2. Adquirir y persistir datos crudos.
    3. Construir log-retornos (y opcionalmente estandarizar).
    4. Estadistica descriptiva + figuras (PDF, ACF).
    5. Test de Markov (Chapman-Kolmogorov) sobre delta_t configurable.
    6. Estimar coeficientes de Kramers-Moyal + Pawula.
    7. Ajustar modelos M0, M1, M2.
    8. Simular trayectorias bajo Euler-Maruyama.
    9. Validar (KS, JS, momentos, ACF^2, Hill) por modelo.
   10. Analisis por ventanas (no-equilibrio).
   11. Generar reporte Markdown + tablas LaTeX.

Cada stage es invocable de forma aislada via 'scripts/stages/<stage>.py'.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Asegurar que el root del proyecto este en el path al ejecutar como script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import load_prices  # noqa: E402
from src.kramers_moyal.estimators import estimate_km_coefficients  # noqa: E402
from src.kramers_moyal.pawula import pawula_ratios  # noqa: E402
from src.kramers_moyal.smoothing import smooth  # noqa: E402
from src.markov.chapman_kolmogorov import chapman_kolmogorov_test  # noqa: E402
from src.models.baseline_gaussian import GaussianBaseline  # noqa: E402
from src.models.langevin_reconstructed import LangevinReconstructed  # noqa: E402
from src.models.ornstein_uhlenbeck import OrnsteinUhlenbeck  # noqa: E402
from src.preprocessing.returns import log_returns, standardize, winsorize  # noqa: E402
from src.reporting.markdown_report import render_report  # noqa: E402
from src.reporting.tables import comparison_table, descriptive_table, save_table  # noqa: E402
from src.simulation.euler_maruyama import simulate_paths  # noqa: E402
from src.statistics.descriptive import describe  # noqa: E402
from src.utils.config import PipelineConfig, load_config  # noqa: E402
from src.utils.io import ensure_dir, save_dataframe, save_json, snapshot_config  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402
from src.validation.metrics import compare_all  # noqa: E402
from src.validation.rolling_analysis import rolling_km  # noqa: E402
from src.visualization.plots import (  # noqa: E402
    plot_acf,
    plot_chapman_kolmogorov,
    plot_km_coefficients,
    plot_pdf_vs_normal,
    plot_potential,
    plot_price_and_returns,
    plot_rolling_metrics,
    plot_simulated_vs_empirical,
)
from src.visualization.style import apply_thesis_style  # noqa: E402

logger = logging.getLogger("econophys.pipeline")


def _prepare_returns(prices: pd.Series, cfg: PipelineConfig) -> tuple[pd.Series, pd.Series]:
    r = log_returns(prices, drop_na=cfg.preprocessing.drop_na)
    if cfg.preprocessing.winsorize.enabled:
        r = winsorize(r, cfg.preprocessing.winsorize.quantiles)
    if cfg.preprocessing.standardize.enabled:
        z = standardize(
            r,
            method=cfg.preprocessing.standardize.method,
            rolling_window=cfg.preprocessing.standardize.rolling_window,
        )
    else:
        z = r.copy().rename("returns_raw")
    return r, z


def _run_ck_tests(series: pd.Series, cfg: PipelineConfig, rng: np.random.Generator) -> list[dict]:
    out: list[dict] = []
    for dt in cfg.markov_test.delta_t_steps:
        if dt < 2:
            continue
        res = chapman_kolmogorov_test(
            series,
            delta_t_steps=int(dt),
            n_bins=cfg.markov_test.n_bins_state,
            divergence=cfg.markov_test.divergence,
            bootstrap_samples=cfg.markov_test.bootstrap_samples,
            bootstrap_confidence=cfg.markov_test.bootstrap_confidence,
            rng=rng,
        )
        out.append(res.to_dict())
        # figura de la primera dt para diagnostico visual
        if dt == cfg.markov_test.delta_t_steps[1] if len(cfg.markov_test.delta_t_steps) > 1 else dt == cfg.markov_test.delta_t_steps[0]:
            plot_chapman_kolmogorov(
                res.bin_edges, res.transition_direct, res.transition_composed,
                Path(cfg.project.figures_root) / f"ck_dt{dt}.{cfg.reporting.figure_format}",
            )
    return out


def _smooth_km(km, cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    s = cfg.kramers_moyal.smoothing
    d1 = km.coefficients[1]
    d2 = km.coefficients[2]
    if s.enabled:
        d1 = smooth(d1, method=s.method, window=s.window, polyorder=s.polyorder)
        d2 = smooth(d2, method=s.method, window=s.window, polyorder=s.polyorder)
    return d1, d2


def _fit_models(returns: pd.Series, km, cfg: PipelineConfig) -> dict[str, object]:
    models: dict[str, object] = {}
    if cfg.models.m0_gaussian.enabled:
        models["M0_gaussian"] = GaussianBaseline.fit(returns)
    if cfg.models.m1_ornstein_uhlenbeck.enabled:
        if cfg.models.m1_ornstein_uhlenbeck.fit_method == "mle":
            models["M1_OU"] = OrnsteinUhlenbeck.fit_mle(returns)
        else:
            models["M1_OU"] = OrnsteinUhlenbeck.fit_km(
                km.x_centers, km.coefficients[1], km.coefficients[2]
            )
    if cfg.models.m2_langevin_reconstructed.enabled:
        models["M2_langevin"] = LangevinReconstructed.from_km_estimate(
            km,
            diffusion_floor=cfg.models.m2_langevin_reconstructed.diffusion_floor,
            drift_kind=cfg.models.m2_langevin_reconstructed.drift_interpolation,
            diffusion_kind=cfg.models.m2_langevin_reconstructed.diffusion_interpolation,
        )
    return models


def _simulate_models(
    models: dict[str, object],
    returns: pd.Series,
    cfg: PipelineConfig,
) -> dict[str, np.ndarray]:
    n_steps = cfg.simulation.n_steps or len(returns)
    sims: dict[str, np.ndarray] = {}
    for i, (name, m) in enumerate(models.items()):
        sims[name] = simulate_paths(
            m,
            n_paths=cfg.simulation.n_paths,
            n_steps=n_steps,
            dt=cfg.simulation.dt,
            burn_in=cfg.simulation.burn_in,
            parent_seed=cfg.project.random_seed + i,
            initial_condition=cfg.simulation.initial_condition,
            data_first=float(returns.iloc[0]),
            sample_pool=returns.to_numpy(),
        )
    return sims


def _evaluate(returns: pd.Series, sims: dict[str, np.ndarray]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    emp = returns.to_numpy()
    for name, paths in sims.items():
        # Promedio sobre el ultimo paso de todas las trayectorias
        flat_last = paths[-1]
        # Tambien comparar trayectoria 1 completa vs empirica para forma global
        flat_full = paths[:, 0]
        m_last = compare_all(emp, flat_last)
        m_full = compare_all(emp, flat_full)
        out[name] = {**{f"last.{k}": v for k, v in m_last.items()},
                     **{f"full.{k}": v for k, v in m_full.items()},
                     "ks_stat": m_full.get("ks_stat", float("nan")),
                     "jensen_shannon": m_full.get("jensen_shannon", float("nan")),
                     "moments": m_full.get("moments", {})}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline de reconstruccion estocastica.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--logging-config", default="configs/logging.yaml")
    args = parser.parse_args()

    setup_logging(args.logging_config)
    apply_thesis_style()
    cfg = load_config(args.config)
    rng = seed_everything(cfg.project.random_seed)

    out_root = ensure_dir(cfg.project.output_root)
    fig_root = ensure_dir(cfg.project.figures_root)
    tab_root = ensure_dir(cfg.project.tables_root)
    snapshot_config(cfg, out_root)

    # 2. Datos
    prices = load_prices(
        ticker=cfg.data.ticker,
        start=cfg.data.start,
        end=cfg.data.end,
        cache_dir=cfg.data.cache_dir,
        fallback_ticker=cfg.data.fallback_ticker,
        field=cfg.data.field,
    )

    # 3. Retornos
    raw_returns, returns = _prepare_returns(prices, cfg)
    save_dataframe(raw_returns.to_frame(), out_root / "log_returns.parquet")
    save_dataframe(returns.to_frame(), out_root / "log_returns_processed.parquet")

    # 4. Descriptiva + figuras
    desc = describe(returns)
    save_json(desc.to_dict(), out_root / "descriptive_stats.json")
    plot_price_and_returns(prices, raw_returns, fig_root / f"price_and_returns.{cfg.reporting.figure_format}")
    plot_pdf_vs_normal(returns, fig_root / f"pdf_vs_normal.{cfg.reporting.figure_format}", bins=cfg.statistics.histogram_bins)
    plot_acf(returns, fig_root / f"acf.{cfg.reporting.figure_format}", nlags=cfg.statistics.acf.nlags)
    save_table(
        descriptive_table(desc.to_dict()),
        tab_root / "descriptive.csv",
        latex_path=tab_root / "descriptive.tex" if cfg.reporting.generate_latex_tables else None,
    )

    # 5. CK
    ck_results = _run_ck_tests(returns, cfg, rng)
    save_json({"results": ck_results}, out_root / "chapman_kolmogorov.json")

    # 6. Kramers-Moyal
    km = estimate_km_coefficients(
        returns,
        orders=[1, 2] + list(cfg.kramers_moyal.pawula.compute_orders),
        n_bins=cfg.kramers_moyal.n_bins,
        binning=cfg.kramers_moyal.binning,
        delta_t_step=cfg.kramers_moyal.delta_t_step,
        min_obs_per_bin=cfg.kramers_moyal.min_obs_per_bin,
        bootstrap_n=cfg.kramers_moyal.bootstrap.n_resamples if cfg.kramers_moyal.bootstrap.enabled else 0,
        rng=rng,
    )
    d1_smooth, d2_smooth = _smooth_km(km, cfg)
    plot_km_coefficients(
        km.x_centers, d1_smooth, d2_smooth,
        fig_root / f"km_coefficients.{cfg.reporting.figure_format}",
        d1_se=km.coefficients_se.get(1),
        d2_se=km.coefficients_se.get(2),
    )
    pawula = pawula_ratios(km.x_centers, km.coefficients, higher_orders=list(cfg.kramers_moyal.pawula.compute_orders))
    save_json({"summary": pawula.summary}, out_root / "pawula.json")

    # 7. Modelos
    # Reemplazamos coefs suavizados antes de construir M2
    km_eff = km
    km_eff.coefficients[1] = d1_smooth
    km_eff.coefficients[2] = d2_smooth
    models = _fit_models(returns, km_eff, cfg)

    # Potencial efectivo M2
    if "M2_langevin" in models:
        m2 = models["M2_langevin"]
        x_grid, U = m2.effective_potential()
        plot_potential(x_grid, U, fig_root / f"potential_M2.{cfg.reporting.figure_format}",
                       title="Potencial efectivo (M2)")
        save_dataframe(
            pd.DataFrame({"x": x_grid, "U": U}),
            out_root / "potential_M2.parquet",
        )

    # 8. Simulacion
    sims = _simulate_models(models, returns, cfg)
    for name, paths in sims.items():
        plot_simulated_vs_empirical(
            returns.to_numpy(), paths[:, 0],
            fig_root / f"sim_vs_emp_{name}.{cfg.reporting.figure_format}",
            label_sim=name,
        )

    # 9. Validacion
    comparison = _evaluate(returns, sims)
    save_json(comparison, out_root / "comparison_metrics.json")
    save_table(
        comparison_table(comparison),
        tab_root / "comparison.csv",
        latex_path=tab_root / "comparison.tex" if cfg.reporting.generate_latex_tables else None,
    )

    # 10. Rolling
    if cfg.validation.windows.enabled:
        rr = rolling_km(
            returns,
            window_size=cfg.validation.windows.size_days,
            step=cfg.validation.windows.step_days,
            n_bins=max(15, cfg.kramers_moyal.n_bins // 3),
            min_obs_per_bin=20,
            delta_t_step=cfg.kramers_moyal.delta_t_step,
        )
        save_dataframe(rr.metrics, out_root / "rolling_km.parquet")
        if not rr.metrics.empty:
            plot_rolling_metrics(
                rr.metrics,
                fig_root / f"rolling_km.{cfg.reporting.figure_format}",
            )

    # 11. Reporte
    if cfg.reporting.generate_markdown:
        figure_paths = {
            "price_and_returns": str(fig_root / f"price_and_returns.{cfg.reporting.figure_format}"),
            "pdf_vs_normal": str(fig_root / f"pdf_vs_normal.{cfg.reporting.figure_format}"),
            "acf": str(fig_root / f"acf.{cfg.reporting.figure_format}"),
            "km_coefficients": str(fig_root / f"km_coefficients.{cfg.reporting.figure_format}"),
        }
        if "M2_langevin" in models:
            figure_paths["potential_M2"] = str(fig_root / f"potential_M2.{cfg.reporting.figure_format}")
        if cfg.validation.windows.enabled:
            figure_paths["rolling_km"] = str(fig_root / f"rolling_km.{cfg.reporting.figure_format}")
        for name in sims.keys():
            figure_paths[f"sim_vs_emp_{name}"] = str(fig_root / f"sim_vs_emp_{name}.{cfg.reporting.figure_format}")

        render_report(
            out_path=Path("reports") / "pipeline_report.md",
            config_snapshot=cfg.model_dump(),
            descriptive=desc.to_dict(),
            ck_results=ck_results,
            km_summary={
                "delta_t": km.delta_t,
                "n_bins": km.x_centers.size,
                "n_used": km.n_used,
            },
            pawula_summary=pawula.summary,
            comparison=comparison,
            figure_paths=figure_paths,
        )

    logger.info("Pipeline finalizado correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
