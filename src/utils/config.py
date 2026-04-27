# --- FILE: src/utils/config.py ---
"""
Validacion tipada de la configuracion del pipeline mediante Pydantic v2.

Trazabilidad tesis -> codigo:
    - Cada bloque del YAML mapea a un Pydantic submodelo aqui descrito.
    - Los nombres de campos espejean fielmente la nomenclatura usada en
      'configs/pipeline.yaml' para evitar drift semantico.

El objetivo central es: ningun parametro cientifico puede ser hardcodeado en el
codigo. Toda decision metodologica entra por configuracion validada y queda
serializada junto con los outputs (ver utils.io.snapshot_config).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, PositiveInt, field_validator


class ProjectCfg(BaseModel):
    name: str
    random_seed: int = 20260426
    output_root: str = "data/processed"
    figures_root: str = "reports/figures"
    tables_root: str = "reports/tables"


class LoggingCfg(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    file: str = "logs/pipeline.log"
    jsonl: str = "logs/pipeline.jsonl"


class DataCfg(BaseModel):
    source: Literal["yfinance", "csv"] = "yfinance"
    ticker: str = "^GSPC"
    fallback_ticker: str | None = "SPY"
    start: str = "1990-01-01"
    end: str = "2024-12-31"
    cache_dir: str = "data/raw"
    field: str = "Adj Close"


class StandardizeCfg(BaseModel):
    enabled: bool = True
    method: Literal["global", "rolling"] = "global"
    rolling_window: PositiveInt = 252


class WinsorizeCfg(BaseModel):
    enabled: bool = False
    quantiles: tuple[float, float] = (0.001, 0.999)


class PreprocessingCfg(BaseModel):
    log_returns: bool = True
    drop_na: bool = True
    standardize: StandardizeCfg = StandardizeCfg()
    winsorize: WinsorizeCfg = WinsorizeCfg()


class AcfCfg(BaseModel):
    nlags: PositiveInt = 60


class StatisticsCfg(BaseModel):
    acf: AcfCfg = AcfCfg()
    acf_squared: AcfCfg = AcfCfg()
    histogram_bins: PositiveInt = 100
    qq_distribution: str = "norm"


class MarkovTestCfg(BaseModel):
    delta_t_steps: list[PositiveInt] = Field(default_factory=lambda: [1, 2, 5, 10, 20])
    n_bins_state: PositiveInt = 25
    bootstrap_samples: PositiveInt = 200
    bootstrap_confidence: float = 0.95
    divergence: Literal["jensen_shannon", "kl"] = "jensen_shannon"


class SmoothingCfg(BaseModel):
    enabled: bool = True
    method: Literal["savgol", "moving_average", "none"] = "savgol"
    window: PositiveInt = 7
    polyorder: int = 3


class BootstrapCfg(BaseModel):
    enabled: bool = True
    n_resamples: PositiveInt = 200
    confidence: float = 0.95


class PawulaCfg(BaseModel):
    compute_orders: list[PositiveInt] = Field(default_factory=lambda: [4, 6])


class KramersMoyalCfg(BaseModel):
    n_bins: PositiveInt = 50
    binning: Literal["quantile", "uniform"] = "quantile"
    state_range: tuple[float, float] | None = None
    delta_t_step: PositiveInt = 1
    min_obs_per_bin: PositiveInt = 30
    smoothing: SmoothingCfg = SmoothingCfg()
    bootstrap: BootstrapCfg = BootstrapCfg()
    pawula: PawulaCfg = PawulaCfg()


class M0Cfg(BaseModel):
    enabled: bool = True


class M1Cfg(BaseModel):
    enabled: bool = True
    fit_method: Literal["mle", "km"] = "mle"


class M2Cfg(BaseModel):
    enabled: bool = True
    drift_interpolation: Literal["linear", "cubic"] = "cubic"
    diffusion_interpolation: Literal["linear", "cubic"] = "cubic"
    diffusion_floor: float = 1e-8


class ModelsCfg(BaseModel):
    m0_gaussian: M0Cfg = M0Cfg()
    m1_ornstein_uhlenbeck: M1Cfg = M1Cfg()
    m2_langevin_reconstructed: M2Cfg = M2Cfg()


class SimulationCfg(BaseModel):
    method: Literal["euler_maruyama"] = "euler_maruyama"
    n_paths: PositiveInt = 500
    n_steps: int | None = None
    dt: float = 1.0
    burn_in: int = 200
    initial_condition: Literal["data_first", "zero", "sample"] = "data_first"


class WindowsCfg(BaseModel):
    enabled: bool = True
    size_days: PositiveInt = 504
    step_days: PositiveInt = 126


class NoneqCfg(BaseModel):
    compute_probability_current: bool = True


class ValidationCfg(BaseModel):
    metrics: list[str] = Field(
        default_factory=lambda: [
            "ks_distance",
            "jensen_shannon",
            "moments",
            "acf_squared_mse",
            "tail_index_hill",
        ]
    )
    windows: WindowsCfg = WindowsCfg()
    noneq_evidence: NoneqCfg = NoneqCfg()


class ReportingCfg(BaseModel):
    generate_markdown: bool = True
    generate_latex_tables: bool = True
    figure_format: Literal["pdf", "png", "svg"] = "pdf"
    figure_dpi: PositiveInt = 150


class PipelineConfig(BaseModel):
    project: ProjectCfg
    logging: LoggingCfg = LoggingCfg()
    data: DataCfg
    preprocessing: PreprocessingCfg = PreprocessingCfg()
    statistics: StatisticsCfg = StatisticsCfg()
    markov_test: MarkovTestCfg = MarkovTestCfg()
    kramers_moyal: KramersMoyalCfg = KramersMoyalCfg()
    models: ModelsCfg = ModelsCfg()
    simulation: SimulationCfg = SimulationCfg()
    validation: ValidationCfg = ValidationCfg()
    reporting: ReportingCfg = ReportingCfg()

    @field_validator("project")
    @classmethod
    def _ensure_seed_positive(cls, v: ProjectCfg) -> ProjectCfg:
        if v.random_seed < 0:
            raise ValueError("random_seed debe ser >= 0")
        return v


def load_config(path: str | Path) -> PipelineConfig:
    """Carga y valida un YAML de configuracion del pipeline."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return PipelineConfig.model_validate(raw)
