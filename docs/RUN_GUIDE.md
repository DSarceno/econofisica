# Guía de ejecución

## 0. Requisitos

- Python 3.11+
- Conexión a internet en la primera corrida (descarga `^GSPC` vía `yfinance`)
- ~200 MB de espacio para cache + outputs
- Sistemas: Windows / macOS / Linux (probado en Win11 + bash)

## 1. Instalación

```bash
git init                                       # opcional, si aún no es repo
python -m venv .venv
source .venv/Scripts/activate                  # Windows bash
# source .venv/bin/activate                    # macOS / Linux
pip install -e ".[dev]"
```

Verificación rápida:

```bash
python -c "import src; print(src.__version__)"
pytest -q tests/unit/test_config.py
```

## 2. Configuración

Toda decisión metodológica vive en `configs/pipeline.yaml`. Antes de la primera
corrida revisa al menos:

- `data.start` / `data.end` — período de análisis.
- `project.random_seed` — semilla maestra (se deriva por path en simulaciones).
- `kramers_moyal.n_bins` y `kramers_moyal.binning` — discretización del espacio de estados.
- `markov_test.delta_t_steps` — escalas a evaluar para el test de Markov.
- `simulation.n_paths` / `simulation.n_steps` — costo computacional.

> **Nota**: `configs/pipeline.yaml` valida via Pydantic (`src/utils/config.py`).
> Cualquier inconsistencia produce un `ValidationError` con campo problemático.

## 3. Ejecución end-to-end

```bash
python scripts/run_pipeline.py --config configs/pipeline.yaml
```

Tiempo aproximado (Intel i5, 8 GB):
- Descarga (1ª vez): ~10 s
- Descriptiva + figuras: ~5 s
- Test C-K (5 escalas, 200 bootstraps): ~30 s
- Estimación K-M + Pawula + bootstrap: ~20 s
- Ajuste M0/M1/M2 + simulación 500 paths × 8800 steps: ~25 s
- Validación + reporte: ~5 s

Total típico: **~1.5 min** en hardware modesto.

## 4. Outputs principales

```
data/processed/
├── config_snapshot.json          # snapshot inmutable de la corrida
├── log_returns.parquet           # serie cruda r_t
├── log_returns_processed.parquet # serie postprocesada (estandarizada)
├── descriptive_stats.json
├── chapman_kolmogorov.json
├── pawula.json
├── potential_M2.parquet          # potencial efectivo reconstruido
├── comparison_metrics.json       # KS, JS, momentos, ACF^2 por modelo
└── rolling_km.parquet            # diagnóstico por ventanas

reports/
├── figures/*.pdf                 # ver THEORY_TO_CODE.md para mapeo
├── tables/*.csv,*.tex            # tablas LaTeX (booktabs)
└── pipeline_report.md            # reporte ejecutivo

logs/
└── pipeline.log                  # log estructurado
```

## 5. Stages independientes

Útiles durante desarrollo o exploración:

```bash
# Solo descargar + cachear
python scripts/stages/fetch_data.py

# Solo test C-K (lee desde cache si existe)
python scripts/stages/run_markov_test.py

# Solo estimar K-M
python scripts/stages/run_km.py

# Simulación ad-hoc (no requiere datos)
python scripts/stages/run_simulation.py --theta 0.3 --D 0.4 --n-paths 1000 --n-steps 5000
```

## 6. Tests

```bash
pytest -q                             # toda la suite
pytest -q tests/unit                  # solo unit
pytest -q tests/integration           # smoke end-to-end
pytest --cov=src --cov-report=term    # cobertura
```

Los tests usan series **sintéticas** (OU exacto, gaussiano IID) para validar
que los estimadores K-M recuperan parámetros conocidos. Esto garantiza que la
implementación es correcta antes de aplicarla a datos reales.

## 7. Troubleshooting

| Síntoma | Diagnóstico / solución |
|---|---|
| `DataLoadError: yfinance retorno vacio` | Verifica conexión y rango de fechas. Probar `fallback_ticker: SPY`. |
| `ValueError: Serie demasiado corta` | Reduce `kramers_moyal.delta_t_step` o amplía el período. |
| `Bootstrap CI` muy ancho | Sube `bootstrap_samples`; o reduce `n_bins_state`. |
| `D^(2)` con NaN en muchos bins | Sube `min_obs_per_bin` solo si tienes muchos datos; o reduce `n_bins`. |
| Figuras vacías | Asegúrate de no estar ejecutando dentro de un entorno headless sin backend Agg; revisa `matplotlib.use("Agg")` si necesitas. |

## 8. Workflow recomendado para iteración

1. Editar `configs/pipeline.yaml`.
2. Correr `pytest -q -k km or markov` para validar suposiciones.
3. Correr `python scripts/run_pipeline.py`.
4. Revisar `reports/pipeline_report.md` y figuras.
5. Iterar.
