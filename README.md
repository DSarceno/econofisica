# Reconstrucción de dinámicas estocásticas tipo Langevin para el S&P 500

Implementación computacional de la tesis _"Reconstrucción de dinámicas estocásticas
fuera del equilibrio en series financieras mediante ecuaciones de Langevin y de
Fokker–Planck"_ (Diego R. Sarceno, USAC, 2026).

## Propósito

Este repositorio convierte el contenido teórico-metodológico de la tesis en un
**ecosistema reproducible** que:

1. descarga y cachea precios históricos del S&P 500 (`^GSPC`, fallback `SPY`);
2. construye log-retornos diarios y los caracteriza estadísticamente;
3. valida la hipótesis markoviana mediante el **test de Chapman–Kolmogorov**;
4. estima los **coeficientes de Kramers–Moyal** `D^(1)`, `D^(2)`, ... no
   paramétricamente y aplica el **test de Pawula**;
5. ajusta **tres modelos** (M0 gaussiano, M1 Ornstein–Uhlenbeck, M2 Langevin
   reconstruido) y simula trayectorias con **Euler–Maruyama**;
6. compara modelos vs datos empíricos con métricas (KS, Jensen–Shannon, KL,
   ACF², índice de Hill, momentos);
7. realiza **análisis por ventanas** para evidencia de no-equilibrio;
8. genera figuras y tablas listas para la tesis y un **reporte Markdown**
   ejecutivo.

## Trazabilidad teoría ↔ código

Cada módulo científico declara explícitamente el capítulo y la ecuación de la
tesis que implementa. El mapa completo está en
[`docs/THEORY_TO_CODE.md`](docs/THEORY_TO_CODE.md).

## Estructura

```
configs/             YAML con TODA decisión metodológica
data/raw/            Cache de precios descargados
data/processed/      Outputs del pipeline (parquet, JSON)
src/data/            Loaders (yfinance + cache)
src/preprocessing/   Log-retornos, estandarización, winsorización
src/statistics/      Descriptiva, ACF, Hill
src/markov/          Test de Chapman–Kolmogorov
src/kramers_moyal/   Estimadores K-M, suavizado, Pawula
src/models/          M0, M1, M2 + potencial efectivo
src/simulation/      Euler–Maruyama vectorizado
src/validation/      Métricas y análisis por ventanas
src/visualization/   Figuras (estilo tesis-ready)
src/reporting/       Tablas LaTeX, reporte Markdown
src/utils/           Config (Pydantic), logging, IO, semillas
scripts/             Orquestador y stages CLI
tests/unit/          Pruebas unitarias (recuperación de OU, métricas, etc.)
tests/integration/   Smoke test end-to-end con datos sintéticos
docs/                Documentación técnica y guías
```

## Instalación

```bash
python -m venv .venv
source .venv/Scripts/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Ejecución

### Pipeline completo (un solo comando)

```bash
python scripts/run_pipeline.py --config configs/pipeline.yaml
```

Outputs:
- `data/processed/*.parquet|*.json` — series, descriptiva, K-M, métricas.
- `reports/figures/*.pdf` — figuras de la tesis.
- `reports/tables/*.csv|*.tex` — tablas en CSV y LaTeX (booktabs).
- `reports/pipeline_report.md` — reporte ejecutivo de la corrida.
- `logs/pipeline.log` — log estructurado.

### Stages independientes

```bash
python scripts/stages/fetch_data.py        --config configs/pipeline.yaml
python scripts/stages/run_markov_test.py   --config configs/pipeline.yaml
python scripts/stages/run_km.py            --config configs/pipeline.yaml
python scripts/stages/run_simulation.py    --theta 0.5 --mu 0 --D 0.5 --n-paths 500 --n-steps 5000
```

### Tests

```bash
pytest -q                     # unit + integration
pytest -q -m "not slow"       # solo rápidos
pytest --cov=src --cov-report=term-missing
```

## Reproducibilidad

- Semilla global y semillas derivadas por path (`src/utils/seeding.py`).
- `config_snapshot.json` se persiste junto a los outputs en cada corrida.
- Hash SHA-256 de los datos cargados se imprime en log.
- Lockfiles disponibles en `requirements.txt`.

## Documentación adicional

- [`docs/THEORY_TO_CODE.md`](docs/THEORY_TO_CODE.md) — mapa tesis ↔ módulo ↔ ecuación.
- [`docs/RUN_GUIDE.md`](docs/RUN_GUIDE.md) — guía paso a paso de ejecución y troubleshooting.
- [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md) — decisiones de diseño y justificación.

## Licencia

MIT. Citar la tesis si se reutiliza con fines académicos.
