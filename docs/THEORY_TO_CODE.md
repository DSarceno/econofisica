# Mapa Teoría ↔ Código

Cada fila relaciona un concepto/ecuación de la tesis con su implementación en el
repositorio. Permite trazabilidad bidireccional para revisión académica.

## Capítulo 1 — Física estadística fuera del equilibrio (`tch1.tex`)

| Sección/Ecuación tesis | Concepto | Módulo / función |
|---|---|---|
| §"Movimiento browniano" Ec. `dx = -γ v dt + ξ dt` | Langevin lineal | `src/models/ornstein_uhlenbeck.py::OrnsteinUhlenbeck.drift` |
| §"Ecuación de Langevin" `dx = D^{(1)}dt + √(2D^{(2)})dW` | SDE genérica | `src/models/base.py::StochasticModel`, `src/simulation/euler_maruyama.py::simulate_paths` |
| §"De Langevin a Fokker-Planck" `∂P/∂t = -∂(D^{(1)}P) + ∂²(D^{(2)}P)` | FP | (Reconstruido analíticamente; se evalúa vía simulación EM) |
| §"Corriente de probabilidad" `J = D^{(1)}P - ∂(D^{(2)}P)` | Irreversibilidad | `src/validation/rolling_analysis.py::probability_current_proxy` |
| §"Potencial efectivo" `D^{(1)} = -dU/dx` | Potencial | `src/models/potential.py::potential_from_drift`, `src/models/langevin_reconstructed.py::potential` |

## Capítulo 2 — Reconstrucción dinámica (`tch2.tex`)

| Sección/Ecuación | Concepto | Módulo / función |
|---|---|---|
| §"Procesos de Markov" + Chapman–Kolmogorov | Hipótesis markoviana | `src/markov/chapman_kolmogorov.py::chapman_kolmogorov_test` |
| §"Ecuación maestra" | Tasa de transición | (Implícita en estimación K-M) |
| §"Expansión de Kramers–Moyal" `D^{(k)} = (1/k!) lim ⟨(Δx)^k|x⟩/Δt` | K-M | `src/kramers_moyal/estimators.py::estimate_km_coefficients` |
| §"Teorema de Pawula" | Validez del truncamiento | `src/kramers_moyal/pawula.py::pawula_ratios` |
| §"Relación con Langevin" | Equivalencia FP↔Langevin | `src/models/langevin_reconstructed.py::LangevinReconstructed` |
| §"Estimación empírica" | Momentos condicionales | `src/kramers_moyal/estimators.py::_km_single` |
| §"Verificación de Markovianidad" | C-K + JS/KL | `src/markov/chapman_kolmogorov.py` |

## Capítulo 3 — Contexto financiero (`tch3.tex`)

| Sección | Concepto | Módulo / función |
|---|---|---|
| §"Log-retorno" Ec. `r_t = ln P_t - ln P_{t-1}` | Variable central | `src/preprocessing/returns.py::log_returns` |
| §"Hechos estilizados" — colas pesadas | PDF empírica + Hill | `src/statistics/descriptive.py::empirical_pdf`, `hill_tail_index` |
| §"Hechos estilizados" — clustering vol | ACF² | `src/statistics/descriptive.py::squared_autocorrelation` |
| §"Hechos estilizados" — exceso de curtosis | Momentos | `src/statistics/descriptive.py::describe` |
| §"Selección S&P 500/SPY" | Adquisición | `src/data/loaders.py::load_prices` |

## Capítulo 4 — Metodología (`tch4.tex`)

| Sección | Concepto | Módulo / función |
|---|---|---|
| §"Discretización y bins" | Espacio de estados | `src/kramers_moyal/estimators.py::_build_edges` |
| §"Estimación K-M" Ec. `D^{(k)}(r) ≈ ⟨[r(t+Δt)-r]^k|r⟩/(k!Δt)` | Estimador | `src/kramers_moyal/estimators.py` |
| §"Dependencia en Δt" | Barrido de escalas | Configurable en `pipeline.yaml::markov_test.delta_t_steps` |
| §"Robustez y validación" | Comparación sim vs emp | `src/validation/metrics.py::compare_all` |

## Apéndice — Derivaciones (`apendice.tex`)

| Sección | Concepto | Módulo / función |
|---|---|---|
| §"Maestra → K-M" | Derivación formal | Documentado en docstring de `src/kramers_moyal/estimators.py` |
| §"K-M → Fokker–Planck" | Truncamiento difusivo | `src/kramers_moyal/pawula.py` |
| §"Itô vs Stratonovich" | Convención SDE | Itô, declarado en `src/simulation/euler_maruyama.py` |
| §"Producción de entropía" | Irreversibilidad | `src/validation/rolling_analysis.py::probability_current_proxy` (proxy) |

## Protocolo (objetivos específicos)

| Objetivo | Implementación |
|---|---|
| Obtener y preprocesar datos S&P 500 | `src/data/loaders.py` + `scripts/stages/fetch_data.py` |
| Construir log-retornos | `src/preprocessing/returns.py::log_returns` |
| Caracterizar PDF/momentos/ACF | `src/statistics/descriptive.py` |
| Test C-K | `src/markov/chapman_kolmogorov.py` |
| Estimar D^(1), D^(2) | `src/kramers_moyal/estimators.py` |
| Interpretación física (potencial) | `src/models/potential.py`, `src/models/langevin_reconstructed.py::effective_potential` |
| Simulación Euler–Maruyama | `src/simulation/euler_maruyama.py` |
| Comparación con M0/M1 | `src/models/baseline_gaussian.py`, `src/models/ornstein_uhlenbeck.py` |
| Análisis por ventanas | `src/validation/rolling_analysis.py::rolling_km` |
| Pipeline reproducible | `scripts/run_pipeline.py` + `configs/pipeline.yaml` |
