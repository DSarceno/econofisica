# Decisiones de diseño

Registro explícito de las decisiones metodológicas y de ingeniería que sustentan
el repositorio. Cada decisión incluye contexto, alternativas consideradas y
justificación.

## D1. Variable de estudio: log-retornos diarios

- **Contexto**: la tesis (capítulo 3) justifica los log-retornos por aditividad
  temporal y mejor comportamiento estadístico que precios o retornos simples.
- **Alternativas**: precios crudos, retornos simples, log-precios.
- **Decisión**: log-retornos diarios `r_t = ln P_t − ln P_{t−1}` sobre `Adj Close`.
- **Implementación**: `src/preprocessing/returns.py::log_returns`.
- **Reflejar en tesis**: capítulo 4 §"Construcción de variables".

## D2. Fuente de datos: yfinance con cache local

- **Contexto**: la tesis requiere un proveedor gratuito reproducible.
- **Alternativas**: Quandl/Nasdaq Data Link (requiere API key), AlphaVantage
  (rate-limited), descarga manual de Yahoo Finance.
- **Decisión**: `yfinance` para `^GSPC` con fallback `SPY`; cache local en
  Parquet con clave `<ticker>_<start>_<end>.parquet`.
- **Justificación**: zero-config, histórico largo, ampliamente usado en
  econofísica académica.
- **Implementación**: `src/data/loaders.py`.

## D3. Estandarización: global por defecto, rolling opcional

- **Contexto**: literatura econofísica usa frecuentemente
  `z = (r − μ)/σ` para comparar bins simétricamente.
- **Decisión**: global por defecto; rolling de 252 días como variante para
  controlar heterocedasticidad lenta.
- **Implementación**: `src/preprocessing/returns.py::standardize`.

## D4. Binning equipoblado (cuantil)

- **Contexto**: bins uniformes producen alta varianza en colas y sesgo en el
  centro de la distribución.
- **Decisión**: bins por cuantiles (`np.quantile`) de modo que cada bin tenga
  ~equal poblational mass.
- **Justificación**: estabiliza la varianza del estimador K-M por bin
  (Friedrich & Peinke 2000). Uniforme disponible vía config.
- **Implementación**: `src/kramers_moyal/estimators.py::_build_edges`.

## D5. Convención SDE: Itô

- **Contexto**: el apéndice de la tesis lo indica explícitamente.
- **Decisión**: Itô. La estimación K-M empírica naturalmente apunta a Itô:
  el momento condicional se evalúa en `x(t)`, no en el punto medio.
- **Implementación**: `src/simulation/euler_maruyama.py` (esquema explícito Itô).

## D6. Suavizado de coeficientes K-M: Savitzky–Golay

- **Contexto**: D^(1), D^(2) por bin son ruidosos; el potencial U(x) integra
  la pendiente y amplifica el ruido.
- **Alternativas**: media móvil, splines de regularización, kernel.
- **Decisión**: Savitzky–Golay (orden 3, ventana 7) — preserva máximos/minimos
  locales mejor que media móvil; configurable.
- **Implementación**: `src/kramers_moyal/smoothing.py`.

## D7. Test de Pawula: ratios `D^(2k) / [(2k−1)!! · (D^(2))^k]`

- **Contexto**: probar la validez del truncamiento difusivo (capítulo 2).
- **Decisión**: calcular para órdenes pares 4 y 6; reportar
  `mean|R_k|`, `median|R_k|`, `max|R_k|`. Si `mean|R_k| ≪ 1`, FP es buena
  aproximación.
- **Implementación**: `src/kramers_moyal/pawula.py`.

## D8. Modelos M0/M1/M2 con interfaz común

- **Contexto**: el orquestador debe simular cualquier modelo sin ramificar.
- **Decisión**: clase abstracta `StochasticModel` con `drift(x,t)` y
  `diffusion(x,t)`; cada modelo concreto implementa `fit*` apropiado.
- **Implementación**: `src/models/base.py`.

## D9. Reproducibilidad bit-exacta de simulaciones

- **Contexto**: distintos paths deben ser independientes pero reproducibles.
- **Decisión**: `np.random.SeedSequence((parent_seed, stream_id))`. La
  simulación recibe `parent_seed` y deriva 1 RNG para condiciones iniciales y
  otro para incrementos brownianos.
- **Implementación**: `src/utils/seeding.py::derived_generator`.

## D10. Snapshot inmutable de configuración

- **Contexto**: cualquier corrida debe ser auditable a posteriori.
- **Decisión**: al iniciar el pipeline se persiste `config_snapshot.json` junto
  a los outputs. El reporte Markdown referencia este snapshot.
- **Implementación**: `src/utils/io.py::snapshot_config`.

## D11. Validación de config con Pydantic v2

- **Contexto**: errores de tipo/parámetro no deben aparecer en mitad de un
  pipeline largo.
- **Decisión**: cada bloque YAML mapea a un `BaseModel`; la carga falla
  inmediatamente si hay incongruencias.
- **Implementación**: `src/utils/config.py::PipelineConfig`.

## D12. Tests sintéticos antes que tests sobre datos reales

- **Contexto**: validar correctness de K-M sobre datos reales es circular.
- **Decisión**: generar series OU sintéticas con parámetros conocidos y
  verificar que el estimador recupera la pendiente y la difusión esperadas.
- **Implementación**: `tests/conftest.py::synthetic_ou_series` + tests en
  `tests/unit/test_kramers_moyal.py`.

## D13. CLI con stages independientes

- **Contexto**: durante desarrollo es costoso re-correr todo el pipeline.
- **Decisión**: orquestador completo en `scripts/run_pipeline.py`; stages
  individuales en `scripts/stages/`.
- **Implementación**: ver `scripts/stages/{fetch_data, run_km, run_markov_test, run_simulation}.py`.

## D14. Visualización: paleta accesible, formato PDF

- **Decisión**: paleta categórica de 6 colores accesibles (azul/rojo/verde/violeta/naranja/gris); figuras en PDF (vectorial) por defecto para inclusión directa en la tesis LaTeX. PNG/SVG vía config.
- **Implementación**: `src/visualization/style.py`.

## D15. Métricas: KS + JS + KL + momentos + ACF² + Hill

- **Justificación**: conjunto que cubre forma global de la distribución
  (KS, JS), discrepancia direccional (KL), momentos individuales, dependencia
  no lineal (ACF²) y régimen de colas (Hill).
- **Implementación**: `src/validation/metrics.py::compare_all`.
