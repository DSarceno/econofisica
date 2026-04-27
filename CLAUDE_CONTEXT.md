# CLAUDE_CONTEXT.md

## Contexto mínimo del proyecto

Este repositorio implementa computacionalmente una tesis de econofísica sobre reconstrucción de dinámicas estocásticas fuera del equilibrio en series financieras, usando ecuaciones de Langevin, Fokker–Planck y coeficientes de Kramers–Moyal.

Activo principal: S&P 500 (`^GSPC`) con fallback `SPY`.

Variable central:
`r_t = ln(P_t) - ln(P_{t-1})`

La arquitectura del pipeline es:

data → preprocessing → statistics → markov → kramers_moyal → models → simulation → validation → reporting

## Objetivo científico

Reconstruir una dinámica estocástica empírica para log-retornos financieros mediante:

- validación de la hipótesis markoviana con Chapman–Kolmogorov;
- estimación no paramétrica de coeficientes de Kramers–Moyal;
- validación del truncamiento difusivo con Pawula;
- construcción de modelos M0, M1 y M2;
- simulación con Euler–Maruyama;
- comparación entre datos reales y simulados.

## Reglas críticas

- No tratar esto como un proyecto genérico de machine learning.
- Mantener trazabilidad teoría ↔ código.
- Justificar matemáticamente cualquier transformación.
- No usar winsorización como paso principal; solo como diagnóstico.
- Preservar orden temporal.
- Evitar fuga de información.
- Mantener reproducibilidad con configuración, hashes y semillas.
- Toda decisión metodológica debe vivir en `configs/pipeline.yaml` o en documentación.

## Módulos clave

- `src/data/loaders.py`: descarga/cache de precios con `yfinance`, fallback y hash.
- `src/preprocessing/returns.py`: log-retornos, estandarización y winsorización diagnóstica.
- `src/markov/`: test de Chapman–Kolmogorov.
- `src/kramers_moyal/`: estimación de D^(k), suavizado y Pawula.
- `src/models/`: modelos M0, M1 Ornstein–Uhlenbeck y M2 Langevin reconstruido.
- `src/simulation/`: Euler–Maruyama bajo convención Itô.
- `src/validation/`: KS, JS, KL, ACF², Hill y momentos.
- `src/reporting/`: tablas, figuras y reporte final.

## Modelos

M0: referencia gaussiana.  
M1: Ornstein–Uhlenbeck.  
M2: Langevin reconstruido empíricamente; es el modelo central.

## Instrucciones para Claude

Actúa como arquitecto de software científico e investigador en econofísica.

Cuando propongas cambios:

1. explica la razón científica;
2. indica el módulo afectado;
3. respeta la arquitectura existente;
4. evita soluciones de caja negra;
5. conserva reproducibilidad;
6. escribe todo en español.

Cuando escribas código:

- usa funciones modulares;
- conserva logging;
- respeta configuración centralizada;
- no dupliques lógica;
- añade docstrings en español;
- mantén compatibilidad con el pipeline existente.

## Archivos de referencia

- `README.md`
- `docs/THEORY_TO_CODE.md`
- `docs/DESIGN_DECISIONS.md`
- `docs/RUN_GUIDE.md`
- `configs/pipeline.yaml`