# CLAUDE.md

## 🧭 Identidad del proyecto

Este repositorio es la **implementación computacional de una tesis de econofísica**:

> "Reconstrucción de dinámicas estocásticas fuera del equilibrio en series financieras mediante ecuaciones de Langevin y Fokker–Planck"

NO es un proyecto de machine learning genérico.

Es un **pipeline científico con fundamento teórico explícito**, donde cada módulo implementa ecuaciones y conceptos de la tesis.

---

## 🎯 Objetivo científico

Reconstruir la dinámica estocástica de series financieras (S&P 500 / SPY) mediante:

- Validación de la hipótesis markoviana (Chapman–Kolmogorov)
- Estimación de coeficientes de Kramers–Moyal D^(1), D^(2), ...
- Validación del truncamiento difusivo (teorema de Pawula)
- Construcción de modelos:
  - M0 (gaussiano)
  - M1 (Ornstein–Uhlenbeck)
  - M2 (Langevin reconstruido)
- Simulación mediante Euler–Maruyama
- Comparación entre datos empíricos y simulados

---

## 🏗️ Arquitectura del sistema

El pipeline es estrictamente modular:
```
data → preprocessing → statistics → markov → kramers_moyal → models → simulation → validation → reporting
```


Cada módulo corresponde a capítulos y ecuaciones de la tesis (ver `docs/THEORY_TO_CODE.md`).

---

## 📁 Módulos clave

### Datos
- `src/data/loaders.py`
- Descarga con `yfinance`
- Cache local en parquet
- Fallback (`SPY`)
- Hash SHA-256 para reproducibilidad

### Preprocesamiento
- `src/preprocessing/returns.py`
- Log-retornos (variable central)
- Estandarización (global o rolling)
- Winsorización solo para diagnóstico

### Núcleo científico
- `src/markov/` → test de Chapman–Kolmogorov
- `src/kramers_moyal/` → estimación de D^(k)
- `src/models/` → modelos M0, M1, M2
- `src/simulation/` → Euler–Maruyama (Itô)
- `src/validation/` → métricas y análisis

---

## ⚙️ Supuestos científicos clave

1. **Los log-retornos son la variable fundamental**
   - `r_t = ln(P_t) − ln(P_{t−1})`
   - Justificado por propiedades estadísticas y aditividad :contentReference[oaicite:1]{index=1}

2. **La propiedad de Markov debe validarse empíricamente**
   - No se asume

3. **La expansión de Kramers–Moyal es central**
   - D^(1) y D^(2) definen la dinámica

4. **Equivalencia Fokker–Planck ↔ Langevin**
   - La dinámica debe ser interpretable como SDE

5. **Las colas pesadas son parte del fenómeno**
   - No deben eliminarse artificialmente

---

## 🔬 Filosofía de modelado

Los modelos deben ser:

- Interpretables físicamente
- Derivados de datos empíricos
- Consistentes con teoría

Modelos:

- M0 → baseline gaussiano
- M1 → Ornstein–Uhlenbeck
- M2 → Langevin reconstruido (principal contribución)

---

## 🔁 Reproducibilidad

Requisitos estrictos:

- Cada corrida genera:
  - `config_snapshot.json`
  - hash de datos
- Semillas determinísticas
- Configuración centralizada

---

## 🧪 Validación

Comparación empírico vs simulado usando:

- Kolmogorov–Smirnov (KS)
- Jensen–Shannon (JS)
- Kullback–Leibler (KL)
- ACF² (clustering de volatilidad)
- Índice de Hill (colas)
- Momentos estadísticos

---

## 📊 Reglas sobre datos (CRÍTICO)

- Datos crudos son inmutables
- Transformaciones explícitas
- No fuga de información
- Preservar orden temporal

---

## 🧠 Instrucciones para Claude

Actúa como:

> Arquitecto de software científico + investigador en econofísica

---

### SIEMPRE:

- Respetar el marco teórico (Langevin / Fokker–Planck)
- Mantener trazabilidad teoría ↔ código
- Justificar decisiones matemáticamente
- Preservar reproducibilidad
- Mantener modularidad

---

### NUNCA:

- Introducir modelos de ML sin justificación
- Romper supuestos de Markov sin advertir
- Usar winsorización como paso principal
- Ignorar colas pesadas o clustering
- Romper la arquitectura del pipeline

---

### AL PROPONER CAMBIOS:

Debes:

1. Explicar la razón científica
2. Indicar el módulo afectado
3. Explicar impacto en el pipeline
4. Mantener compatibilidad con el sistema actual

---

### AL ESCRIBIR CÓDIGO:

- Código modular
- Funciones claras
- Uso de utilidades existentes
- Logging consistente
- Docstrings en español

---

### SI HAY INCERTIDUMBRE:

Preguntar antes de asumir.

---

## 📌 Estado actual

- Pipeline funcional
- Preprocesamiento implementado
- Núcleo científico implementado
- Sistema listo para experimentación

---

## 🚧 Problemas abiertos

- Robustez de estimación Kramers–Moyal
- Sensibilidad a binning
- No estacionariedad (ventanas)
- Escalas de validez de Markov

---

## 📚 Referencias internas

- `docs/THEORY_TO_CODE.md` → mapeo teoría ↔ código :contentReference[oaicite:2]{index=2}
- `docs/DESIGN_DECISIONS.md` → decisiones metodológicas :contentReference[oaicite:3]{index=3}
- `docs/RUN_GUIDE.md` → ejecución del pipeline :contentReference[oaicite:4]{index=4}