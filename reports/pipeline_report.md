# Reporte de pipeline — econophysics-langevin-spx

_Generado:_ `2026-04-29T01:33:22.334633Z`

## 1. Configuracion

- Ticker: `^GSPC`
- Periodo: 1990-01-01 — 2024-12-31
- Semilla: 20260426

## 2. Estadistica descriptiva

| Estadistico | Valor |
|---|---|
| n | 8815 |
| mean | -1.16879e-17 |
| std | 1 |
| skewness | -0.398993 |
| kurtosis_excess | 10.6698 |
| min | -11.2493 |
| max | 9.60413 |
| Jarque-Bera p | 0.000e+00 |

## 3. Test de Markov (Chapman-Kolmogorov)

| dt | divergencia media | bootstrap CI |
|---|---|---|
| 2 | 1.464e-02 | [2.012e-02, 2.569e-02] |
| 5 | 1.201e-02 | [1.694e-02, 2.219e-02] |
| 10 | 1.108e-02 | [1.498e-02, 2.017e-02] |
| 20 | 1.042e-02 | [1.303e-02, 1.674e-02] |

## 4. Coeficientes de Kramers-Moyal (resumen)

- delta_t: 1.0, n_bins: 50
- n_used: 8814

## 5. Test de Pawula

| Orden | mean(|R_k|) | median | max |
|---|---|---|---|
| 4 | 3.105e-01 | 2.356e-01 | 1.516e+00 |
| 6 | 8.664e-02 | 2.782e-02 | 1.274e+00 |

## 6. Comparacion de modelos (vs empirico)

| Modelo | KS | JS | err mean | err std | err skew | err kurt |
|---|---|---|---|---|---|---|
| M0_gaussian | 9.655e-01 | 6.377e-01 | 1.020e+02 | 4.547e+01 | 1.021e+00 | 1.120e+01 |
| M1_OU | 9.881e-02 | 2.267e-02 | 1.139e-02 | 5.041e-03 | 3.887e-01 | 1.066e+01 |
| M2_langevin | 2.033e-01 | 1.116e-01 | 4.628e-01 | 3.986e+00 | 8.129e-01 | 5.027e+00 |

## 7. Figuras

- **price_and_returns**: `reports\figures\price_and_returns.pdf`
- **pdf_vs_normal**: `reports\figures\pdf_vs_normal.pdf`
- **acf**: `reports\figures\acf.pdf`
- **km_coefficients**: `reports\figures\km_coefficients.pdf`
- **potential_M2**: `reports\figures\potential_M2.pdf`
- **rolling_km**: `reports\figures\rolling_km.pdf`
- **sim_vs_emp_M0_gaussian**: `reports\figures\sim_vs_emp_M0_gaussian.pdf`
- **sim_vs_emp_M1_OU**: `reports\figures\sim_vs_emp_M1_OU.pdf`
- **sim_vs_emp_M2_langevin**: `reports\figures\sim_vs_emp_M2_langevin.pdf`