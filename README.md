# Fraude con tarjeta — Kaggle Playground Series S6E11

**Competencia:** Predicting Credit Card Fraud (Kaggle Playground Series S6E11).
Métrica **ROC AUC**, envío CSV `id,is_fraud` con probabilidades 0–1.
Cierre 30 nov 2026, 11:59 PM UTC.

## Estado

- **Envío hecho 2026-11-08:** AUC público **0.94375**, posición **1027 de 1142**
  (1 envío registrado). El modelo combinado (crudas + señales) entrenado con
  todo train; CV 5 pliegues 0.9448 ± 0.0031.
- **Notebook en Kaggle (privado):** https://www.kaggle.com/code/elnzainoscorona/fraude-tarjetas
  (12 celdas, 12/12 PASS en Kaggle) + dataset utilitario privado con el
  adaptador: https://www.kaggle.com/datasets/elnzainoscorona/fraude-tarjetas-adaptador
- **Ajuste ligero (2026-11-08): NO GANA NADA.** Rejilla HGB (10 configs),
  LightGBM, features extra, interacciones y desviaciones vs patrón del grupo:
  ninguna variante supera 0.9449 por más de la std (~0.0035). El submission
  NO se regeneró (md5 `c07bfd803b2704a13ec7f7d2afff7bd8`).
- **Datos externos (original CC0): PROBADO y DESCARTADO.** AUC idéntico con y
  sin externo; no se usó como entrenamiento extra.

Detalle completo en `kaggle-fraude-informe.md` (§1–§12, con las 3 figuras
incrustadas: autenticación, escalera del conteo de señales, cuota por
categoría).

## Método (resumen)

- **Conteo de señales de riesgo distintas** por transacción (10 definidas:
  extranjero, comercio nuevo, VPN, mismatch de país, mismatch de
  facturación/envío, intento de estafa por IA, sin autenticación, reintento
  de CVV ≥ 1, velocidad sobre p90, distancia sobre p90). Umbral ≥ 5: 549
  txns (0.08%), 87.61% de fraude, lift 52.46x. El conteo solo tiene AUC
  0.8702.
- **Autenticación:** No Authentication 6.44% de fraude vs Biometric 0.50%
  (37.0% de todo el fraude sin autenticación).
- **Concentración por categoría:** cuota de fraude por conteo y por monto
  (Electronics 46.9% del dinero defraudado).
- **Modelo:** HistGradientBoosting, CV estratificada 5 pliegues, sin fuga
  (las tablas de referencia se calculan solo con el pliegue de entrenamiento
  de cada fold). Veredicto honesto: las señales no añaden AUC sobre las
  crudas (0.9448 vs 0.9449); su valor es la lectura.
- **Serie temporal:** DATOS INSUFICIENTES (el id no es temporal; no se
  inventó ninguna serie). No hay ID de usuario en los datos.

## Estructura

| Ruta | Qué es |
|---|---|
| `fraude_tarjetas.py` | Adaptador v1.0.0: señales, cuota, autenticación, features, CV, entrenamiento final, envío y verificación. |
| `fraude-tarjetas.ipynb` | Notebook de 12 celdas (6 markdown + 6 código). Resuelve rutas Kaggle (`/kaggle/input/**`) o locales. |
| `kaggle-fraude-informe.md` | Informe completo (§1–§12) con las 3 figuras incrustadas. |
| `salida-fraude/` | `submission.csv` (299,984 filas), `informe-fraude.md` y las 3 figuras. |
| `verificacion/` | Verificación independiente, comparación con externo, ASI por tipo de tarjeta, logs del ajuste ligero y de las corridas del notebook. |
| `datos/` | Datos de la competencia (**no están en git**, ver abajo). |

## Datos (no están en git)

| Archivo | Filas | Nota |
|---|---|---|
| `datos/train.csv` | 699,960 × 26 | Fraude 1.67% (11,689). |
| `datos/test.csv` | 299,984 × 25 | |
| `datos/sample_submission.csv` | 299,984 × 2 | Constante 0.01669953. |
| `datos/credit-card-fraud-detection-2026.csv` | 20,000 × 26 | Original CC0 (probado y descartado, §7 del informe). |

Copia local de referencia:
`~/workspace/snt-taxonomia/kaggle-fraude-tarjetas/datos/` (los originales;
este repo es la copia de trabajo). Descarga: competición
`playground-series-s6e11` en Kaggle (sesión de Elán).

## Reglas

- 10 envíos/día, 2 finales seleccionables; el leaderboard privado decide.
- No se entrena con el test ni se infieren sus etiquetas. Sin AutoML, sin
  etiquetado del test, sin redistribuir los datos.
- Ningún envío nuevo sin decisión de Elán.
