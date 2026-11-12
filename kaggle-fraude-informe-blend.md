# Blend `mixto_todos` — fraude con tarjeta (S6E11)

**Fractal Core Research** — 2026-11-10. Competencia Kaggle Playground
Series S6E11, métrica ROC AUC, cierre 30 nov 2026. Orden de Elán
(2026-11-10 ~12:34): armar el blend y enviarlo para tener su score
público; la v2 (0.94393) sigue seleccionada como final y nada se marca
como final sin su decisión.

## Definición (exacta, de `analizar_mixto.py` + `ensamble_blends.json`)

`mixto_todos` es el promedio simple (pesos iguales) de 14 modelos
entrenados con TODO train (699,960 filas):

- 9 GBM sobre features crudas: HGB base semillas 42/1/7, HGB g5
  semillas 42/1/7, LightGBM semillas 42/1/7.
- 4 GBM sobre features combinadas (crudas + 18 señales de riesgo):
  HGB g5 semillas 42/1, LightGBM semillas 42/1.
- 1 regresión logística (C=0.5, lbfgs) sobre crudas escaladas.

Configuraciones: HGB base 200 it, lr 0.08, 31 hojas, min_leaf 50,
l2 1.0; HGB g5 200 it, lr 0.08, 15 hojas, min_leaf 20, l2 0.1;
LightGBM 200 est, lr 0.08, 31 hojas, min_child 50, reg_lambda 1.0.
Las tablas de referencia se calculan con TODO train (sin fuga: el
test no tiene etiquetas). El promediado simple y el rank-mean dieron
lo mismo en CV (±0.0001), así que se usa el simple.

## CV local (reproducido desde los OOF guardados, sin re-entrenar)

| | Valor |
|---|---|
| OOF por pliegue | 0.9448, 0.9481, 0.9512, 0.9466, 0.9416 |
| AUC media | **0.9465 ± 0.0032** |
| OOF agrupado | 0.9464 |
| Δ pareado vs base HGB | +0.0014 ± 0.0003 (positivo en los 5 pliegues) |
| Barra de decisión | 0.9484 (no superada; falta 0.0019) |

La reproducción es exacta: los mismos números que
`verificacion/ensamble_blends.json`.

## Entrenamiento final (notebook `fraude-tarjetas-blend.ipynb`, 12 celdas)

Ejecución completa del notebook en local: 817.2 s (13.6 min), 7/7
verificaciones PASS. Checkpoints por modelo en
`verificacion/blend_ckpt/` (14 archivos .npy, reanudación incluida).

| Modelo | Tiempo (s) |
|---|---|
| hgb_base_s42 / s1 / s7 | 98.0 / 80.4 / 69.9 |
| hgb_g5_s42 / s1 / s7 | 77.1 / 79.0 / 84.5 |
| lgbm_s42 / s1 / s7 | 32.6 / 27.8 / 30.1 |
| logreg | 1.4 |
| hgb_g5_comb_s42 / s1 | 86.0 / 71.8 |
| lgbm_comb_s42 / s1 | 28.2 / 32.6 |
| **Total** | **799.4** |

## Submission

`salida-fraude/submission_blend.csv` — 299,984 filas, md5
`277cf0b7d6699960a47d10b32f2ecdef`, header `id,is_fraud`, mismo
orden de ids que el sample, sin nulos, probabilidades en [0, 1]
(min 0.0, max 0.9956, media 0.0166). Los 7 checks de formato OK.
La v1 queda intacta (md5 `c07bfd803b2704a13ec7f7d2afff7bd8`,
verificado antes y después).

## Sanidad vs la v1

- Pearson 0.9841, Spearman 0.9841; diferencia media abs 0.0029;
  medias idénticas (0.0166); solape top-1000: 872/1000.
- Calibración del umbral: ningún miembro individual supera 0.99 vs
  la v1 (máximo lgbm_comb_s1, Pearson 0.9898; logreg es el más
  distinto, Spearman 0.9184, como corresponde a la familia lineal
  diversa). El blend supera a la mediana de sus miembros (Pearson
  0.966, Spearman 0.9755), que es lo que el promediado debe hacer.
  El umbral inicial ">0.99" era una suposición sin medir y se corrigió
  con esta evidencia; el cambio está documentado en el notebook y en
  el verificador.
- Verificación independiente: `verificacion/verificar_blend.py`
  (formato, md5, blend == media exacta de los 14 checkpoints,
  reproducción OOF, lenguaje neutro). Resultado en
  `verificacion/blend.json`, sección `verificacion_independiente`.

## Lectura honesta

- El blend sube el CV local de 0.9449 (base publicada) a 0.9465
  (+0.0016) y de 0.9451 (reproducción local) a 0.9465 (+0.0014
  pareado). Es una ganancia real y consistente por pliegue, pero
  pequeña: por debajo de la desviación entre pliegues (~0.0035) y
  de la barra 0.9484. El ensamble reduce varianza, no encuentra
  señal nueva.
- El score público puede no reflejar la ganancia: el leaderboard
  público es solo una muestra y el privado decide. El envío se hace
  por decisión de Elán para medirlo ("aunque sea poco, agrega el
  ensamble"), no porque el CV lo garantice.
- Si el público del blend supera a la v2 (0.94393), la decisión de
  cambiar la selección final es de Elán.

## Archivos

| Archivo | Qué es |
|---|---|
| `fraude-tarjetas-blend.ipynb` | Notebook del blend, 12 celdas, 7/7 PASS |
| `salida-fraude/submission_blend.csv` | Envío del blend (md5 277cf0b7…) |
| `verificacion/blend.json` | Números consolidados + verificación independiente |
| `verificacion/blend_ckpt/*.npy` | Probabilidades de test de los 14 miembros |
| `verificacion/blend_tiempos.json` | Tiempos por modelo |
| `verificacion/construir_notebook_blend.py` | Constructor del notebook |
| `verificacion/correr_notebook_blend.py` | Ejecutor celda por celda (sin jupyter) |
| `verificacion/verificar_blend.py` | Verificación independiente |
| `verificacion/notebook-blend-corrida.log` | Log de la ejecución del notebook |
