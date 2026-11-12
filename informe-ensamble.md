# Ronda de ensamble — fraude con tarjeta (S6E11)

**Fractal Core Research** — 2026-11-09/10. Competencia Kaggle Playground
Series S6E11, métrica ROC AUC, cierre 30 nov 2026. Orden de Elán
(~17:44): intentar hacer más, no dejarlo en la v2 (pública 0.94393;
v1 pública 0.94375).

## Regla de decisión (vinculante, informe §12)

Regenerar `submission.csv` SOLO si una variante o ensamble supera la
baseline 0.9449 por más de la std entre pliegues (~0.0035), es decir
**AUC > 0.9484**. Si no se supera, el submission queda intacto y todo
lo probado se documenta como descartado. Nada a Kaggle, nada como
final.

## Protocolo (exacto del informe §5)

CV estratificada de 5 pliegues, semilla 42, ROC AUC. Las tablas de
referencia (cuota de fraude por categoría/canal) se calculan SOLO con
el pliegue de entrenamiento de cada fold: sin fuga. Los mismos folds
para todas las variantes, así que la comparación pareada por pliegue
es válida. Predicciones out-of-fold guardadas en
`verificacion/oof_ensamble.npz` (crudas) y
`verificacion/oof_ensamble_comb.npz` (combinadas).

Reproducción de la baseline: HGB base sobre crudas da 0.9451 ± 0.0035
con pliegues [0.9432, 0.9472, 0.9500, 0.9452, 0.9398], frente a la
publicada 0.9449 ± 0.0035 [0.9428, 0.9471, 0.9498, 0.9451, 0.9396].
Diferencia +0.0002: el protocolo se reproduce (la diferencia es ruido
de versión de librería: sklearn 1.6.1 aquí, 1.9.1 en la corrida
original).

## Fase 1 — variantes sobre features crudas

| Variante | AUC media | std | AUC OOF agrupado | Pliegues |
|---|---|---|---|---|
| hgb_base_s42 | 0.9451 | 0.0035 | 0.9450 | 0.9432, 0.9472, 0.9500, 0.9452, 0.9398 |
| hgb_base_s1 | 0.9451 | 0.0032 | 0.9451 | 0.9434, 0.9465, 0.9500, 0.9453, 0.9404 |
| hgb_base_s7 | 0.9450 | 0.0032 | 0.9449 | 0.9433, 0.9470, 0.9496, 0.9450, 0.9401 |
| hgb_g5_s42 | 0.9453 | 0.0033 | 0.9452 | 0.9433, 0.9468, 0.9501, 0.9458, 0.9403 |
| hgb_g5_s1 | 0.9451 | 0.0033 | 0.9451 | 0.9433, 0.9465, 0.9501, 0.9456, 0.9402 |
| hgb_g5_s7 | 0.9452 | 0.0034 | 0.9451 | 0.9433, 0.9467, 0.9501, 0.9457, 0.9400 |
| lgbm_s42 | 0.9454 | 0.0034 | 0.9454 | 0.9434, 0.9478, 0.9501, 0.9456, 0.9402 |
| lgbm_s1 | 0.9454 | 0.0032 | 0.9453 | 0.9438, 0.9472, 0.9499, 0.9457, 0.9403 |
| lgbm_s7 | 0.9455 | 0.0034 | 0.9454 | 0.9441, 0.9477, 0.9503, 0.9451, 0.9402 |
| logreg | 0.9419 | 0.0034 | 0.9419 | 0.9413, 0.9435, 0.9469, 0.9416, 0.9363 |

- HGB base: 200 árboles, lr 0.08, 31 hojas, min_samples_leaf 50, l2 1.0.
- HGB g5: mejor config del ajuste ligero: 200 árboles, lr 0.08,
  15 hojas, min_samples_leaf 20, l2 0.1.
- LightGBM: 200 estimadores, lr 0.08, 31 hojas, min_child_samples 50,
  reg_lambda 1.0.
- LogReg: regresión logística C=0.5 sobre features escaladas (familia
  diversa, lineal). Más débil sola (0.9419), aporta diversidad al blend.
- La variación entre semillas del mismo modelo es ±0.0001: el
  promediado de semillas apenas mueve el AUC; la ganancia del ensamble
  viene de mezclar familias, no semillas.

## Fase 2 — combinadas, mejores familias

| Variante | AUC media | std | AUC OOF agrupado | Pliegues |
|---|---|---|---|---|
| hgb_g5_comb_s42 | 0.9452 | 0.0033 | 0.9452 | 0.9434, 0.9463, 0.9501, 0.9461, 0.9402 |
| hgb_g5_comb_s1 | 0.9452 | 0.0028 | 0.9451 | 0.9432, 0.9461, 0.9493, 0.9460, 0.9412 |
| lgbm_comb_s42 | 0.9453 | 0.0031 | 0.9453 | 0.9431, 0.9470, 0.9498, 0.9459, 0.9409 |
| lgbm_comb_s1 | 0.9453 | 0.0029 | 0.9452 | 0.9432, 0.9463, 0.9497, 0.9459, 0.9413 |

Las combinadas (crudas + 18 señales) quedan igual que las crudas para
el mismo modelo (g5 0.9452 vs 0.9451–0.9453; lgbm 0.9453 vs
0.9454–0.9455): las señales no añaden AUC, como ya decía el informe
§5. Su valor en el ensamble es solo diversidad.

## Blends (pesos iguales, sin ajuste a OOF)

| Blend | Composición | AUC media | std | OOF | Δ pareado vs base |
|---|---|---|---|---|---|
| promedio_hgb_base | 3 semillas HGB base | 0.9456 | 0.0034 | 0.9456 | +0.0006 ± 0.0001 |
| promedio_hgb_g5 | 3 semillas HGB g5 | 0.9457 | 0.0033 | 0.9456 | +0.0006 ± 0.0003 |
| promedio_lgbm | 3 semillas LightGBM | 0.9458 | 0.0033 | 0.9457 | +0.0007 ± 0.0002 |
| gran_promedio_gbm | 9 GBM crudas | 0.9461 | 0.0033 | 0.9460 | +0.0010 ± 0.0002 |
| gran_promedio_todos | 9 GBM + logreg | 0.9462 | 0.0034 | 0.9461 | +0.0011 ± 0.0002 |
| g5_lgbm | 6 (g5 + lgbm crudas) | 0.9461 | 0.0033 | 0.9460 | +0.0010 ± 0.0002 |
| rankmean_gran_gbm | rank-mean 9 GBM | 0.9460 | 0.0033 | 0.9460 | +0.0010 ± 0.0002 |
| rankmean_todos | rank-mean 10 | 0.9461 | 0.0033 | 0.9461 | +0.0011 ± 0.0002 |
| promedio_comb | 4 combinadas | 0.9459 | 0.0030 | 0.9459 | +0.0008 ± 0.0007 |
| mixto_gbm | 13 GBM (9 crudas + 4 comb) | 0.9464 | 0.0032 | 0.9463 | +0.0013 ± 0.0003 |
| **mixto_todos** | **13 GBM + logreg** | **0.9465** | **0.0032** | **0.9464** | **+0.0014 ± 0.0003** |
| mixto_modos | ½ gran GBM + ½ comb | 0.9464 | 0.0031 | 0.9464 | +0.0014 ± 0.0004 |
| rankmean_mixto_gbm | rank-mean 13 GBM | 0.9464 | 0.0032 | 0.9463 | +0.0013 ± 0.0003 |
| rankmean_mixto_todos | rank-mean 14 | 0.9464 | 0.0032 | 0.9464 | +0.0013 ± 0.0003 |

La base de comparación pareada es hgb_base_s42 reproducida aquí
(0.9451). El promedio simple y el rank-mean dan lo mismo (±0.0001):
no hay nada que ganar cambiando la forma de promediar.

## Veredicto: NO GANA

El mejor blend (mixto_todos, 0.9465 ± 0.0032) supera la baseline
publicada (0.9449) por +0.0016 y a la reproducción local (0.9451) por
+0.0014 pareado. Ambas diferencias están por debajo de la std entre
pliegues (~0.0035) y por debajo de la barra 0.9484 (falta 0.0019).

La ganancia del ensamble es real y consistente por pliegue (Δ pareado
+0.0014 ± 0.0003, positivo en los 5 pliegues), pero pequeña: el
ensamble reduce varianza, no encuentra señal nueva. Es el mismo patrón
que el ajuste ligero (§12): todo se mueve dentro del ruido.

**Consecuencia: NO se generó submission nuevo.** `salida-fraude/submission.csv`
queda intacto (md5 `c07bfd803b2704a13ec7f7d2afff7bd8`, verificado antes
y después de la ronda).

## Lectura honesta

- El techo de estos datos con este protocolo está cerca de 0.946–0.947
  en CV. El modelo ya extrae casi toda la señal que las columnas
  crudas traen; las señales derivadas, las interacciones, las
  desviaciones de grupo y ahora el ensamble multi-familia mueven el
  AUC en ±0.001.
- La diferencia entre la v2 pública (0.94393) y la v1 (0.94375) es
  +0.00018: el mismo orden de magnitud que estas ganancias. Un envío
  del ensamble podría subir el público en ~+0.001, pero no hay forma
  de saberlo sin enviar, y la regla de decisión prohíbe regenerar por
  debajo de la barra.
- Si se quisiera seguir buscando, el camino no es otro modelo sobre
  las mismas columnas: haría falta señal nueva (identificador de
  tarjeta o comercio para patrones por entidad, serie temporal real),
  y los datos no la traen (§4, §9 del informe).

## Incidentes

- **Reinicio de la caja ~00:07** durante la fase 1: el proceso murió
  a mitad del pliegue 2 y los paquetes pip (sklearn, lightgbm)
  desaparecieron. Se reinstaló sklearn 1.6.1 + lightgbm 4.7.0 con las
  mismas versiones y se añadió checkpoint por pliegue con reanudación
  a `ensamble.py` y `ensamble_fase2.py` (no se perdió más trabajo).
- **ExtraTrees descartado:** 300 árboles, profundidad 12, midió >12
  min por ajuste en un solo pliegue en esta caja (1.5 CPU
  compartidas). Su aporte marginal a un blend de 10+ miembros no
  compensa ~60 min extra. La regresión logística queda como familia
  diversa.
- La caja borró los paquetes pip dos veces en el día (17:00 y ~00:07);
  el entorno se reinstaló con las mismas versiones en ambos casos
  (numpy sigue 1.26.4, no se actualizó a 2.x).

## Archivos

| Archivo | Qué es |
|---|---|
| `ensamble.py` | Fase 1: 10 variantes sobre crudas, CV 5 pliegues, OOF |
| `ensamble_fase2.py` | Fase 2: 4 variantes sobre combinadas, mismo protocolo |
| `analizar_ensamble.py` | Blends sobre crudas (media y rank-mean) |
| `analizar_mixto.py` | Blends mixtos crudas + combinadas |
| `verificacion/oof_ensamble.npz` | Probabilidades OOF, fase 1 (10 variantes) |
| `verificacion/oof_ensamble_comb.npz` | Probabilidades OOF, fase 2 (4 variantes) |
| `verificacion/ensamble_fase1.json` | AUCs por variante, fase 1 |
| `verificacion/ensamble_fase2.json` | AUCs por variante, fase 2 |
| `verificacion/ensamble_blends.json` | AUCs por blend + deltas pareados |
| `verificacion/ensamble.json` | Números consolidados + veredicto |
| `verificacion/ensamble-corrida.log` | Log fase 1 |
| `verificacion/ensamble-corrida-fase2.log` | Log fase 2 |
| `informe-ensamble.md` | Este informe |

Tiempos: fase 1 2,016 s (~34 min), fase 2 1,491 s (~25 min).
Verificación `verificacion/verificar_fraude.py` corrida al final y en
verde (ver `verificacion/verificacion-fraude.json`).
