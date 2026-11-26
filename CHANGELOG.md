# CHANGELOG — Fraude Tarjetas S6E11

Historial verificado del proyecto. Fechas en hora central (CST).

## 2026-11-08

- **Adaptador + notebook construidos.** `fraude_tarjetas.py` v1.0.0
  (señales de riesgo) y `fraude-tarjetas.ipynb` (12 celdas). CV 5-fold
  estratificado (HistGradientBoosting, sin leakage): solo crudo
  0.9449 ± 0.0035; solo señales 0.9075 ± 0.0054; combinado
  0.9448 ± 0.0031. **Las señales NO suman AUC** (Δ −0.0001);
  interpretabilidad sí. Escalera de conteo: 0→0.19% … 5→85.40% …
  7–8→100% fraude; umbral ≥5: 549 transacciones, 87.61% fraude, lift
  52.46x. Informe `kaggle-fraude-informe.md` (§1–§12).
- **submission.csv** (299,984 filas, md5
  `c07bfd803b2704a13ec7f7d2afff7bd8`).
- **Envío v1 a Kaggle ~17:03: público ROC AUC 0.94375**, lugar 1027
  de 1142.
- **ASI por tipo de tarjeta ~17:57** (orden expresa de Elán; única
  excepción a la regla de no-teoría, en términos neutros): ASI
  ≈ 0.20–0.21 en los cinco tipos (Amex 0.210889, Discover 0.213645,
  Mastercard 0.202934, RuPay 0.210396, Visa 0.207073, global 0.20615).
  El umbral >1, calibrado en HackerEarth, no transfiere (F ≈ 2.5).
  Informe §11; `verificacion/asi_por_tarjeta.json`.
- **Ajuste ligero completado ~19:49** (7,090.1 s): **NO GANA NADA**
  (mejor g5 0.9454 ± 0.0031; LightGBM 0.9453). Submission intacto.
  Informe §12.
- **Notebook subido a Kaggle ~22:24 (PRIVADO).** V1 falló en 24 s
  (ruta de Kaggle); fix aplicado idéntico local + Kaggle. **V2
  exitosa en 386.1 s: 12/12 PASS**, CV reproducido (combinado
  0.9448 ± 0.0031 exacto).
- **Envío v2 de Elán ~22:21** (por navegador, notebook "fraude-tarjetas
  - Version 2"; el archivo NO está en esta máquina): **público ROC AUC
  0.94393** (+0.00018 sobre v1), **seleccionado como final**.
- **3 figuras de fraude incrustadas en el informe ~23:27**
  (corrección de Elán: hablaba del proyecto de tarjetas, no de la
  Sombra).

## 2026-11-09

- **Repos creados ~09:45** (orden de Elán): este repo
  (`fraude-tarjetas-s6e11`, git local, sin remote). README en
  lenguaje neutro (cero menciones de la SNT, por la regla de Elán).
- **Respaldo en Drive ~11:49.** Respaldos Fractal Core / Fraude
  Tarjetas S6E11 / `respaldo-fraude-s6e11-2026-11-09.zip`
  (34,669,692 B, verificado de vuelta; informe, submission, notebook,
  adaptador, figuras y datos incluidos).
- **Ronda de ensamble completada ~19:11** (orden de Elán ~17:44
  "intentar hacer más, no dejarlo en la v2"): 14 modelos, mejor blend
  `mixto_todos` 0.9465 ± 0.0032 (Δ +0.0014 ± 0.0003, positivo en los
  5 folds); **barra 0.9484 NO superada — NO GANA**. Sin submission
  nuevo; v1 intacta (md5 c07bfd80… verificado antes y después).
  `informe-ensamble.md`; `verificacion/ensamble.json`. Commit
  `afcb5db`. Incidencia: reset de la caja ~00:07 durante la fase 1
  (paquetes de pip perdidos, reinstalados; checkpoints por fold).
- **Regla permanente de Elán ~21:00** (tablero §H): cada proyecto debe
  tener README + AGENTS.md y/o CLAUDE.md, todos CHANGELOG, y los
  serios logs de salida. Este AGENTS.md y CHANGELOG.md se crean por
  esa regla.

## 2026-11-10

- **Envío del blend `mixto_todos` a Kaggle: público ROC AUC 0.94662**
  (nuevo récord personal entonces; CV 0.9465 ± 0.0032).
- **Finales marcadas 2/2 y verificadas: blend (0.94662) + v2
  (0.94393).**

## 2026-11-20

- **P3 medido (Kali, venv `.venv-fraude`, 329.7 s): llaves de
  dígitos de amount_usd NO GANAN.** 6 llaves (primer dígito,
  primeros dos, último dígito, últimos dos, n dígitos, dólar
  entero; función determinística por fila, sin etiqueta, sin
  fuga). Deltas pareados: crudas HGB −0.0002 (2/5), crudas
  lgbm_l4_it500 +0.0000 (2/5), combinadas HGB +0.0001 (3/5),
  combinadas lgbm_l4_it500 −0.0000 (0/5); ninguno pasa el corte
  +0.0005. **El ~+0.0015 del foro no se reproduce.** Diagnóstico
  de artefacto (todo train): último dígito 1.51%–1.79% fraude
  (dispersión 0.0028, casi uniforme); primer dígito 1.35%
  (dígito 3) a 2.07% (dígito 4), dispersión 0.0072 — confundido
  con la magnitud del monto (Benford + distribución); dólar
  entero 2.09% (n=6,559) vs 1.67% resto. Los modelos ya ven
  amount_usd: los dígitos no añaden nada. Medición solamente;
  Elán decide si se usa (no hay nada que usar: delta ≈ 0).
  `verificacion/p3_llaves_digitos.py/.json/.log` (JSON md5
  `57ab63b8…`, idéntico Kali y local). Reproducción exacta de
  los 4 baselines (dif media +0.0000).
- **P4 medido (Kali, venv `.venv-fraude`, 180.3 s): boosting
  desde GLM GANA en los dos modos.** API nativa
  (`lgb.train`, `init_score` = logit del GLM; la API sklearn
  no tiene init_score). Predicción: sigmoid(raw + logit_glm).
  Comprobación de cordura: con 1 ronda, predict(raw) == valor
  de hoja (sin_init e init_0.5, dif 0.00e+00) — el predict no
  añade init ni base; el margen completo es logit_glm +
  árboles. (Corrección: el primer intento de cordura comparaba
  init_0 contra sin_init y falló — con razón: sin init el punto
  de partida es el log-odds de la media del label, con init_0
  es 0; los árboles difieren porque los gradientes difieren.
  La comprobación correcta es raw contra hojas.) GLM propio
  0.9419 en ambos modos (= logreg 0.9419 de referencia).
  Deltas LIMPIOS pareados (nativo_glm vs nativo_cero, mismo
  código, solo cambia el init): **crudas +0.0006 ± 0.0003
  [0.0005, 0.0010, 0.0007, 0.0, 0.0008], 4/5 positivos**;
  **combinadas +0.0010 ± 0.0005 [0.0017, 0.0012, 0.0009, 0.0002,
  0.0010], 5/5 positivos** — ambos pasan el corte +0.0005.
  Contra el baseline sklearn: crudas +0.0005, combinadas
  +0.0009. Medias nativo_glm: crudas 0.9472, combinadas
  0.9473 (el blend v3 CV es 0.9466; un solo lgbm con init GLM
  lo supera en CV; la barra 0.9484 NO se alcanza). **No hay v4
  construido: solo medición — Elán decide si se integra el
  init GLM al blend y si se envía.** `verificacion/
  p4_boosting_desde_glm.py/.json/.log` (JSON md5 `b6914f90…`,
  idéntico Kali y local). Reproducción exacta de los 2 baselines
  sklearn (dif media +0.0000). Submissions 4/4 intactos en
  ambas cajas (md5 antes/después). Lenguaje neutro verificado
  con grep (cero menciones prohibidas en los 6 archivos nuevos).

- **Blend v4 CONSTRUIDO (GO de Elán 03:23, por widget).** Mismo
  esqueleto que el v3 (14 miembros, mismos hiperparámetros,
  mismas semillas, media simple), con los 5 LightGBM
  inicializados desde GLM por API nativa (lgb.train, init_score
  = logit del GLM, especificación P4 exacta; comprobación de
  cordura repetida: PASS 0.00e+00). Los otros 9 miembros quedan
  igual que en v3. Corrida en Kali (venv `.venv-fraude`,
  301.6 s): **blend v4 CV pooled 0.9469** (por pliegue [0.9454,
  0.9486, 0.9517, 0.9470, 0.9421], media 0.9470 ± 0.0032) vs v3
  pooled 0.9466 — **delta pareado +0.0003 ± 0.0001, 5/5
  pliegues positivos**; la reproducción del v3 desde los OOF
  guardados es EXACTA. Deltas por miembro vs v3 (mismo slot y
  semilla): s42 +0.0005 (4/5), s1 +0.0006 (5/5), s7 +0.0007
  (5/5), comb_s42 +0.0009 (5/5), comb_s1 +0.0009 (5/5). GLM de
  init CV 0.9419 en ambos modos. **La barra 0.9484 NO se
  alcanza** (0.9469). `salida-fraude/submission_blend_v4.csv`
  (299,984 filas, md5 `f03b3a7e…`, 7/7 checks, media_prob
  0.0166); los 4 submissions viejos intactos antes/después en
  ambas cajas. `verificacion/blend_v4.py/.json/.log`
  (JSON md5 `76c5ef42…`, idéntico Kali y local) +
  `oof_blend_v4_lgbm_glm.npz`. **Nada a Kaggle, nada de marcar
  finales: el envío y la selección son decisiones aparte de
  Elán.** Cierre 30 nov 2026.

## 2026-11-13

- **Auditoría externa recibida ~20:59** (agente externo, vía Elán;
  leyó el repo público, ejecutó nada; Kaggle le respondió
  "Unauthenticated"; ganancias del foro, no medidas). Propuestas
  P0–P6. **Verificación de Mole ~21:00:** barra 0.9484 y delta
  pareado ya estaban en AGENTS.md:84; `n_iter_` SÍ se lee en
  `capacidad_profunda.py` pero `early_stopping` nunca se fijó en
  `ajuste_ligero.py:93-103` (punto real); repo público pero clon sin
  remote (regla 4 desactualizada); ganancias P1–P6 sin verificar.
- **GO de Elán ~21:04: escribir el pipeline v2 (P0–P4).** El agente
  externo reportó que su notebook/cálculo se detuvo de su lado;
  revisión en esta caja, Kali y Drive: ningún notebook de fraude
  corriendo ni detenido aquí (última actividad 12 nov) — la
  continuación se hizo aquí.
- **P0 medido (Kali, venv `.venv-fraude`): early stopping 'auto' SÍ
  cortaba** (base combinadas n_iter [200,97,131,168,137]) **pero NO
  movía el AUC** (delta pareado ≤ +0.0005 en las 5 configs);
  reproducción de los AUC registrados exacta (±0.0006/pliegue).
  `verificacion/p0_early_stopping.py/.json/.log` (620.2 s).
- **`ensamble_final.py` completo:** 14 miembros con todo train →
  `salida-fraude/submission_blend_v2.csv` (299,984 filas, md5
  `6b47d39c…`, 7/7 checks, media prob 0.0166); OOF pooled 0.9464 =
  documentado; submissions existentes intactos (md5 antes/después).
- **P1 medido: LightGBM 4 hojas/500 it GANA +0.0013 pareado
  (crudas) y +0.0011 (combinadas, 5/5); 8 hojas +0.0010/+0.0012.**
  Un solo lgbm_l4_it500 (0.9467 crudas) iguala al blend de 14
  (0.9465).
- **Blend v3 reconstruido (GO de Elán ~22:10):** 5 LightGBM a 4
  hojas/500 it, otros 9 igual; **CV 0.9466 vs v2 0.9465, delta
  pareado +0.0002 (5/5); la estimación ~0.9475 NO se materializó.**
  `salida-fraude/submission_blend_v3.csv` (299,984 filas, md5
  `df1e532c…`, 7/7 checks). `verificacion/blend_v3.py/.json/.log`
  (264.1 s en Kali).
- **Envío v3 a Kaggle 22:28:22 (GO de Elán ~22:27, por navegador):
  público ROC AUC 0.94691, lugar 1167, equipo Fractal Core — nuevo
  récord personal** (mejora 0.94662). Estado Complete; 10
  envíos/día: 10 → 9 restantes.
- **Finales actualizadas ~22:32 (GO de Elán ~22:31): 2/2 =
  submission_blend_v3.csv (0.94691) + submission_blend.csv
  (0.94662); retirado el v2 (0.94393).** Kaggle confirmó:
  "'submission_blend_v3.csv' will be evaluated for your final
  score".
- **P2 en curso (GO de Elán ~22:37):** target encoding con OOF
  anidado, en Kali (control no-anidado para demostrar la inflación
  del 0.981 del foro). Si gana, la integración al blend sería un v4
  (sin construir sin decisión).

- **P2 medido (Kali, 467.8 s): target encoding anidado NO GANA.** 5
  columnas categóricas (merchant_category 12, card_type 5, auth_method
  5, channel 5, device_type 8; smoothing m=100). Deltas pareados
  anidados: crudas HGB −0.0001 (0/5), crudas lgbm_l4_it500 −0.0001
  (0/5), combinadas HGB +0.0000 (3/5), combinadas lgbm_l4_it500
  +0.0000 (2/5); ninguno pasa el corte +0.0005. **Corrección: el
  control no anidado NO muestra inflación** (−0.0001/−0.0000/+0.0002/
  +0.0001) — el 0.981 del foro no se reproduce ni con fuga (categorías
  ≥5,285 filas; la autofuga es despreciable). El TE no suma porque no
  es información nueva (one-hot ya da la identidad; combinadas ya
  tiene cuota_fraude_categoria). **No hay v4 por P2.** Siguen P3 y P4.
  `verificacion/p2_target_encoding.py/.json/.log`. (Nota: la frase
  "la inflación queda demostrada" del JSON es falsa contra sus
  propios números; corregida en el reporte.)

- **Notebook del blend v3 creado y validado (orden de Elán ~22:48).**
  `fraude-tarjetas-blend-v3.ipynb` (12 celdas: 5 markdown + 7 código,
  29,425 B, md5 `1cf62b6c…`), construido por
  `verificacion/construir_notebook_blend_v3.py` (md5 `f14383bb…`) desde
  `ensamble_final.py` + `blend_v3.py` (5 LightGBM a 4 hojas/500 it;
  checkpoints nuevos en `verificacion/blend_v3_ckpt/`). Validación
  completa en Kali (venv `.venv-fraude`): corrida 1 en 124.4 s (full
  train), corrida 2 en 11.9 s (checkpoints); **12/12 verificaciones
  PASS, veredicto LISTO** (`verificacion/blend_v3_notebook.json`,
  logs `notebook_blend_v3_validacion.log` y
  `notebook_blend_v3_validacion_run2.log`): OOF v2 y v3 reproducidos
  exactos, delta pareado +0.0002 en 5/5, 7/7 checks, 3 submissions
  existentes intactos. Acuerdo numérico vs la referencia: dif máx
  abs 4.463e-14, media 2.811e-16, 0 filas >1e-9 (Pearson/Spearman
  1.000000). **El md5 NO coincide** (e7de3b7c… vs df1e532c…): el
  full-train no es bit-determinista (orden de suma en coma flotante
  entre hilos); el md5 quedó como informativo y la verificación
  exige acuerdo numérico < 1e-9 contra la instantánea
  `verificacion/submission_blend_v3_referencia.csv` (md5 df1e532c…).
  La copia de Kali se restauró a la referencia tras validar (los 4
  md5 coinciden en ambas cajas). Correcciones al construirlo:
  crudas 54 columnas y combinadas 72 (el notebook del v2 dice 60/78,
  desactualizado); la frase del P2 sobre la inflación se corrigió
  (el control no anidado no infla). Lenguaje neutro verificado con
  grep (cero menciones prohibidas). Listo para subir a Kaggle cuando
  Elán lo ordene; nada enviado.

- **Notebook del blend v3 subido a Kaggle ~23:12** (orden de Elán "Okay,
  súbelo"; por navegador): https://www.kaggle.com/code/elnzainoscorona/fraude-tarjetas-blend-v3,
  privado, Version 1 (Quick Save, sin ejecutar; 7.2 s, 0 salidas).
  Entradas verificadas: competencia playground-series-s6e11
  (train/test/sample) + dataset fraude-tarjetas-adaptador. El
  notebook "fraude-tarjetas" existente no se tocó. Sin envío a la
  competencia.

- **2026-11-25 — consultor ensamble_v2 adoptado; público 0.95112 (nuevo
  récord); finales consultor v2 + v4; frente cerrado.** El consultor
  empujó la rama `ensamble-v2` (`98b341f5…`): `ensamble_v2.py`, 6
  candidatos con nuestro protocolo (StratifiedKFold 5, seed 42, TE
  anidado, CPU). Veredicto (Kali, 467 s, tercer intento; los dos
  primeros murieron por reinicios de la Kali): glm_lgb4 0.95066
  (Δ +0.00401 vs lgb4_raw, +5/5) GANA; blend_rank 0.95028; glm 0.95034;
  lgb4_te 0.95000; lgb4_raw 0.94665. GO Elán 18:15 (adoptar glm_lgb4);
  `--final --mejor glm_lgb4` 18:16–18:25 (CV repetido 0.95066, +5/5);
  `salida-fraude/submission_consultor_v2.csv` (299,984 filas, 0 NaN,
  md5 `9f79f63fa5bd532869834d421d3bea63`, 8,576,929 B). "Subanlo" 18:35
  → Kaggle ref **58917706** (conector custom.kaggle, total_bytes
  8576929 exacto). **Público 0.95112 (18:43, COMPLETE)**, +0.00389
  sobre la v4 y +0.00046 sobre el CV. **Finales cambiadas por Elán
  18:46: consultor v2 + v4** (v3 fuera). "Prueba todo" (Elán 18:50):
  TE + GLM-init ya era el glm_lgb4 (sin corrida duplicada); stacking
  + blend ponderado 109 s (blend anidado 0.95072, +0.00006, empate;
  stack_lr 0.95037; stack_lr_rank 0.95028); tuning 762 s (A 0.95066
  exacto, B 0.95068 ruido, C 0.95060, D 0.95058, E 0.95063) — nada
  supera al glm_lgb4, sin submission nuevo. Detalle en
  `verificacion/VEREDICTO_CONSULTOR_2026-11-25.md`. GitHub (orden
  Elán 19:55): commit en la línea local + bundle nuevo para el push
  --force a local/fraude-2026-11-25 desde Katana; main NO se toca.
