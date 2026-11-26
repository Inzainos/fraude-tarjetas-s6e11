# AGENTS.md — Fraude Tarjetas S6E11

Reglas operativas para cualquier agente que trabaje en este repo.
Proyecto: Kaggle Playground Series S6E11 "Predicting Credit Card Fraud"
(métrica ROC AUC; submission `id,is_fraud`; 10 envíos/día; 2 finales).
Chat experto: "Qué es la entropía" (61e8d7ad). Coordinador: chat
"Coordinador" (aa46da50). Tablero: `~/workspace/coordinacion/TABLERO.md`.

## Reglas que no se rompen

1. **NUNCA ASUMAS, SIEMPRE VERIFICA.** Antes de reportar un resultado,
   léelo del archivo o del log. El estado "running/active" de un agente
   no es evidencia: se revisan los LOGS.
2. **REGLA DE LENGUAJE (Elán, 2026-11-08 ~13:54, vinculante): el
   adaptador, el notebook, el informe y TODOS los entregables de fraude
   NO deben mencionar la Shadow Node Theory en absoluto.** Nada de
   "SNT", "nodo sombra", ni vocabulario muro/Roche/ASI/luz-sombra-hub.
   Lenguaje neutro de fraude/ML únicamente. Única excepción, por orden
   expresa de Elán (~17:54, 2026-11-08): ASI por tipo de tarjeta en el
   §11 del informe, en términos neutros. Verificar con grep antes de
   entregar.
3. **Nada a Kaggle sin decisión de Elán.** Ningún envío, ningún cambio
   de selección final (finales: consultor v2 0.95112 ref 58917706 + blend
   ponderado v2 0.95119 ref 58925302, cambiadas 2026-11-25 21:59 por GO
   de Elán; la v4 0.94723 ref 58607873 quedó fuera), ningún
   notebook/dataset público.
4. **GitHub — regla actualizada por orden de Elán 2026-11-25 19:55
   ("Actualiza todo en guthub").** Remote: origin
   https://github.com/Inzainos/fraude-tarjetas-s6e11.git. Ramas: main @
   16d3dee; ensamble-v2 @ 98b341f5 (consultor, ya en GitHub);
   local/fraude-2026-11-25 (línea local, destino del push). **La línea
   local (HEAD 0cc5ece y siguientes) NO desciende de main 16d3dee
   (historias sin ancestro común): el push es --force solo a
   local/fraude-2026-11-25 y NUNCA a main sin decisión aparte de Elán.
   El push no se puede hacer desde esta caja (sin ruta de auth): se
   hace desde Katana con el bundle.** La regla anterior ("Nada a
   GitHub, trabajo solo en local") quedó superada por la orden de Elán.
5. **Respaldos a Google Drive**, carpeta "Respaldos Fractal Core /
   Fraude Tarjetas S6E11" (regla de Elán 2026-11-09 ~11:46). Nada de
   respaldos solo en local. GitHub = código versionado; Drive = respaldos.
6. **Código completo y con logs de ejecución.** Nada de fragmentos con
   "insertar resto aquí". Cada corrida deja su log en `verificacion/`.
7. **Los submissions existentes no se tocan.**
   `salida-fraude/submission.csv` (md5 c07bfd803b2704a13ec7f7d2afff7bd8),
   `salida-fraude/submission_blend.csv` (md5
   277cf0b7d6699960a47d10b32f2ecdef),
   `salida-fraude/submission_blend_v2.csv` (md5
   6b47d39cfb33131bc9a10b4572805cae),
   `salida-fraude/submission_blend_v3.csv` (md5
   df1e532c709ae521e3d187fd0c03128f),
   `salida-fraude/submission_blend_v4.csv` (md5
   f03b3a7e230419c478851190e88a8ac6) y
   `salida-fraude/submission_blend_v5.csv` (md5
   3f2a97426f89ebc982227d0cd2db1aab, construido 2026-11-25 por P5)
   se verifican por md5 antes y después de cada paso.
8. **Cierre de la competencia: 30 nov 2026, 11:59 PM UTC.**

## Entorno (esta caja)

- Python 3.12.3 del sistema: numpy 1.26.4, pandas 2.1.4, scipy 1.11.4.
  **Nunca actualizar numpy a 2.x** en el python del sistema.
- **La caja se resetea y los paquetes de pip desaparecen.** Tras un
  reset, reinstalar lo que falte con `pip --break-system-packages`:
  scikit-learn 1.6.1, lightgbm 4.7.0. (ExtraTrees se descartó: >12 min
  por ajuste en un fold con 1.5 CPU compartidas.)
- ~7 GB RAM, 4 CPU (1.5 compartidas efectivas). **Un solo proceso pesado
  a la vez.** El ajuste ligero completo tomó 7,090.1 s; la fase 1 del
  ensamble 2,016 s y la fase 2 1,491 s.
- Sin config global de git. Identidad local en los commits de Elán:
  deamon / deamon@local.

## Estructura

- `fraude_tarjetas.py` — adaptador v1.0.0 (señales de riesgo).
- `fraude-tarjetas.ipynb` — notebook, 12 celdas (con el fix de ruta de
  Kaggle; md5 c6a96126…; `.bak-v1` es la versión anterior).
- `kaggle-fraude-informe.md` — informe §1–§12, con las 3 figuras
  incrustadas (fig-auth, fig-conteo-senales, fig-cuota-fraude-categoria).
- `informe-ensamble.md` — ronda de ensamble (veredicto NO GANA).
- `salida-fraude/` — submission.csv + informe + 3 figuras.
- `verificacion/` — logs de corridas (`ajuste-ligero-corrida*.log`,
  `ensamble-corrida*.log`, `notebook-corrida*.log`), JSONs de métricas
  (`ajuste_ligero.json`, `ensamble.json`, `asi_por_tarjeta.json`),
  OOF (`oof_ensamble*.npz`), scripts de verificación.
- `datos/` — train.csv (699,960 × 26), test.csv (299,984 × 25),
  sample_submission.csv. **Fuera de git** (.gitignore). Tasa de fraude
  1.67% (11,689). Datos sintéticos.

## Estado verificado (2026-11-09 ~21:00)

- CV 5-fold estratificado (HistGradientBoosting, sin leakage): solo
  crudo 0.9449 ± 0.0035; solo señales 0.9075 ± 0.0054; combinado
  0.9448 ± 0.0031. **Las señales derivadas NO suman AUC** (Δ −0.0001);
  su valor es interpretabilidad.
- Escalera de conteo de señales: 0→0.19%, 1→1.06%, 2→5.40%,
  3→22.38%, 4→59.66%, 5→85.40%, 6→98.77%, 7–8→100% fraude.
- Envío v1 (2026-11-08 ~17:03): **público ROC AUC 0.94375**, lugar
  1027 de 1142.
- Envío v2 de Elán (2026-11-08 ~22:21, por navegador; el archivo NO
  está en esta máquina): **público ROC AUC 0.94393** (+0.00018 sobre
  v1), **seleccionado como final**.
- Ajuste ligero (2026-11-08 ~19:49): **NO GANA NADA** (mejor 0.9454
  ± 0.0031, ruido; LightGBM 0.9453). Submission intacto.
- Ronda de ensamble (2026-11-09 ~19:11): mejor blend `mixto_todos`
  0.9465 ± 0.0032, Δ +0.0014 ± 0.0003 real pero chico; **barra 0.9484
  NO superada — NO GANA**. Sin submission nuevo.
- Pendiente de Elán: ¿se sube el blend del ensamble o se cierra en la
  v2?; ¿el notebook/dataset se hacen públicos?

## Estado verificado (2026-11-13 ~22:40)

- Envíos a Kaggle: v1 0.94375, v2 (Elán) 0.94393, blend 0.94662, v3
  0.94691 (lugar 1167, nuevo récord personal).
- Finales 2/2: blend (0.94662) + v3 (0.94691), marcadas 2026-11-13
  ~22:32.
- v2: P0 (early stopping cortaba, no movía AUC), P1 (LightGBM 4
  hojas/500 it +0.0013), blend v3 (CV 0.9466, +0.0002).
- P2 (target encoding anidado) en curso en Kali. Cierre 30 nov 2026.

## Estado verificado (2026-11-25 ~09:20)

- Finales 2/2: **v4 (0.94723) + v3 (0.94691)**, cambiadas por Elán
  2026-11-20 ~04:42 (se deseleccionó el blend, se seleccionó el v4,
  se mantuvo el v3). **Verificado en vivo en Kaggle 2026-11-25 ~09:20**
  (conector, grupo Selected: v4 ref 58607873 + v3 ref 58331285).
  El blend (0.94662) está FUERA; v2 y v1 tampoco están.
- Envíos a Kaggle (todos COMPLETE): v1 0.94375 (8 nov), v2 0.94393
  (9 nov), blend 0.94662 (10 nov), v3 0.94691 (14 nov), **v4 0.94723
  (20 nov 04:11:52 CST, nuevo récord personal)**. Cierre 30 nov 2026,
  11:59 PM UTC (5 días). Límite 10 envíos/día; 10/10 restantes el
  25 nov. Leaderboard top 0.97051; rank Elán 2001/3108.
- P2 (target encoding anidado) NO GANA: deltas −0.0001..+0.0002,
  ninguno pasa el corte +0.0005. CERRADO.
- P3 (llaves de dígitos de amount_usd) NO GANA: deltas
  −0.0002/+0.0000/+0.0001/−0.0000; el ~+0.0015 del foro no se
  reproduce. CERRADO.
- P4 (boosting desde GLM, init_score = logit del GLM, API nativa de
  LightGBM) GANA: crudas +0.0006 ± 0.0003 (4/5 pliegues), combinadas
  +0.0010 ± 0.0005 (5/5). Blend v4 (mismo esqueleto que v3, 14
  miembros, 5 LightGBM con init GLM): CV pooled 0.9469 vs v3 0.9466,
  delta pareado +0.0003 ± 0.0001 (5/5); público 0.94723. **La barra
  0.9484 NO se alcanzó** (ni en CV ni en público). La ganancia se
  diluye en el blend (solo 5 de 14 miembros llevan init GLM).
- Capacidad profunda (Exp. 3) INCOMPLETA: solo pliegue 1/5 de
  HGB-127 hojas (AUC 0.9412, 1757 s); sin JSON; nada corriendo.
  Completarla son ~7 h y el único pliegue va por debajo de la base
  (0.9449) — no se recomienda completarla así.
- Entorno: esta caja NO tiene sklearn/lightgbm (se pierden en los
  resets). La Kali SÍ: venv `.venv-fraude` (sklearn 1.6.1, lightgbm
  4.7.0) + repo + datos en
  `/home/deamon/wt_deamonx/CASMI-sandbox/pruebas/fraude-s6e11/`
  (train.csv md5 505e0ac5…). Los experimentos corren en la Kali.
- Git: HEAD 64dc05c (commit del blend); v3/v4/P3/P4 sin commitear
  (el git va detrás del trabajo real).
- Opciones de mejora (orden de Elán 2026-11-25 09:14: "Está bien, no
  significa que no pueda mejorar"; pendiente su GO): A — P5: más
  LightGBM con init GLM → v5 (+0.0001..+0.0003 CV, ETA ~1 h); B — P6:
  predicción GLM como FEATURE para HGB (+0.0002..+0.0005 en el blend,
  ETA ~2 h, la más prometedora sin medir); C — P7: pesos optimizados
  (~30 min, riesgo de sobreajuste); D — P8: sonda LightGBM 127 hojas
  (~15 min); E — Mantener (v4 + v3). Nada a Kaggle sin GO de Elán.

## Estado verificado (2026-11-25 ~10:43)

- **GO de Elán 09:21 (widget, "A y B"):** arrancar P5 (A) y P6 (B) en
  la Kali. **GO de Elán 09:29 (widget, "Sí, busquen el punch
  estructural"):** cazar el punch estructural además de A y B.
- **P5 (A) COMPLETO — GANA, ganancia chica.** Los 6 miembros nuevos
  LightGBM con init GLM (s3, s5, s11; crudas y combinadas) ganan cada
  uno vs sin init: crudas +0.0007 (5/5 pliegues cada uno), combinadas
  +0.0009..+0.0010 (5/5). Comprobación de cordura PASS. **Blend v5
  (v4 + 6 nuevos = 20 miembros, pesos iguales): CV pooled 0.9472 vs
  v4 0.9469 (v4 reproducido exacto: [0.9454, 0.9486, 0.9517, 0.9470,
  0.9421]); delta pareado +0.0002 ± 0.0001, positivo en 5/5 pliegues.**
  La ganancia (+0.0002) está dentro del rango esperado (+0.0001..
  +0.0003) pero **por debajo del corte de adopción +0.0005** y **la
  barra 0.9484 NO se alcanzó** (CV v5 0.9472). `submission_blend_v5.csv`
  construido y verificado (299,984 filas, id,is_fraud, sin NaN, media
  prob 0.0166, min 0.0000 max 0.9982, 7/7 checks OK; md5
  3f2a97426f89ebc982227d0cd2db1aab). Los 5 submissions anteriores
  intactos por md5. **El envío de la v5 a Kaggle es decisión de Elán
  (no cubierto por el GO de P5).** Archivos: `verificacion/
  p5_mas_lgbm_glm.py/.json/.log` (+ `p5_mas_lgbm_glm_parcial.json`).
  El script de P5 crasheó AL FINAL al loguear el veredicto
  (`TypeError: log() got an unexpected keyword argument 'flush'`,
  línea 629) — después de hacer todo el trabajo; el veredicto quedó
  en el JSON (`gana_v5: true`). El subagente P5 terminó sin entregar
  su reporte final; el resultado se verificó en disco (JSON + log +
  formato del CSV).
- **Punch estructural: NO HAY PUNCH (caza cerrada 09:50).** Forense:
  las continuas son discretas de baja cardinalidad
  (account_balance_usd 43,317 valores; 93.6% de filas en valores que
  se repiten ≥5 veces); primer dígito decimal con ratio 1.73–1.76x;
  0 casi-duplicados; árbol poco profundo 0.749–0.834 (no hay regla
  simple); ninguna feature separa las clases (mejor 0.7489). Medición
  v2 (TE anidado OOF, CV 5 pliegues semilla 42; base 0.9467): H1a–H1d
  +0.0001/−0.0001, H2a +0.0001, H3 (mejor) +0.0002 — ninguna pasa el
  corte +0.0005. Los +0.00113 (TE) y +0.00147 (dígitos) del foro NO se
  reproducen con TE OOF anidado correcto. La v1 tenía un bug (TE con
  auto-inclusión, artefacto −0.0749); la v2 es la válida. Discusión:
  escalera 0.9419→0.9511 (techo legítimo ~0.951–0.952); el top lo
  dominan cuentas nuevísimas (probablemente sondeo del LB, no
  estructural legítimo). Archivos: `verificacion/punch_forense.py/
  .json/.log`, `punch_medicion.py/.json/.log` (+ parcial). Nada a
  Kaggle/GitHub/Drive.
- **Incidentes:** reinicio de la Kali ~09:39–09:40 CST (uptime 4 min
  a las 09:43:34): mató la medición v1 de la caza y el proceso de P6
  (log de P6 termina 09:37:25); el subagente P6 sigue corriendo y
  relanzando su medición. P6 en marcha; su hito se reporta en cuanto
  salga.

## Estado verificado (2026-11-25 ~10:45)

- **P6 (B) COMPLETO — a nivel miembro GANA, a nivel blend NO GANA.**
  La feature GLM (logit de la prob OOF del GLM, recortado a [-10, 10],
  columna `glm_logit_oof`) ayuda a CADA miembro HGB por separado
  (OOF anidado, pareado, 5 pliegues, semilla 42): hgb_base_s42
  +0.0006 (4/5), hgb_base_s1 +0.0006 (4/5), hgb_base_s7 +0.0006
  (5/5), hgb_g5_s42 +0.0004 (4/5), hgb_g5_s1 +0.0005 (3/5),
  hgb_g5_s7 +0.0006 (4/5), hgb_g5_comb_s42 +0.0006 (4/5),
  hgb_g5_comb_s1 +0.0004 (3/5). Para LightGBM la feature es PEOR que
  el init: lgbm_l4_it500_s42 feature +0.0001 vs sin feature pero
  −0.0004 vs init GLM (0/5); lgbm_l4_it500_comb_s42 feature +0.0003
  vs sin feature pero −0.0005 vs init GLM (0/5). **El init sigue
  siendo la mejor forma de inyectar el GLM en LightGBM.**
  **Sin fuga verificado:** OOF anidado (ninguna fila aporta su propia
  etiqueta a su feature ni información del pliegue de evaluación);
  el diagnóstico simple-vs-anidado da −0.0001 (si simple >> anidado
  sería inflación por fuga, estilo P2 — no es el caso).
  **Corrección de conteo:** el blend v4 tiene **8 miembros HGB, no 6**
  (hgb_base_s42/s1/s7, hgb_g5_s42/s1/s7, hgb_g5_comb_s42/s1); la
  variante de blend reemplaza los 8.
  **Blend variante (8 HGB con feature GLM): CV pooled 0.9466 vs v4
  0.9469 (v4 reproducido exacto); delta pareado −0.0003 ± 0.0002,
  0/5 pliegues positivos.** La ganancia por miembro NO se apila en el
  blend — la feature GLM hace a los 8 HGB más parecidos entre sí
  (comparten la señal del GLM) y el blend pierde diversidad. **NO se
  construyó submission** (resultado negativo honesto; no existe
  `submission_blend_v5_p6.csv`). Los 5 submissions anteriores
  intactos por md5. Archivos: `verificacion/p6_glm_como_feature.py/
  .json/.log` (+ parcial, + `oof_p6_glm.npz`, `oof_p6_hgb_feat.npz`,
  `oof_p6_lgbm_feat.npz`). Tiempo de cómputo: 6.0 s (los OOF ya
  estaban calculados antes del reinicio de la Kali y se cargaron de
  checkpoint). El subagente P6 terminó sin entregar su reporte final;
  el resultado se verificó en disco (JSON + log).
- **Resultado completo P5 + P6:** P5 GANA chico (v5 CV 0.9472, delta
  +0.0002, 5/5; submission construido, md5 `3f2a9742…`). P6 GANA a
  nivel miembro (HGB +0.0004..+0.0006) pero NO GANA a nivel blend
  (variante −0.0003, 0/5; sin submission). **La combinación P5+P6 NO
  se recomienda:** añadir la feature GLM a los HGB del v5
  probablemente lo empeoraría (la variante P6 es negativa). **El
  mejor submission sigue siendo la v5 (P5).** El punch estructural
  no existe (caza cerrada, NO HAY PUNCH). **Nada a Kaggle sin GO de
  Elán:** el envío de la v5 es otra decisión, no cubierta por los GOs
  de P5/P6. Cierre 30 nov 2026, 10 envíos/día, hoy quedan 10/10.

## Estado verificado (2026-11-25 ~11:31)

- **Rank blend (cierre del loop del consultor externo) — NO GANA.**
  GO del Coordinador ~11:27; medido en la Kali en 6.8 s, sin entrenar
  nada nuevo (solo OOF guardados de los 20 miembros de la v5). Los 3
  mejores por AUC pooled (sin redondear): `lgbm_l4_it500_comb_s42`
  (0.947296), `lgbm_l4_it500_comb_s3` (0.947251),
  `lgbm_l4_it500_comb_s5` (0.947234); el 4to (`comb_s11`,
  0.947232) difiere del 3ro en 0.000002 (empate esencial; los 4 son
  el mismo modelo con distinta semilla). **v5 simple recalculada
  desde los OOF: reproducción EXACTA** ([0.9456, 0.9489, 0.9519,
  0.9471, 0.9424], pooled 0.9472). **Rank blend top3: [0.9457,
  0.9492, 0.9519, 0.9471, 0.9428], pooled 0.9473 — idéntico al
  promedio simple de los mismos 3 en los 5 pliegues.** Deltas
  pareados: (a) rank vs v5: [+0.0001, +0.0003, 0.0, 0.0, +0.0004],
  media **+0.0002** ± 0.0002, 3/5 positivos; (b) rank vs simple-3:
  [0.0 × 5], media **+0.0000** ± 0.0000, 0/5. **Criterio (delta
  ≥ +0.0005 con ≥4/5) NO se cumple: NO GANA.** El rankeo en sí da
  exactamente +0.0000 vs el promedio simple (los 3 miembros son el
  mismo modelo con distinta semilla, tan correlados que rankear no
  cambia nada). El +0.00004 del foro ni siquiera se alcanza como
  incremento atribuible al rankeo. **No se construyó submission.**
  Simple-3 (pooled 0.9473) queda una milésima sobre la v5 (0.9472),
  delta +0.0002, 3/5 — también por debajo del corte; ruido, no
  accionable. Archivos: `verificacion/rank_blend_v5.py/.json/.log`
  (md5 `780091d1…` / `6acfae59…` / `b08f7dc7…`, idénticos Kali y
  local). Los 6 submissions intactos por md5. **Con esto queda
  medido el ÚNICO elemento sin medir del script del consultor:**
  su pipeline completo, verificado con nuestro protocolo, no añade
  nada a la v5. El loop del consultor está CERRADO.

## Estado verificado (2026-11-25 ~19:07)

- **Consultor ensamble_v2 — ADOPTADO, enviado, récord, finales cambiadas.
  Frente CERRADO.** El consultor empujó la rama `ensamble-v2` en GitHub
  (`98b341f5…`): 6 candidatos con nuestro protocolo (StratifiedKFold 5,
  seed 42, TE anidado). Veredicto (Kali, 467 s): glm_lgb4 0.95066
  (Δ +0.00401 vs lgb4_raw, +5/5) GANA; blend_rank 0.95028; glm 0.95034;
  lgb4_te 0.95000; lgb4_raw 0.94665. GO Elán 18:15 (adoptar glm_lgb4);
  `--final` 18:16–18:25 (CV repetido 0.95066); "Subanlo" 18:35 → ref
  **58917706**; **público 0.95112 (18:43, COMPLETE)** — nuevo récord
  (+0.00389 sobre la v4). **Finales: consultor v2 + v4 (Elán, 18:46).**
  Detalle completo en `verificacion/VEREDICTO_CONSULTOR_2026-11-25.md`.
- **"Prueba todo" (Elán, 18:50) — medido, nada supera al glm_lgb4.**
  TE + GLM-init ya era el glm_lgb4 (no se duplicó ninguna corrida).
  Stacking + blend ponderado (`medir_stacking_blend.py`, 109 s, OOF):
  blend anidado 0.95072 (+0.00006 vs glm_lgb4, empate), stack_lr
  0.95037, stack_lr_rank 0.95028 — sin submission. Tuning del glm_lgb4
  (`tuning_glm_lgb4.py`, 762 s, 5 variantes): A_control 0.95066
  (reproduce exacto; harness válido), B_fino 0.95068 (+0.00001, ruido),
  C 0.95060, D 0.95058, E 0.95063 — ninguna supera al control.
- **GitHub (orden Elán 19:55):** commit con el consultor + medición +
  tuning + submission_consultor_v2.csv (md5 `9f79f63f…`) en la línea
  local; bundle nuevo para el push --force a
  local/fraude-2026-11-25 desde Katana. Main NO se toca.
