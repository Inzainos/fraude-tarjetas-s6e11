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
   de selección final (la v2 de Elán, 0.94393, ya está seleccionada
   como final), ningún notebook/dataset público.
4. **Nada a GitHub.** Este repo no tiene remote configurado; Elán decide
   si se crea. Trabajo solo en local.
5. **Respaldos a Google Drive**, carpeta "Respaldos Fractal Core /
   Fraude Tarjetas S6E11" (regla de Elán 2026-11-09 ~11:46). Nada de
   respaldos solo en local. GitHub = código versionado; Drive = respaldos.
6. **Código completo y con logs de ejecución.** Nada de fragmentos con
   "insertar resto aquí". Cada corrida deja su log en `verificacion/`.
7. **El submission existente no se toca.**
   `salida-fraude/submission.csv` (md5 c07bfd803b2704a13ec7f7d2afff7bd8)
   se verifica por md5 antes y después de cada paso.
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
