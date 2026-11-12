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
