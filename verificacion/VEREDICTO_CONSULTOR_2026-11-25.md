# Veredicto — consultor ensamble_v2 y cierre del frente (2026-11-25)

Competición: Predicting Credit Card Fraud, Playground Series S6E11 (Kaggle id 125225).
Cierra: 30 nov 2026, 23:59 UTC. Límite: 10 envíos/día.

## Origen

El consultor empujó la rama `ensamble-v2` en GitHub
(`98b341f53179a39c62ef5443ffdf140f5d370783`, padre main `16d3dee`):
`ensamble_v2.py` (327 líneas), 6 candidatos medidos con nuestro protocolo
(StratifiedKFold 5, shuffle, seed 42, TE anidado sin fuga, CPU).

## Veredicto del CV (Kali, 467 s, tercer intento; dos intentos previos murieron por reinicios de la Kali)

| Candidato | Media | Δ vs lgb4_raw | +n/5 |
|---|---|---|---|
| glm_lgb4 (adoptado) | 0.95066 | +0.00401 | 5/5 |
| blend_rank | 0.95028 | +0.00363 | 5/5 |
| glm solo | 0.95034 | +0.00369 | 5/5 |
| lgb4_te | 0.95000 | +0.00335 | 5/5 |
| lgb4_raw (referencia) | 0.94665 | — | — |

Regla nuestra: delta pareado media >= +0.0005 y >= 4/5 pliegues positivos.
Los cuatro ganan. La combinación TE + GLM-boost junta hizo lo que los
componentes solos no habían hecho (P2/P3/P4/P5/P6 y la caza del punch ya
estaban medidos y cerrados sin punch).

`ensamble_v2_veredicto_2026-11-25.json` + OOF respaldados antes del `--final`.

## Adopción y envío (GO de Elán)

- 18:15 — adoptar glm_lgb4 para la v2.
- `--final --mejor glm_lgb4` (18:16–18:25, 551 s): el CV repetido confirmó
  0.95066, +5/5. Archivo `salida-fraude/submission_consultor_v2.csv`:
  299,984 filas, id/is_fraud, 0 NaN, 0 ids duplicados,
  is_fraud 0.0000000194–0.9998529263, media 0.0165352,
  md5 `9f79f63fa5bd532869834d421d3bea63`, 8,576,929 B.
- 18:35 — "Subanlo". Subido por el conector custom.kaggle (la Kali no tiene
  kaggle CLI): ref **58917706**, total_bytes 8576929 (exacto),
  fecha 2026-11-26T00:36:48.203Z, descripción "Consultor ensamble_v2:
  glm_lgb4 (CV 0.95066, delta +0.00401 vs lgb4_raw, +5/5)".
- **Score público (18:43): 0.95112, COMPLETE** — +0.00389 sobre la v4
  (0.94723) y +0.00046 sobre el CV. El público confirmó el CV.
- **Finales cambiadas por Elán (18:46, verificado en vivo): consultor v2
  (0.95112, ref 58917706) + v4 (0.94723, ref 58607873).** La v3
  (0.94691, ref 58331285) quedó fuera.

## "Prueba todo" — lo que se midió después y no superó al glm_lgb4

1. **TE + GLM-init** (pedido por Elán 18:48): YA EXISTÍA — glm_lgb4 ES
   TE + GLM-init (LightGBM sobre XT = crudas+TE, init_score = logit del
   GLM). No se lanzó ninguna corrida duplicada.
2. **Stacking + blend ponderado** (`medir_stacking_blend.py`, 109 s,
   sobre el OOF del veredicto, sin reentrenar bases):
   blend_pond_nested 0.95072 (Δ +0.00407, +5/5, pesos ~0.67/0.10/0.20/0.0);
   blend_pond_global 0.95073 (optimista); stack_lr 0.95037;
   stack_lr_rank 0.95028. Los cuatro ganan vs lgb4_raw, pero NINGUNO
   supera al glm_lgb4 (el mejor queda +0.00006, empate técnico).
   No se construyó submission.
3. **Tuning del glm_lgb4** (`tuning_glm_lgb4.py`, 762 s, 5 variantes,
   TE+GLM compartidos por pliegue): A_control 0.95066 (reproduce EXACTO
   el CV; lgb4_raw 0.94665 idéntico al veredicto — harness válido);
   B_fino 0.95068 (Δ vs A +0.00001, +3/5, ruido); C_mas_boost 0.95060;
   D_mas_hojas 0.95058; E_mas_reg 0.95063. NINGUNA supera al control.

## Cierre

Frente de fraude CERRADO en 0.95112 público. Todo lo medible sin
reentrenar bases está medido. El techo legítimo según el foro es
~0.951–0.952; el top 0.96–0.97 del leaderboard probablemente es LB probing.

## Archivos de esta noche (en este commit)

- `ensamble_v2.py` (consultor, raíz) + `fraude_tarjetas.py` (ya estaba,
  md5 idéntico en ambas cajas: `cfb1d657…`).
- `medir_stacking_blend.py`, `tuning_glm_lgb4.py` (raíz; importan
  `fraude_tarjetas` y `ensamble_v2` desde su propio directorio).
- `verificacion/ensamble_v2_veredicto_2026-11-25.json`,
  `verificacion/medir_stacking_blend.json`,
  `verificacion/tuning_glm_lgb4.json`.
- `salida-fraude/submission_consultor_v2.csv` (el envío 0.95112).
