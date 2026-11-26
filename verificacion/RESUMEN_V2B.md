# RESUMEN — ensamble-v2b + requirements (Kali, 2026-11-26)

## Alcance
- Ramas probadas: ensamble-v2b (42748a928537932638b36237a52e664cb477604d) y requirements (13cf135ebf6ef804236beb1ff8211393daf0bd97).
- Scratch: /home/deamon/fraude-test-v2b/ (repo/, requirements/, salida-ensamble-v2b/, verificar-entorno.log, nohup-v2b.out, REANUDACION.log, verificar_npz.py, verificar_npz2.py). El scratch de 4 ramas y el workdir de Kali NO se tocaron.
- Salud Kali: boot 4007b40a (el mismo del 04:19, sin reinicio durante la corrida; sigue sin reboot #9), :8787 200, :8179 200, launcher active/running. H20 ya estaba desplegado antes de lanzar (GO 07:01).
- Datos: train.csv md5 505e0ac51da1d3661dfdfbc2044db34b (verificado antes de correr; coincide con el del test de 4 ramas).
- Venv: /home/deamon/.venv-fraude, Python 3.13.15, sklearn 1.6.1, lightgbm 4.7.0.
- Nada a Kaggle. Sin push ni PRs. main y local/fraude-2026-11-25 intactos.

## Verificacion de archivos de la rama (blob SHAs, git ls-tree)
- ensamble_v2b.py blob bbe05956... OK; ensamble_v2.py blob 45459034..., md5 eb21cf3a7ae3133d02c09c3cafaebd94 — identico al usado en final-blend-v2, por eso NO hizo falta copiarlo desde el scratch de 4 ramas (ya estaba en la rama). fraude_tarjetas.py blob c3f30089...
- requirements.txt blob c884c1f3... OK; verificar_entorno.py blob 63d99c64... OK; requirements-informe.txt blob 14f06cda...
- ensamble_v2.json (control) copiado del test de 4 ramas a salida-ensamble-v2b/ ANTES de la corrida, md5 0e377b81b1b4490c466f6391038f6d49 OK.

## Verificador de entorno (requirements, 07:06:51, exit 1)
- Log: verificar-entorno.log, 13 l. / 866 B, md5 a324500ccb7e0ba519d9d6aca380f920.
- Python 3.13.15 vs referencia 3.12 -> AVISO.
- OK: lightgbm 4.7.0, scikit-learn 1.6.1.
- DIFIERE: numpy instalado 2.2.6, esperado 1.26.4; pandas instalado 2.3.3, esperado 2.1.4; scipy instalado 1.16.3, esperado 1.11.4.
- FALTA: ninguna. FALLA: ninguna.
- Imports: numpy, pandas, scipy, sklearn, lightgbm — los 5 OK.
- RESULTADO: entorno NO coincide (exit 1).
- Lectura: el venv actual derivo de los pines (numpy 2.x cuando requirements.txt prohibe explicitamente numpy 2.x). PERO la deriva es benigna para este pipeline segun la evidencia medida: el control A@400 reprodujo glm_lgb4 de v2 con dif maxima 5e-06 por pliegue (ver abajo) y el OOF pooled de A_400 (0.95064) reproduce exacto el OOF de v2 (0.95064). El verificador hace bien su trabajo (detecta la deriva); la deriva no movio los numeros.

## Corrida ensamble_v2b (07:07:10-07:16:49, 580 s, 2 hilos)
- Comando: python ensamble_v2b.py --datos /home/deamon/wt_deamonx/CASMI-sandbox/pruebas/fraude-s6e11/datos --salida /home/deamon/fraude-test-v2b/salida-ensamble-v2b --final (nohup, PID 288565).
- Sin reinicio de Kali durante la corrida: no hubo reanudacion (REANUDACION.log solo tiene el lanzamiento).
- train (699960, 26), fraude 0.01670. Columnas con fraccion (6): amount_usd, account_balance_usd, distance_from_home_km, velocity_score, hours_since_last_txn, merchant_risk_score.
- Por pliegue: TE+GLM 20-23 s (glm n_iter_=[7]); modelo A (137 columnas) 44-46 s; modelo B (149 columnas) 47-49 s.
- Sin errores en log ni nohup.

### Control A@400 vs glm_lgb4 de v2
- A@400 por pliegue: [0.94985, 0.95105, 0.95567, 0.95138, 0.94537], media 0.95066.
- glm_lgb4 de v2 (test de 4 ramas): CV media 0.95066 / OOF 0.95064.
- Dif maxima por pliegue: 0.000005 (OK, tolerancia <=2e-5). EL CONTROL REPRODUCE v2.
- OOF pooled recomputado del npz: A_400 0.95064 = OOF de v2 exacto.

### Curvas (AUC media por k, 5 pliegues)
- A: 200: 0.95063, 300: 0.95067, 400: 0.95066, 500: 0.95062, 600: 0.95061, 800: 0.95060, 1000: 0.95055, 1200: 0.95047. k* A = 300. Curva plana (rango 0.00020), cae despues de 300.
- B: 200: 0.95066, 300: 0.95069, 400: 0.95070, 500: 0.95069, 600: 0.95068, 800: 0.95069, 1000: 0.95070, 1200: 0.95063. k* B = 1000. Curva plana (rango 0.00007).
- B@400 por pliegue: [0.94985, 0.95096, 0.95574, 0.95146, 0.94549]. Deltas B-A por pliegue: [0.00000, -0.00009, +0.00007, +0.00008, +0.00012].
- Spearman A_400 vs B_400 (OOF): 0.99932 — los centavos apenas mueven el ranking.

### Candidatos (regla: delta > 0 en 5/5, media >= +0.0001, media > 2x std, vs A@400)
| Candidato | Media | Delta pareado (media +- std) | +x/5 | Cumple regla | Veredicto |
| A_kstar (A@300) | 0.95067 | +0.00001 +- 0.00003 | +2/5 | NO | NO se adopta |
| B_400 (B@400) | 0.95070 | +0.00003 +- 0.00007 | +3/5 | NO | NO se adopta |
| B_kstar (B@1000) | 0.95070 | +0.00004 +- 0.00012 | +3/5 | NO | NO se adopta |
- A_kstar pliegues: [0.94985, 0.95107, 0.95572, 0.95136, 0.94536], std 0.00332.
- B_400 pliegues: [0.94985, 0.95096, 0.95574, 0.95146, 0.94549], std 0.00328.
- B_kstar pliegues: [0.94980, 0.95092, 0.95571, 0.95161, 0.94547], std 0.00329.
- Mejor delta +0.00004 (3/5, media < 2x std 0.00024) = ruido, igual que en v3.

### --final
- Ningun candidato cumple la regla -> NO se escribe submission. repo/salida-fraude/ sigue con solo informe-fraude.md; no existe submission_v2b.csv en el scratch. El envio de 0.95112 queda como esta. Nada a Kaggle.

### OOF pooled recomputado del npz (verificacion independiente)
- A_400 0.95064, A_kstar 0.95064, B_400 0.95067, B_kstar 0.95066. Consistentes con las medias CV (diferencia 0.00002-0.00003, esperable: media de pliegues vs pooled).

## Archivos de salida (md5s)
- salida-ensamble-v2b/ensamble_v2b.json — 1,773 B, 85 l., md5 9d93424b6cd399eb1c59eb355ff6c349
- salida-ensamble-v2b/oof_ensamble_v2b.npz — 21,263,491 B, md5 ec418acd12eacfe83125e265581c1d43; keys A_400, A_kstar, B_400, B_kstar; cada una (699960,) float64, 0 NaN, rango ~6e-09-0.99994 (probabilidades)
- salida-ensamble-v2b/ensamble-v2b-corrida.log — 2,605 B, 36 l., md5 ae7b97a9176dbf7553d8c30ce86d0504
- salida-ensamble-v2b/ensamble_v2.json (control, copiado) — 2,541 B, md5 0e377b81b1b4490c466f6391038f6d49
- verificar-entorno.log — 866 B, 13 l., md5 a324500ccb7e0ba519d9d6aca380f920
- nohup-v2b.out — 2,605 B, md5 ae7b97a9176dbf7553d8c30ce86d0504 (mismo contenido que el log de corrida)
- REANUDACION.log — 216 B, md5 9a488b6af83b6e8de9bd3606e757cf0c

## Veredicto
- ensamble-v2b CORRE BIEN (sin errores, control reproduce v2 exacto) y NO MEJORA: ningun candidato cumple la regla de adopcion; el mejor delta es +0.00004 (3/5, ruido). Ni mas arboles (curva A plana y decreciente tras 300) ni los centavos/digitos (B, delta +0.00003-0.00004) mueven el glm_lgb4 0.95066/0.95064.
- requirements: el verificador CORRE BIEN y reporta correctamente que el venv actual NO coincide con los pines (numpy 2.2.6, pandas 2.3.3, scipy 1.16.3; Python 3.13.15 vs 3.12). La deriva no movio los numeros medidos (control 5e-06), pero requirements.txt documenta el entorno que produjo los numeros verificados (0.95112); el venv actual es otro entorno. Pendiente de decision: alinear el venv a los pines o actualizar los pines al venv. No se instalo nada; el verificador no modifica nada.
- Finales sin cambio: blend ponderado 0.95119 + consultor v2 0.95112.
