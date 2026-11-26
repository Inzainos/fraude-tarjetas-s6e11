# Resumen — re-corrida ensamble-v2b con pines actualizados al venv (2026-11-26)

Orden de Elán 07:41 (voz): re-correr la prueba con los pines actualizados
(commit aa1a89b9) y revisar qué resultados da.

## Entorno (lo único que cambió es el documento, no el entorno)
- Mismo venv /home/deamon/.venv-fraude (Python 3.13.15, numpy 2.2.6,
  pandas 2.3.3, scipy 1.16.3, scikit-learn 1.6.1, lightgbm 4.7.0), mismo
  código (rama ensamble-v2b 42748a9), mismos datos (train.csv md5
  505e0ac51da1d3661dfdfbc2044db34b). Los pines se actualizaron PARA IGUALAR
  al venv; no se reinstaló nada.
- Rama requirements en aa1a89b98dc2b3de4371b1c0adeb9adf8ebd718b ("Pines
  alineados al venv de la Kali (medido 2026-11-26; control v2b dif 5e-06)").
- Verificador re-corrido 07:43: exit 0, 5/5 pines OK, 5/5 imports OK,
  "entorno coincide con requirements.txt" (log
  requirements-nuevo/verificar-entorno-rerun.log, md5 1de1f29b...).
- Salud Kali: boot 4007b40a (04:19), SIN reboot durante la re-corrida;
  launcher PID 160040, :8787/api/bots 200, :8179 200.

## Re-corrida
- 07:43:10-07:52:40, 573 s (json tiempo_s 572.1), 2 hilos, PID 297391,
  nohup. Sin reanudación (REANUDACION.log: solo el lanzamiento).
- Salida nueva: salida-ensamble-v2b-rerun/ (la primera corrida en
  salida-ensamble-v2b/ quedó intacta). Control ensamble_v2.json copiado
  antes de correr, md5 0e377b81...

## Comparación lado a lado (primera 07:07-07:16, 580 s / re-corrida 07:43-07:52, 573 s)
- Control A@400 por pliegue: IDENTICO [0.94985, 0.95105, 0.95567, 0.95138,
  0.94537], media 0.95066; dif vs glm_lgb4 de v2: 5e-06 (OK); dif vs
  primera corrida: 0.000000 (tolerancia 1e-5).
- Curva A: IDENTICA (k* = 300; 200: 0.95063, 300: 0.95067, 400: 0.95066,
  500: 0.95062, 600: 0.95061, 800: 0.95060, 1000: 0.95055, 1200: 0.95047).
- Curva B: IDENTICA (k* = 1000; 200: 0.95066, 300: 0.95069, 400: 0.95070,
  500: 0.95069, 600: 0.95068, 800: 0.95069, 1000: 0.95070, 1200: 0.95063).
- Candidatos (regla: delta > 0 en 5/5, media >= +0.0001, media > 2x std):
  A_kstar +0.00001 +- 0.00003 (2/5) NO; B_400 +0.00003 +- 0.00007 (3/5) NO;
  B_kstar +0.00004 +- 0.00012 (3/5) NO. IDENTICOS a la primera corrida.
- OOF pooled recomputado del npz nuevo: A_400 0.95064, A_kstar 0.95064,
  B_400 0.95067, B_kstar 0.95066. IDENTICO a la primera corrida.

## Archivos (md5s verificados en disco)
- salida-ensamble-v2b-rerun/ensamble_v2b.json — 1,773 B, md5
  edc305f93e26667675967e1d07ec87bb. Difiere del primero (9d93424b...) SOLO
  en tiempo_s (572.1 vs 579.1); todo lo demás idéntico.
- salida-ensamble-v2b-rerun/oof_ensamble_v2b.npz — 21,263,491 B, md5
  ec418acd12eacfe83125e265581c1d43. IDENTICO byte por byte al primero
  (mismo md5). Keys A_400, A_kstar, B_400, B_kstar; cada una (699960,)
  float64, 0 NaN.
- salida-ensamble-v2b-rerun/ensamble-v2b-corrida.log — 2,605 B, 36 l.,
  md5 0292fde9a2ce6e28e39c695d71a504ab (= nohup-v2b-rerun.out).
- Primera corrida intacta: salida-ensamble-v2b/ensamble_v2b.json md5
  9d93424b..., oof md5 ec418acd... (re-verificados).

## --final y veredicto
- --final: ningún candidato cumple la regla; NO se escribió
  submission_v2b.csv (repo/salida-fraude/ sigue con solo
  informe-fraude.md). El envío de 0.95112 queda como está.
- VEREDICTO: la re-corrida REPRODUCE EXACTO la primera (dif 0 en todos
  los números; npz byte-idéntico). Ningún candidato pasa ahora tampoco.
  Finales sin cambio: blend ponderado 0.95119 + consultor v2 0.95112.
  Nada a Kaggle, sin push ni PRs.
- Seguridad: el chequeo de ejecución no volvió a saltar en los exec de
  esta re-corrida (fetch + lanzamiento + verificaciones, sin aviso en los
  resultados); todo se hizo solo en Kali, en scratch, bajo la orden de
  Elán.
