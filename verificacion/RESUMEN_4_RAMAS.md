# Prueba de las 4 ramas de fraude en Kali — resumen (2026-11-26, 03:28–05:22 CST)
Orden de Elan 03:20, confirmacion explicita 03:26 tras aviso del chequeo de ejecucion.
Todo en /home/deamon/fraude-test-4ramas/ (scratch). Sin push, sin PRs, sin envios a Kaggle,
sin tocar main ni el trabajo existente (/home/deamon/wt_deamonx/CASMI-sandbox/pruebas/fraude-s6e11/).
Clon local de la caja (~/workspace/fraude-tarjetas-s6e11) no tocado.
Entorno: venv /home/deamon/.venv-fraude (sklearn 1.6.1, lightgbm 4.7.0, pandas 2.3.3, numpy 2.2.6, scipy 1.16.3).
Datos: train.csv md5 505e0ac51da1d3661dfdfbc2044db34b; test.csv 6c95c68a6f98b1b1845220a82db884f7;
sample_submission.csv 55728bff392254016853fa4c5e1c1b36.
Referencia (envio 0.95112): referencia-submission_consultor_v2.csv, 299,985 lineas,
md5 9f79f63fa5bd532869834d421d3bea63 (extraido de local/fraude-2026-11-25, commit 360ebc86).
Reboots Kali durante la prueba: #6 ~03:38, #7 ~04:01-04:04, #8 boot 04:19.
Regla Elan 03:44:30 (reanudar, no reiniciar) aplicada en cada caso; ver REANUDACION.log por rama.

## 1. ensamble-v2 (98b341f5) — CORRE. NO mejora.
- CV 5 pliegues, semilla 42: 450 s (03:29:14-03:36:44). Refit final 03:46:49-03:49 (reanudado tras reboot #6).
- ensamble_v2.json md5 0e377b81b1b4490c466f6391038f6d49 (2,541 B).
- oof_ensamble_v2.npz md5 9171a790a54a4c48d82650415fd66a31 (29,141,242 B; 6 claves x 699,960, sin NaN).
- submission_v2.csv md5 33f06f9c7cb9ac385bfa1d457003da3f (7,878,126 B; 299,984 filas; media 0.50000 = RANGOS, no probabilidades; ids = sample; sin NaN).
- CV media / OOF agrupado: lgb31_raw 0.94540/0.94536; lgb4_raw 0.94665/0.94663 (+0.00125, 5/5);
  lgb4_te 0.95000/0.94996 (+0.00460, 5/5); glm 0.95034/0.95030 (+0.00493, 5/5);
  glm_lgb4 0.95066/0.95064 (+0.00526, 5/5); blend_rank 0.95028/0.95028 (+0.00488, 5/5).
  Deltas pareados vs lgb31_raw. Reproduce exacto los numeros del consultor.
- Delta pareado blend_rank - glm_lgb4 por pliegue: [-0.00073, +0.00012, -0.00066, -0.00050, -0.00015], media -0.00038 (gana 1/5).
- Spearman submission_v2 (blend_rank) vs referencia 0.95112: 0.99177.
- Veredicto (b): NO mejora. La rama ES el pipeline del consultor (glm_lgb4 = fuente del envio 0.95112);
  el envio por defecto (blend_rank) mide POR DEBAJO de glm_lgb4 en CV.

## 2. ensamble-v3 (ff8ca97b) — CORRE. NO mejora.
- CV 5 pliegues, semilla 42: 776.9 s (04:05:22-04:18:20; re-lanzamiento determinista tras reboot #7, reproduce pliegues 1-4 exactos).
  Refit final 04:27:00-04:30 (reanudado tras reboot #8).
- ensamble_v3.json md5 5395f960f5d85484e9add83b2c608e53 (2,505 B).
- oof_ensamble_v3.npz md5 8bcb9eaaaa4608a8ffe6b570f1dddeb1 (29,177,628 B; 6 claves, sin NaN).
- submission_v3.csv md5 de9c0893cf97c36f7f36dfa9588b9da2 (7,880,169 B; 299,984 filas; media 0.50000 = RANGOS; ids = ref; sin NaN).
- CV media / OOF: glm_lgb4_base 0.95066/0.95064 (control = v2 glm_lgb4 exacto);
  glm_lgb4_pares 0.95070/0.95066 (delta +0.00003 +- 0.00012, 3/5);
  glm_lgb4_pares_bag3 0.95078/0.95074 (delta +0.00011 +- 0.00013, 4/5);
  mlp 0.94605/0.94591 (delta -0.00462 +- 0.00063, 0/5);
  blend_90_10 0.95079/0.95079 (delta +0.00013 +- 0.00015, 4/5);
  blend_80_20 0.95068/0.95068 (delta +0.00001 +- 0.00018, 2/5).
  Regla de adopcion (5/5, media >= +0.0001, media > 2x std): NINGUN candidato la cumple.
- Spearman submission_v3 (blend_90_10) vs referencia 0.95112: 0.99818. Spearman v3 vs v2: 0.99115.
- Veredicto (b): NO mejora. Mejor delta medido +0.00013 (4/5 pliegues, media < 2x std): ruido, no adopcion.

## 3. final-blend-v2 (afc88cd3) — CORRE. NO mejora (empate).
- 112 s (04:34:15-04:36:08), exit 0. Nota: la rama NO trae ensamble_v2.py (sale de main);
  se copio desde el checkout ensamble-v2 (md5 eb21cf3a7ae3133d02c09c3cafaebd94) antes de correr, segun correccion.
- submission_blend_v2.csv md5 1faa2c9e4d05ff8f1976ea58cb3985db (8,571,729 B; 299,984 filas;
  media 0.01658, min 5.1e-07, max 0.99967 = probabilidades). Formato 7/7 OK vs sample_submission.
- final-blend-v2.log md5 be85e3a3268025625960c12e07a09f74.
- Spearman blend vs glm_lgb4 (misma corrida): 0.99903. Spearman blend vs referencia 0.95112: 0.99903.
  Spearman glm_lgb4 reentrenado vs referencia: 1.00000 (reproduce exacto el envio 0.95112).
- CV del blend (docstring, medido por medir_stacking_blend.py): 0.95072 anidado, delta +0.00006 vs glm_lgb4
  = empate tecnico (std entre pliegues ~0.0033; +0.00006 es 0.02 sigma).
- Veredicto (b): NO mejora medida. Empate; el propio docstring lo dice: "piso/cobertura, no AUC adicional".

## 4. stack-publico (61e7c051) — NO SE PUDO CORRER. Falta insumo obligatorio.
- Requiere OBLIGATORIAMENTE: gpt2_s2026_oof_5fold.npy (train, 699,960 filas) y
  gpt2_s2026_test_5fold.npy (test, 299,984 filas), fuente goodpjw2008/s6e11-target-first-gpt2-seed-2026 (CC0).
- Busqueda en Kali (find /home/deamon): NO existen. /tmp tampoco.
- HuggingFace (curl, 05:21): pagina del dataset 401; ambos .npy (resolve/main) 401.
  Gated: requiere login/token de HuggingFace. No se obtienen sin credenciales.
- Kaggle: no hay ~/.kaggle/kaggle.json en Kali (solo un access_token, contenido no verificado ni usado).
  Descargar el dataset de Kaggle requeriria credenciales de API nuevas.
- Rama DETENIDA segun la orden (sin credenciales nuevas). No se ejecuto stack_publico.py (sus --gpt-oof/--gpt-test son obligatorios).
  No produce nada; no hay veredicto de mejora posible. Falta exactamente: los 2 .npy del GPT-2.

## Conclusion general
(a) Corren bien: 3 de 4 (ensamble-v2, ensamble-v3, final-blend-v2). La 4a no se pudo correr por falta de insumos.
(b) Ninguna mejora MEDIDA sobre las finales (blend 0.95119 / consultor v2 0.95112):
ninguna rama supera a glm_lgb4 en CV/OOF pareado; los 3 envios producidos correlacionan
0.99177-0.99903 (Spearman) con el envio 0.95112. El score publico solo se conoce enviando a Kaggle; no se envio nada.
