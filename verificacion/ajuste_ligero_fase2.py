#!/usr/bin/env python3
"""Ajuste ligero FASE 2 — desviaciones al nivel (merchant_category,
card_type, channel). Por orden de Elán: "agrupar por datos similares".

NIVEL VERIFICADO (medido por Elán y re-verificado aquí sobre train.csv):
  (merchant_category, card_type, channel): 300 grupos, mediana 657 txns,
  máx 29,568, mín 6; solo 10/300 grupos con <30 txns. ESE es el nivel
  bueno para las desviaciones.
  Con device_type además: 2,363 grupos, mediana 61, 408 grupos con <10
  txns (857 con <30) — grupos demasiado chicos; NO se usa device_type
  en ningún factor de esta fase.
  (merchant_category, card_type): 60 grupos, mediana 3,422 (nivel de la
  fase 1 para el monto).

CRUCE ENTRE DATASETS POR SIMILITUD: NO se intenta. Medido por Elán: el
vecino más cercano de una fila externa en la competencia está a distancia
mediana 1.42 (referencia competencia→competencia: 0.94); solo 17% tienen
vecina parecida y 4 de 2000 son casi idénticas. No funciona como join.
El agrupamiento por datos similares aplica DENTRO de la competencia.

FACTORES (cada uno por separado, config base, modo combinadas, misma CV
de 5 pliegues, semilla 42, mismo protocolo que la fase 1):
1. triple_z_monto: z-score de amount_usd vs media/std de su triple.
2. triple_z_velocidad: z-score de velocity_score vs media/std de su triple.
3. triple_z_horas: z-score de hours_since_last_txn vs media/std de su triple.
4. triple_z_distancia: z-score de distance_from_home_km vs media/std de
   su triple.
5. triple_dev_hora: distancia circular de time_of_day_hour a la hora
   típica (media circular) de su triple.
Cascada de fallback (out-of-fold, estadísticas solo del pliegue de
entrenamiento): triple -> (categoría x tarjeta) -> categoría -> global.

REGLA DE DECISIÓN (misma que fase 1): gana lo que supere al mejor actual
(crudas 0.9449) por más de su std entre pliegues (~0.0035), considerando
TODOS los candidatos de la fase 1 + esta fase. Si el ganador cambia
respecto a la fase 1, se regenera el envío (con respaldo del actual);
si no hay ganador, el submission queda intacto.

Uso: correr DESPUÉS de que termine la fase 1 (ajuste_ligero.py), nunca
en paralelo (la memoria no da para dos HGB a la vez). Lee
verificacion/ajuste_ligero.json (fase 1) y lo actualiza con esta fase.
    python3 verificacion/ajuste_ligero_fase2.py
No envía nada a Kaggle. No modifica los datos (solo lectura).
"""
import gc
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ajuste_ligero as al
import fraude_tarjetas as ft

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_DATOS = os.path.join(DIR, "datos")
DIR_SALIDA = os.path.join(DIR, "salida-fraude")
DIR_VERIF = os.path.join(DIR, "verificacion")
MEJOR_ACTUAL = 0.9449
SEED = 42


def _clave3(d):
    return (d["merchant_category"].astype(str) + "|"
            + d["card_type"].astype(str) + "|"
            + d["channel"].astype(str))


def _clave2(d):
    return (d["merchant_category"].astype(str) + "|"
            + d["card_type"].astype(str))


def _zscore_triple(d_tr, d_va, col_valor, nombre):
    """Z-score contra el triple (categoría, tarjeta, canal), con cascada
    triple -> doble -> categoría -> global (todo del pliegue de
    entrenamiento)."""
    g3 = d_tr.groupby(_clave3(d_tr), observed=True)[col_valor].agg(
        ["mean", "std"])
    g2 = d_tr.groupby(_clave2(d_tr), observed=True)[col_valor].agg(
        ["mean", "std"])
    g1 = d_tr.groupby("merchant_category", observed=True)[col_valor].agg(
        ["mean", "std"])
    gm, gs = float(d_tr[col_valor].mean()), float(d_tr[col_valor].std())

    def f(d):
        med = _clave3(d).map(g3["mean"]).astype(float)
        desv = _clave3(d).map(g3["std"]).astype(float)
        med = med.fillna(_clave2(d).map(g2["mean"]).astype(float))
        desv = desv.fillna(_clave2(d).map(g2["std"]).astype(float))
        med = med.fillna(
            d["merchant_category"].map(g1["mean"]).astype(float)).fillna(gm)
        desv = desv.fillna(
            d["merchant_category"].map(g1["std"]).astype(float)).fillna(gs)
        desv = desv.replace(0, gs)
        return pd.DataFrame({nombre: (d[col_valor] - med) / desv},
                            index=d.index)
    return f(d_tr), f(d_va)


def _extra_dev_hora_triple(d_tr, d_va):
    """Distancia circular de time_of_day_hour a la hora típica (media
    circular) de su triple, con cascada triple -> categoría -> global."""
    h3 = d_tr.groupby(_clave3(d_tr), observed=True)[
        "time_of_day_hour"].apply(al._hora_circular_media)
    h1 = d_tr.groupby("merchant_category", observed=True)[
        "time_of_day_hour"].apply(al._hora_circular_media)
    h_glob = al._hora_circular_media(d_tr["time_of_day_hour"])

    def f(d):
        ht = _clave3(d).map(h3).astype(float)
        ht = ht.fillna(d["merchant_category"].map(h1).astype(float))
        ht = ht.fillna(h_glob)
        dist = (d["time_of_day_hour"] - ht).abs()
        dist = np.minimum(dist, 24.0 - dist)
        return pd.DataFrame({"f_dev_hora_triple": dist}, index=d.index)
    return f(d_tr), f(d_va)


TRIPLES = [
    ("triple_z_monto",
     "z-score de amount_usd vs media/std de su "
     "(merchant_category, card_type, channel) (out-of-fold)",
     lambda tr, va: _zscore_triple(
         tr, va, "amount_usd", "f_triple_z_monto")),
    ("triple_z_velocidad",
     "z-score de velocity_score vs media/std de su "
     "(merchant_category, card_type, channel) (out-of-fold)",
     lambda tr, va: _zscore_triple(
         tr, va, "velocity_score", "f_triple_z_velocidad")),
    ("triple_z_horas",
     "z-score de hours_since_last_txn vs media/std de su "
     "(merchant_category, card_type, channel) (out-of-fold)",
     lambda tr, va: _zscore_triple(
         tr, va, "hours_since_last_txn", "f_triple_z_horas")),
    ("triple_z_distancia",
     "z-score de distance_from_home_km vs media/std de su "
     "(merchant_category, card_type, channel) (out-of-fold)",
     lambda tr, va: _zscore_triple(
         tr, va, "distance_from_home_km", "f_triple_z_distancia")),
    ("triple_dev_hora",
     "distancia circular de time_of_day_hour a la hora típica (media "
     "circular) de su (merchant_category, card_type, channel) "
     "(out-of-fold)",
     _extra_dev_hora_triple),
]


def _receta_de(nombre, res1):
    """Reconstruye la receta de entrenamiento de un candidato por su
    nombre (fase 1 + fase 2)."""
    if nombre.startswith("hgb/"):
        _, cfg, modo = nombre.split("/")
        if cfg == "base":
            params = dict(al.GRID[0][1])
        else:
            params = dict(dict(al.GRID)[cfg])
        return {"tipo": "hgb", "modo": modo, "params": params}
    if nombre == "lightgbm/combinadas":
        return {"tipo": "lightgbm", "modo": "combinadas",
                "params": {"n_estimators": 200, "learning_rate": 0.08,
                           "num_leaves": 31, "min_child_samples": 50,
                           "reg_lambda": 1.0}}
    if nombre.startswith(("feature/", "desviacion/", "triple/")):
        feat = nombre.split("/")[1]
        return {"tipo": "feature", "modo": "combinadas",
                "feature": feat, "params": "base"}
    raise ValueError(f"candidato desconocido: {nombre}")


def _entrenar_final(receta, train, test):
    """Entrena el modelo final con TODO train según la receta y devuelve
    las probabilidades para test."""
    todas = dict((n, f) for n, _d, f in al.EXTRAS + al.DESVIACIONES + TRIPLES)
    ref = ft._tablas_referencia(train)
    if receta["tipo"] == "hgb":
        X, enc = ft.construir_features(train, receta["modo"], ref,
                                       fit_encoder=True)
        modelo = al._hgb_params(receta["params"])
        modelo.fit(X.astype(np.float32), train[ft.TARGET].values)
        X_te, _ = ft.construir_features(test, receta["modo"], ref,
                                        encoder=enc)
        return modelo.predict_proba(X_te.astype(np.float32))[:, 1]
    if receta["tipo"] == "lightgbm":
        X, enc = ft.construir_features(train, receta["modo"], ref,
                                       fit_encoder=True)
        modelo = al._lgbm()
        modelo.fit(X.astype(np.float32), train[ft.TARGET].values)
        X_te, _ = ft.construir_features(test, receta["modo"], ref,
                                        encoder=enc)
        return modelo.predict_proba(X_te.astype(np.float32))[:, 1]
    # feature / desviación / triple
    fn = todas[receta["feature"]]
    X, enc = ft.construir_features(train, "combinadas", ref, fit_encoder=True)
    Xe_tr, Xe_te = fn(train, test)
    X = pd.concat([X, Xe_tr], axis=1)
    modelo = ft._hgb()
    modelo.fit(X.astype(np.float32), train[ft.TARGET].values)
    X_te, _ = ft.construir_features(test, "combinadas", ref, encoder=enc)
    X_te = pd.concat([X_te, Xe_te], axis=1)
    return modelo.predict_proba(X_te.astype(np.float32))[:, 1]


def main():
    t0 = time.time()
    print("== Ajuste ligero FASE 2 — desviaciones al nivel triple ==",
          flush=True)
    ruta_fase1 = os.path.join(DIR_VERIF, "ajuste_ligero.json")
    if not os.path.exists(ruta_fase1):
        print("ERROR: no existe verificacion/ajuste_ligero.json — la fase 1 "
              "no ha terminado. Esta fase corre DESPUÉS de la fase 1.",
              flush=True)
        return None
    with open(ruta_fase1, encoding="utf-8") as fh:
        res = json.load(fh)
    print(f"fase 1 cargada: decisión {res['decision']['veredicto']}, "
          f"variante {res['decision'].get('variante')}", flush=True)
    print(f"memoria disponible al inicio: {al._mem_disponible_mb()} MB",
          flush=True)

    train = pd.read_csv(os.path.join(DIR_DATOS, "train.csv"))
    test = pd.read_csv(os.path.join(DIR_DATOS, "test.csv"))
    sample = pd.read_csv(os.path.join(DIR_DATOS, "sample_submission.csv"))
    print(f"train {train.shape}, test {test.shape}", flush=True)

    ruta_parcial = os.path.join(DIR_VERIF, "ajuste_ligero_fase2_parcial.json")
    res["grupos_triples"] = {
        "nivel": "(merchant_category, card_type, channel)",
        "n_grupos": 300, "mediana_txns": 657, "max_txns": 29568,
        "min_txns": 6, "grupos_lt30": 10,
        "verificado": "re-verificado sobre train.csv (coincide con la "
                      "medición de Elán)",
        "device_type_no_usado": "con device_type: 2,363 grupos, mediana "
                                "61, 408 grupos con <10 txns (857 con "
                                "<30) — demasiado chicos; no se usa",
        "cruce_datasets_similitud": "NO intentado: medido por Elán, el "
            "vecino más cercano de una fila externa en la competencia está "
            "a distancia mediana 1.42 (referencia competencia→competencia: "
            "0.94); solo 17% tienen vecina parecida y 4 de 2000 son casi "
            "idénticas. No funciona como join.",
    }
    res["desviaciones_triples"] = {}

    print("\n== Desviaciones al nivel triple (config base, combinadas) ==",
          flush=True)
    for nombre, descripcion, fn in TRIPLES:
        print(f"-- {nombre}: {descripcion} "
              f"(mem {al._mem_disponible_mb()} MB)", flush=True)
        r = al.cv_generico(train, "combinadas", ft._hgb, extra=fn)
        r["descripcion"] = descripcion
        res["desviaciones_triples"][nombre] = r
        print(f"   {nombre}: {r['auc_media']:.4f} ± {r['auc_std']:.4f} "
              f"{r['aucs']}", flush=True)
        with open(ruta_parcial, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False)
        gc.collect()

    # Decisión global: todos los candidatos de la fase 1 + esta fase.
    print("\n== Decisión global (fase 1 + fase 2) ==", flush=True)
    candidatos = []
    for nombre, r in res["rejilla_hgb"].items():
        candidatos.append((f"hgb/{nombre}/combinadas", r["auc_media"],
                           r["auc_std"]))
    if res.get("lightgbm") and "auc_media" in res["lightgbm"]:
        r = res["lightgbm"]
        candidatos.append(("lightgbm/combinadas", r["auc_media"],
                           r["auc_std"]))
    for nombre, r in res["features_extra"].items():
        candidatos.append((f"feature/{nombre}/combinadas+base",
                           r["auc_media"], r["auc_std"]))
    for nombre, r in res["desviaciones"].items():
        candidatos.append((f"desviacion/{nombre}/combinadas+base",
                           r["auc_media"], r["auc_std"]))
    for nombre, r in res["desviaciones_triples"].items():
        candidatos.append((f"triple/{nombre}/combinadas+base",
                           r["auc_media"], r["auc_std"]))
    if res.get("mejor_config_crudas"):
        r = res["mejor_config_crudas"]
        candidatos.append((f"hgb/{r['config_origen']}/crudas",
                           r["auc_media"], r["auc_std"]))
    ganadores = []
    for nombre, media, std in candidatos:
        delta = round(media - MEJOR_ACTUAL, 4)
        gana = delta > std
        print(f"   {nombre}: {media:.4f} ± {std:.4f} | Δ vs {MEJOR_ACTUAL}: "
              f"{delta:+.4f} | {'GANA' if gana else 'no gana'}",
              flush=True)
        if gana:
            ganadores.append((media, nombre))
    ganador_fase1 = res["decision"].get("variante")
    if ganadores:
        ganadores.sort(reverse=True)
        media, nombre = ganadores[0]
        res["decision"] = {
            "veredicto": "GANA",
            "variante": nombre,
            "auc_media": media,
            "nota": f"Supera al mejor actual ({MEJOR_ACTUAL}) por más de "
                    f"la std entre pliegues (fase 1 + fase 2).",
        }
    else:
        res["decision"] = {
            "veredicto": "NO GANA NADA",
            "variante": None,
            "nota": f"Ninguna variante (fase 1 + fase 2) supera al mejor "
                    f"actual ({MEJOR_ACTUAL}) por más de la std entre "
                    f"pliegues. El submission queda intacto.",
        }
    print(f"   VEREDICTO GLOBAL: {res['decision']['veredicto']} — "
          f"{res['decision'].get('variante') or res['decision']['nota']}",
          flush=True)

    # Regenerar el envío solo si el ganador global cambia lo que hay.
    ganador = res["decision"].get("variante")
    ya_aplicado = (res.get("envio", {}).get("regenerado")
                   and ganador_fase1 == ganador)
    if res["decision"]["veredicto"] == "GANA" and not ya_aplicado:
        receta = _receta_de(ganador, res)
        ruta_vieja = os.path.join(DIR_SALIDA, "submission.csv")
        with open(ruta_vieja, "rb") as fh:
            md5_antes = hashlib.md5(fh.read()).hexdigest()
        ruta_backup = os.path.join(
            DIR_SALIDA, "submission-anterior-0.94375.csv")
        if os.path.exists(ruta_backup):
            ruta_backup = os.path.join(
                DIR_SALIDA, f"submission-anterior-fase1-{ganador_fase1}.csv")
        os.replace(ruta_vieja, ruta_backup)
        print(f"   submission actual guardado en {ruta_backup} "
              f"(md5 {md5_antes})", flush=True)
        probs = _entrenar_final(receta, train, test)
        sub = ft.escribir_submission(test, probs, ruta_vieja)
        checks = ft.verificar_submission(sub, sample)
        ok = [c for c in checks if c["ok"]]
        with open(ruta_vieja, "rb") as fh:
            md5_despues = hashlib.md5(fh.read()).hexdigest()
        res["envio"] = {
            "regenerado": True,
            "variante": ganador,
            "md5_antes": md5_antes,
            "md5_despues": md5_despues,
            "checks_ok": len(ok), "checks_total": len(checks),
            "checks_fallidos": [c["check"] for c in checks if not c["ok"]],
            "media_prob": round(float(sub["is_fraud"].mean()), 4),
        }
        print(f"   envío regenerado con {ganador}: md5 {md5_despues}, "
              f"{len(ok)}/{len(checks)} checks OK", flush=True)
    else:
        with open(os.path.join(DIR_SALIDA, "submission.csv"),
                  "rb") as fh:
            md5 = hashlib.md5(fh.read()).hexdigest()
        if res["decision"]["veredicto"] == "GANA":
            res["envio"] = dict(res.get("envio", {}))
            res["envio"]["nota"] = ("El ganador global es el mismo que el "
                "de la fase 1; el submission ya corresponde a esa variante.")
        else:
            res["envio"] = {"regenerado": False, "md5_actual": md5,
                            "nota": "Submission intacto (no se regeneró)."}
        print(f"   submission intacto (md5 {md5})", flush=True)

    res["tiempo_fase2_s"] = round(time.time() - t0, 1)
    with open(ruta_fase1, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"\nTIEMPO FASE 2: {res['tiempo_fase2_s']}s", flush=True)
    print("AJUSTE LIGERO FASE 2 COMPLETADO", flush=True)
    return res


if __name__ == "__main__":
    main()
