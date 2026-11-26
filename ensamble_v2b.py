#!/usr/bin/env python3
"""Ajustes internos de glm_lgb4 (ensamble_v2.py): curva de árboles + dígitos.

Mide, con el MISMO protocolo de v2 (StratifiedKFold(5, shuffle, 42), TE
anidado con semilla interna 17, mismos hiperparámetros y semilla), dos
cambios dentro del propio modelo, sin añadir familias nuevas:

  A  glm_lgb4 como en v2 pero entrenado con 1200 árboles (lr 0.05). Como
     LightGBM construye los árboles en secuencia con semilla fija, las
     primeras 400 iteraciones son idénticas a las del modelo de 400 árboles
     de v2: A@400 es el CONTROL y debe reproducir glm_lgb4 de v2. Se evalúa
     la curva de AUC en cortes de 200 a 1200 árboles con un solo ajuste
     (predict(num_iteration=k)).
  B  igual que A pero los árboles reciben además, como números crudos,
     los centavos (0..99) y el primer decimal (0..9) de cada columna
     numérica con parte fraccionaria. El GLM NO los recibe (un dígito como
     número no es lineal).

Candidatos pre-registrados (no se agregan más después de ver resultados):
  A_kstar   A con k* = argmax_k de la AUC media de A sobre los 5 pliegues
  B_400     B con 400 árboles
  B_kstar   B con k* propio (argmax de la AUC media de B)

Regla de adopción (pareada por pliegue contra A@400): delta > 0 en 5/5,
media >= +0.0001 y media > 2x la std del delta. Elegir k* con los mismos
pliegues que se puntúan introduce un sesgo pequeño (un escalar entre 8
cortes; un participante estimó ~+0.00002 por esto); por eso el umbral de
+0.0001 y no menos. Con 3 candidatos y esa regla el riesgo de falso positivo
es bajo pero no nulo: si la ganancia es del orden de 1e-4, tómala como
indicio, no como prueba.

Verificación de control: si existe <salida>/ensamble_v2.json, compara los
AUC por pliegue de A@400 con los de glm_lgb4 de v2 (deben coincidir a ~1e-5;
si no, el entorno o los hilos cambiaron).

Uso (en la Kali, desde la raíz del repo, con ensamble_v2.py al lado):
    python3 ensamble_v2b.py --datos datos --salida verificacion
    python3 ensamble_v2b.py --datos datos --salida verificacion --final
    python3 ensamble_v2b.py --datos DIR --salida DIR --max-filas 20000   # humo

Con --final entrena con todo train el MEJOR candidato que cumpla la regla y
escribe salida-fraude/submission_v2b.csv; si ninguno la cumple, no escribe
nada (el envío de 0.95112 queda como está). Nada se envía a Kaggle: eso lo
decide Elán. Log: <salida>/ensamble-v2b-corrida.log. Estimación de tiempo
(sin medir): 20 a 40 minutos de CV con 2 hilos.
"""
import argparse
import gc
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fraude_tarjetas as ft  # noqa: E402
import ensamble_v2 as v2  # noqa: E402

import lightgbm as lgb  # noqa: E402

SEED_CV = 42
N_SPLITS = 5
KS = [200, 300, 400, 500, 600, 800, 1000, 1200]
K_CONTROL = 400
log = logging.getLogger("ensamble_v2b")


def _auc(y, p):
    return float(roc_auc_score(y, p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def cols_con_fraccion(df):
    """Columnas numéricas de v2.COLS_TE_NUM con parte fraccionaria (no usa y)."""
    out = []
    for c in v2.COLS_TE_NUM:
        v = pd.to_numeric(df[c], errors="coerce").values.astype(float)
        v = v[~np.isnan(v)]
        if len(v) and np.any(v != np.floor(v)):
            out.append(c)
    return out


def digitos(df, cols):
    """Centavos (0..99) y primer decimal (0..9) como números crudos."""
    partes = []
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce").fillna(0.0).values.astype(float)
        frac = v - np.floor(v)
        partes.append(np.round(frac * 100) % 100)
        partes.append(np.floor(frac * 10 + 1e-9))
    if not partes:
        return np.zeros((len(df), 0), dtype=np.float32)
    return np.column_stack(partes).astype(np.float32)


def lgb_refuerzo(n_estimators):
    """Mismos hiperparámetros y semilla que el glm_lgb4 de v2."""
    return lgb.LGBMClassifier(
        n_estimators=n_estimators, learning_rate=0.05, num_leaves=4,
        min_child_samples=100, reg_lambda=5.0, colsample_bytree=0.7,
        subsample=0.8, subsample_freq=1, objective="binary",
        verbose=-1, n_jobs=NJOBS, random_state=42)


def preparar(d_tr, d_va, y_tr):
    """Crudas + TE anidado (semilla interna 17), como preparar_pliegue de v2."""
    X_tr, enc = ft.construir_features_crudas(d_tr, fit=True)
    X_va, _ = ft.construir_features_crudas(d_va, encoder=enc)
    X_tr = X_tr.values.astype(np.float32)
    X_va = X_va.values.astype(np.float32)
    v2.SEED_INNER = 17
    te_tr, te_va = v2.target_encoding_anidado(
        v2.construir_llaves(d_tr), y_tr, v2.construir_llaves(d_va))
    return np.hstack([X_tr, te_tr]), np.hstack([X_va, te_va])


def ajustar_pliegue(d_tr, d_va, y_tr, cols_f):
    """Devuelve {'A': {k: p_va}, 'B': {k: p_va}} para todos los cortes KS."""
    t = time.time()
    XT_tr, XT_va = preparar(d_tr, d_va, y_tr)
    sc = StandardScaler()
    XS_tr = sc.fit_transform(XT_tr).astype(np.float32)
    XS_va = sc.transform(XT_va).astype(np.float32)
    g = v2.glm()
    g.fit(XS_tr, y_tr)
    init_tr, init_va = g.decision_function(XS_tr), g.decision_function(XS_va)
    log.info("    TE+GLM %.0fs (glm n_iter_=%s)", time.time() - t,
             getattr(g, "n_iter_", "?"))
    del XS_tr, XS_va
    D_tr, D_va = digitos(d_tr, cols_f), digitos(d_va, cols_f)
    out = {}
    for var, (Xa, Xb) in (("A", (XT_tr, XT_va)),
                          ("B", (np.hstack([XT_tr, D_tr]), np.hstack([XT_va, D_va])))):
        t = time.time()
        m = lgb_refuerzo(max(KS))
        m.fit(Xa, y_tr, init_score=init_tr)
        out[var] = {k: _sigmoid(m.predict(Xb, raw_score=True, num_iteration=k) + init_va)
                    for k in KS}
        log.info("    modelo %s (%d columnas) %.0fs", var, Xa.shape[1], time.time() - t)
    return out


def cv(tr, cols_f):
    y = tr[ft.TARGET].values
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED_CV)
    oof = {v: {k: np.zeros(len(tr)) for k in KS} for v in ("A", "B")}
    aucs = {v: {k: [] for k in KS} for v in ("A", "B")}
    for i, (a, b) in enumerate(skf.split(tr, y)):
        log.info("pliegue %d/%d", i + 1, N_SPLITS)
        p = ajustar_pliegue(tr.iloc[a], tr.iloc[b], y[a].astype(float), cols_f)
        for var in ("A", "B"):
            for k in KS:
                oof[var][k][b] = p[var][k]
                aucs[var][k].append(_auc(y[b], p[var][k]))
        log.info("  A@400 %.5f | B@400 %.5f", aucs["A"][K_CONTROL][-1],
                 aucs["B"][K_CONTROL][-1])
        gc.collect()
    return y, oof, aucs


def resumen(y, oof, aucs, ruta_v2json):
    res = {"protocolo": "StratifiedKFold(5, shuffle, 42), TE anidado semilla 17",
           "control": "A@400", "curvas_auc_media": {}, "candidatos": {}}
    for var in ("A", "B"):
        res["curvas_auc_media"][var] = {str(k): round(float(np.mean(aucs[var][k])), 5)
                                        for k in KS}
        log.info("curva %s: %s", var, res["curvas_auc_media"][var])
    base = np.array(aucs["A"][K_CONTROL])
    log.info("control A@400 por pliegue: %s (media %.5f)",
             [round(float(x), 5) for x in base], base.mean())
    if os.path.exists(ruta_v2json):
        try:
            ref = json.load(open(ruta_v2json, encoding="utf-8"))
            ref_a = np.array(ref["modelos"]["glm_lgb4"]["aucs_por_pliegue"])
            dif = float(np.max(np.abs(ref_a - base)))
            res["control_vs_v2_max_dif"] = round(dif, 6)
            log.info("control vs glm_lgb4 de v2: dif máxima por pliegue %.6f %s", dif,
                     "(OK)" if dif <= 2e-5 else "(AVISO: no reproduce v2; revisa hilos/entorno)")
        except Exception as e:  # noqa: BLE001
            log.info("no se pudo comparar con %s: %s", ruta_v2json, e)
    k_a = max(KS, key=lambda k: np.mean(aucs["A"][k]))
    k_b = max(KS, key=lambda k: np.mean(aucs["B"][k]))
    res["k_estrella"] = {"A": int(k_a), "B": int(k_b)}
    log.info("k* de A = %d | k* de B = %d", k_a, k_b)
    candidatos = {"A_kstar": ("A", k_a), "B_400": ("B", K_CONTROL), "B_kstar": ("B", k_b)}
    oof_out = {"A_400": oof["A"][K_CONTROL]}
    for nombre, (var, k) in candidatos.items():
        a = np.array(aucs[var][k])
        d = a - base
        gana = bool((d > 0).all() and d.mean() >= 0.0001 and d.mean() > 2 * d.std())
        res["candidatos"][nombre] = {
            "variante": var, "k": int(k),
            "aucs_por_pliegue": [round(float(x), 5) for x in a],
            "auc_media": round(float(a.mean()), 5),
            "auc_std": round(float(a.std()), 5),
            "delta_pareado_vs_A400": round(float(d.mean()), 5),
            "delta_std": round(float(d.std()), 5),
            "positivo_en_pliegues": int((d > 0).sum()),
            "cumple_regla_adopcion": gana}
        oof_out[nombre] = oof[var][k]
        log.info("%-8s (%s@%d) media %.5f | Δ pareado %+.5f ± %.5f | +%d/5 | adopta=%s",
                 nombre, var, k, a.mean(), d.mean(), d.std(), int((d > 0).sum()), gana)
    return res, candidatos, oof_out


def main():
    global NJOBS
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default=os.path.join(HERE, "datos"))
    ap.add_argument("--salida", default=os.path.join(HERE, "verificacion"))
    ap.add_argument("--max-filas", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=2,
                    help="hilos de LightGBM; 2 reproduce v2 (otros valores pueden "
                         "cambiar el control en ~1e-5)")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    NJOBS = a.n_jobs
    os.makedirs(a.salida, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(os.path.join(a.salida, "ensamble-v2b-corrida.log"),
                                      mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])
    t0 = time.time()
    tr = pd.read_csv(os.path.join(a.datos, "train.csv"))
    if a.max_filas:
        tr = tr.sample(a.max_filas, random_state=0).reset_index(drop=True)
    cols_f = cols_con_fraccion(tr)
    log.info("train %s fraude %.5f | columnas con fracción (%d): %s", tr.shape,
             tr[ft.TARGET].mean(), len(cols_f), cols_f)
    y, oof, aucs = cv(tr, cols_f)
    res, candidatos, oof_out = resumen(
        y, oof, aucs, os.path.join(a.salida, "ensamble_v2.json"))
    res["tiempo_s"] = round(time.time() - t0, 1)
    res["filas"] = int(len(tr))
    with open(os.path.join(a.salida, "ensamble_v2b.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(a.salida, "oof_ensamble_v2b.npz"), **oof_out)
    if a.final:
        ganadores = [n for n, r in res["candidatos"].items() if r["cumple_regla_adopcion"]]
        if not ganadores:
            log.info("--final: ningún candidato cumple la regla; no se escribe submission. "
                     "El envío de 0.95112 queda como está.")
        else:
            mejor = max(ganadores, key=lambda n: res["candidatos"][n]["delta_pareado_vs_A400"])
            var, k = candidatos[mejor]
            log.info("--final: se entrena %s (%s@%d) con todo train", mejor, var, k)
            te = pd.read_csv(os.path.join(a.datos, "test.csv"))
            p = ajustar_pliegue(tr, te.assign(**{ft.TARGET: 0}),
                                tr[ft.TARGET].values.astype(float), cols_f)[var][k]
            ruta_dir = os.path.join(HERE, "salida-fraude")
            os.makedirs(ruta_dir, exist_ok=True)
            ruta = os.path.join(ruta_dir, "submission_v2b.csv")
            pd.DataFrame({"id": te["id"].values, "is_fraud": p}).to_csv(ruta, index=False)
            log.info("escrito %s (%d filas, media %.5f)", ruta, len(p), p.mean())
    log.info("listo en %.0fs", time.time() - t0)
    return 0


NJOBS = 2

if __name__ == "__main__":
    sys.exit(main())
