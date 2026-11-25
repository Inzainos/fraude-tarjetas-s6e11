#!/usr/bin/env python3
"""Pipeline v2 — fraude con tarjeta (Kaggle S6E11, ROC AUC).

Ideas tomadas de la discusión pública de la competencia (hilo 755149,
"What each idea is worth"), todas medidas aquí con el MISMO protocolo del
repo: StratifiedKFold(5, shuffle=True, random_state=42), tablas/encodings
calculados SOLO con el pliegue de entrenamiento (sin fuga):

  lgb31_raw     LightGBM 31 hojas sobre crudas (baseline comparable al repo)
  lgb4_raw      LightGBM 4 hojas, más árboles (los datos son casi logísticos)
  lgb4_te       lgb4 + target encoding por valor ANIDADO (cross-fit interno
                5 pliegues sobre el entrenamiento) con llaves de dígitos
  glm           regresión logística (newton-cholesky) sobre crudas + logits TE
  glm_lgb4      árboles de 4 hojas que parten del logit del GLM (init_score)
  blend_rank    media de rangos de lgb4_te, glm_lgb4 y lgb4_raw

Regla de decisión (corregida): un candidato se adopta si su delta PAREADO
por pliegue contra lgb31_raw es positivo en los 5 pliegues y su media
supera 2x la desviación del delta. No se exige superar la std entre
pliegues del AUC absoluto (esa varianza la comparten todos los modelos).

Además imprime n_iter_ de HistGradientBoosting en el pliegue 0 para
verificar si el early stopping automático de sklearn (activo con n > 10,000)
estaba recortando las 200 iteraciones del baseline anterior.

Uso:
    python3 ensamble_v2.py --datos datos --salida verificacion
    python3 ensamble_v2.py --datos datos --salida verificacion --final
    python3 ensamble_v2.py --datos DIR --salida DIR --max-filas 30000   # prueba de humo

Sin --final NO escribe ningún submission. Con --final entrena con todo
train y escribe salida-fraude/submission_v2.csv (nunca sobrescribe
submission.csv). Ningún envío a Kaggle: eso lo decide Elán.
Cada corrida escribe su log en <salida>/ensamble-v2-corrida.log.
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
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fraude_tarjetas as ft  # noqa: E402

import lightgbm as lgb  # noqa: E402

SEED_CV = 42
N_SPLITS = 5
SEED_INNER = 17
SUAVIZADO = 20.0           # pseudo-conteos del suavizado bayesiano del TE
COLS_TE_NUM = ["amount_usd", "account_balance_usd", "distance_from_home_km",
               "velocity_score", "hours_since_last_txn", "merchant_risk_score",
               "card_age_months", "customer_age", "txn_count_last_24h",
               "time_of_day_hour", "day_of_week", "cvv_retry_count",
               "prior_disputes"]
log = logging.getLogger("ensamble_v2")


# ---------------------------------------------------------------------------
# Llaves de dígitos y target encoding anidado
# ---------------------------------------------------------------------------
def construir_llaves(df):
    """Llaves enteras (codificadas como texto) por columna: valor, parte
    entera, entera//10, décimas, centavos, primeros dígitos. No usa la
    etiqueta. Devuelve DataFrame de strings, una columna por llave."""
    out = {}
    for c in COLS_TE_NUM:
        v = pd.to_numeric(df[c], errors="coerce").fillna(-1.0).values.astype(float)
        ent = np.floor(v)
        frac = v - ent
        out[f"{c}__val"] = np.round(v, 4).astype(str)
        out[f"{c}__ent"] = ent.astype(np.int64).astype(str)
        out[f"{c}__ent10"] = (ent // 10).astype(np.int64).astype(str)
        out[f"{c}__dec"] = np.floor(frac * 10).astype(np.int64).astype(str)
        out[f"{c}__cts"] = (np.round(frac * 100).astype(np.int64) % 100).astype(str)
        out[f"{c}__lead2"] = np.array([s[:2] for s in ent.astype(np.int64).astype(str)])
    for c in ft.COLS_CAT:
        out[f"{c}__cat"] = df[c].astype(str).values
    return pd.DataFrame(out, index=df.index)


def _codificar(llaves_a, llaves_b):
    """Códigos enteros comunes (factorize sobre la concatenación; no usa y)."""
    cods_a, cods_b = {}, {}
    for c in llaves_a.columns:
        todo = np.concatenate([llaves_a[c].values, llaves_b[c].values])
        cod, uniq = pd.factorize(todo)
        cods_a[c] = cod[:len(llaves_a)]
        cods_b[c] = cod[len(llaves_a):]
    return cods_a, cods_b


def _logit_te(cod_fit, y_fit, cod_apply, prior, n_cod):
    s = np.bincount(cod_fit, weights=y_fit, minlength=n_cod)
    n = np.bincount(cod_fit, minlength=n_cod).astype(float)
    p = (s + SUAVIZADO * prior) / (n + SUAVIZADO)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))[cod_apply]


def target_encoding_anidado(llaves_tr, y_tr, llaves_otro):
    """TE por valor, logit suavizado. Para las filas de entrenamiento se
    usa cross-fitting interno (5 pliegues): cada fila se codifica con
    estadísticas de las demás. Para `llaves_otro` (validación/test) se
    usa TODO el entrenamiento. Devuelve (TE_tr, TE_otro) float32."""
    cod_tr, cod_ot = _codificar(llaves_tr, llaves_otro)
    prior = float(y_tr.mean())
    cols = list(llaves_tr.columns)
    te_tr = np.zeros((len(llaves_tr), len(cols)), dtype=np.float32)
    te_ot = np.zeros((len(llaves_otro), len(cols)), dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED_INNER)
    idx_internos = list(skf.split(np.zeros(len(y_tr)), y_tr))
    for j, c in enumerate(cols):
        a, b = cod_tr[c], cod_ot[c]
        n_cod = int(max(a.max(), b.max())) + 1
        for fit_idx, ap_idx in idx_internos:
            te_tr[ap_idx, j] = _logit_te(a[fit_idx], y_tr[fit_idx], a[ap_idx],
                                         prior, n_cod)
        te_ot[:, j] = _logit_te(a, y_tr, b, prior, n_cod)
    return te_tr, te_ot


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def lgb31(seed=42):
    return lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.08, num_leaves=31,
        min_child_samples=50, reg_lambda=1.0, objective="binary",
        verbose=-1, n_jobs=2, random_state=seed)


def lgb4(seed=42):
    return lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.05, num_leaves=4,
        min_child_samples=100, reg_lambda=5.0, colsample_bytree=0.7,
        subsample=0.8, subsample_freq=1, objective="binary",
        verbose=-1, n_jobs=2, random_state=seed)


def glm():
    return LogisticRegression(C=1.0, solver="newton-cholesky", max_iter=100)


def _auc(y, p):
    return float(roc_auc_score(y, p))


def _rank(p):
    return rankdata(p) / len(p)


NOMBRES = ["lgb31_raw", "lgb4_raw", "lgb4_te", "glm", "glm_lgb4", "blend_rank"]


def preparar_pliegue(d_tr, d_va):
    """Matrices de un pliegue: crudas, TE anidado, y versiones escaladas."""
    X_tr, enc = ft.construir_features_crudas(d_tr, fit=True)
    X_va, _ = ft.construir_features_crudas(d_va, encoder=enc)
    X_tr = X_tr.astype(np.float32)
    X_va = X_va.astype(np.float32)
    y_tr = d_tr[ft.TARGET].values.astype(float)
    te_tr, te_va = target_encoding_anidado(construir_llaves(d_tr), y_tr,
                                           construir_llaves(d_va))
    XT_tr = np.hstack([X_tr.values, te_tr])
    XT_va = np.hstack([X_va.values, te_va])
    sc = StandardScaler()
    XS_tr = sc.fit_transform(XT_tr).astype(np.float32)
    XS_va = sc.transform(XT_va).astype(np.float32)
    return X_tr.values, X_va.values, XT_tr, XT_va, XS_tr, XS_va


def ajustar_predecir(d_tr, d_va, y_tr):
    """Ajusta los 5 modelos + blend en un pliegue. Devuelve dict nombre->p."""
    Xr_tr, Xr_va, XT_tr, XT_va, XS_tr, XS_va = preparar_pliegue(d_tr, d_va)
    p = {}
    t = time.time()
    m = lgb31(); m.fit(Xr_tr, y_tr); p["lgb31_raw"] = m.predict_proba(Xr_va)[:, 1]
    log.info("    lgb31_raw %.0fs", time.time() - t); t = time.time()
    m = lgb4(); m.fit(Xr_tr, y_tr); p["lgb4_raw"] = m.predict_proba(Xr_va)[:, 1]
    log.info("    lgb4_raw %.0fs", time.time() - t); t = time.time()
    m = lgb4(); m.fit(XT_tr, y_tr); p["lgb4_te"] = m.predict_proba(XT_va)[:, 1]
    log.info("    lgb4_te %.0fs", time.time() - t); t = time.time()
    g = glm(); g.fit(XS_tr, y_tr)
    log.info("    glm n_iter_=%s %.0fs", getattr(g, "n_iter_", "?"), time.time() - t)
    p["glm"] = g.predict_proba(XS_va)[:, 1]
    t = time.time()
    init_tr = g.decision_function(XS_tr)
    init_va = g.decision_function(XS_va)
    m = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=4,
        min_child_samples=100, reg_lambda=5.0, colsample_bytree=0.7,
        subsample=0.8, subsample_freq=1, objective="binary",
        verbose=-1, n_jobs=2, random_state=42)
    m.fit(XT_tr, y_tr, init_score=init_tr)
    raw_va = m.predict(XT_va, raw_score=True) + init_va
    p["glm_lgb4"] = 1.0 / (1.0 + np.exp(-raw_va))
    log.info("    glm_lgb4 %.0fs", time.time() - t)
    p["blend_rank"] = (_rank(p["lgb4_te"]) + _rank(p["glm_lgb4"])
                       + _rank(p["lgb4_raw"])) / 3.0
    return p


def diagnostico_hgb(d_tr, d_va, y_tr, y_va):
    """n_iter_ de HGB base en este pliegue: ¿el early stopping automático
    recortó las 200 iteraciones?"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xr_tr, enc = ft.construir_features_crudas(d_tr, fit=True)
    Xr_va, _ = ft.construir_features_crudas(d_va, encoder=enc)
    out = {}
    for es in ("auto", False):
        m = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
            min_samples_leaf=50, l2_regularization=1.0, random_state=42,
            early_stopping=es)
        m.fit(Xr_tr, y_tr)
        out[f"early_stopping={es}"] = {
            "n_iter_": int(m.n_iter_),
            "auc": round(_auc(y_va, m.predict_proba(Xr_va)[:, 1]), 5)}
    return out


def cv(tr):
    y = tr[ft.TARGET].values
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED_CV)
    oof = {n: np.zeros(len(tr)) for n in NOMBRES}
    aucs = {n: [] for n in NOMBRES}
    diag = None
    for i, (a, b) in enumerate(skf.split(tr, y)):
        log.info("pliegue %d/%d", i + 1, N_SPLITS)
        d_tr, d_va = tr.iloc[a], tr.iloc[b]
        if i == 0:
            diag = diagnostico_hgb(d_tr, d_va, y[a], y[b])
            log.info("  diagnóstico HGB (pliegue 0): %s", json.dumps(diag))
        p = ajustar_predecir(d_tr, d_va, y[a].astype(float))
        for n in NOMBRES:
            oof[n][b] = p[n]
            aucs[n].append(_auc(y[b], p[n]))
            log.info("  %-11s AUC pliegue = %.5f", n, aucs[n][-1])
        gc.collect()
    return y, oof, aucs, diag


def resumen(y, oof, aucs):
    base = np.array(aucs["lgb31_raw"])
    res = {"protocolo": "StratifiedKFold(5, shuffle, 42), ROC AUC, TE anidado",
           "modelos": {}}
    for n in NOMBRES:
        a = np.array(aucs[n])
        d = a - base
        gana = bool((d > 0).all() and d.mean() > 2 * d.std()) if n != "lgb31_raw" else False
        res["modelos"][n] = {
            "aucs_por_pliegue": [round(float(x), 5) for x in a],
            "auc_media": round(float(a.mean()), 5),
            "auc_std": round(float(a.std()), 5),
            "auc_oof_agrupado": round(_auc(y, oof[n]), 5),
            "delta_pareado_vs_lgb31": round(float(d.mean()), 5),
            "delta_std": round(float(d.std()), 5),
            "positivo_en_pliegues": int((d > 0).sum()),
            "cumple_regla_adopcion": gana,
        }
        log.info("%-11s media %.5f ± %.5f | Δ pareado %+.5f ± %.5f | +%d/5 | adopta=%s",
                 n, a.mean(), a.std(), d.mean(), d.std(), int((d > 0).sum()), gana)
    return res


def entrenar_final_y_escribir(tr, te, salida_sub, mejor):
    """Entrena con TODO train el modelo `mejor` (uno de NOMBRES) y escribe
    submission_v2.csv. Nunca sobrescribe submission.csv."""
    y = tr[ft.TARGET].values.astype(float)
    p = ajustar_predecir(tr, te.assign(**{ft.TARGET: 0}), y)[mejor]
    os.makedirs(salida_sub, exist_ok=True)
    ruta = os.path.join(salida_sub, "submission_v2.csv")
    sub = pd.DataFrame({"id": te["id"].values, "is_fraud": p})
    sub.to_csv(ruta, index=False)
    log.info("submission_v2 escrito: %s (%d filas, media %.5f)", ruta, len(sub),
             sub["is_fraud"].mean())
    return ruta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default=os.path.join(HERE, "datos"))
    ap.add_argument("--salida", default=os.path.join(HERE, "verificacion"))
    ap.add_argument("--max-filas", type=int, default=0)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--mejor", default="blend_rank", choices=NOMBRES)
    a = ap.parse_args()
    os.makedirs(a.salida, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(os.path.join(a.salida, "ensamble-v2-corrida.log"),
                                      mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])
    t0 = time.time()
    tr = pd.read_csv(os.path.join(a.datos, "train.csv"))
    if a.max_filas:
        tr = tr.sample(a.max_filas, random_state=0).reset_index(drop=True)
    log.info("train %s fraude %.5f", tr.shape, tr[ft.TARGET].mean())
    y, oof, aucs, diag = cv(tr)
    res = resumen(y, oof, aucs)
    res["diagnostico_hgb_pliegue0"] = diag
    res["tiempo_s"] = round(time.time() - t0, 1)
    res["filas"] = int(len(tr))
    with open(os.path.join(a.salida, "ensamble_v2.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(a.salida, "oof_ensamble_v2.npz"), **oof)
    if a.final:
        te = pd.read_csv(os.path.join(a.datos, "test.csv"))
        entrenar_final_y_escribir(tr, te, os.path.join(HERE, "salida-fraude"), a.mejor)
    log.info("listo en %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
