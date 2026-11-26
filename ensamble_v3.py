#!/usr/bin/env python3
"""Pipeline v3 — fraude con tarjeta (Kaggle S6E11, ROC AUC).

Parte de glm_lgb4 (ensamble_v2.py; CV 0.95066, público 0.95112) y mide, con
el MISMO protocolo (StratifiedKFold(5, shuffle, 42), TE anidado sin fuga,
CPU), tres mejoras pequeñas reportadas en el foro de la competencia:

  1. Llaves de PARES para el target encoding (categoría x riesgo, extranjero
     x mismatch de IP, autenticación x reintentos de CVV, etc.).
  2. Bagging de semillas del cross-fit interno del TE (3 semillas): cada
     semilla da una codificación de entrenamiento distinta; se promedian los
     logits de los 3 modelos.
  3. Una familia distinta: MLP (sklearn) sobre crudas + logits TE. Sus
     errores difieren de los de los árboles/GLM, por eso se mezcla con peso
     chico (pesos FIJOS pre-registrados, sin ajustarlos al OOF).

Candidatos:
  glm_lgb4_base         control: idéntico a glm_lgb4 de v2 (llaves base)
  glm_lgb4_pares        + llaves de pares (semilla 17)
  glm_lgb4_pares_bag3   + bagging de 3 semillas
  mlp                   MLP solo
  blend_90_10           0.9 rango(bag3) + 0.1 rango(mlp)
  blend_80_20           0.8 rango(bag3) + 0.2 rango(mlp)

Regla de adopción (pareada por pliegue contra glm_lgb4_base): delta positivo
en los 5 pliegues, media >= +0.0001 y media > 2x la std del delta. Si ningún
candidato la cumple, NO hay motivo para reemplazar a glm_lgb4 (el envío
actual de 0.95112 queda como está).

Uso:
    python3 ensamble_v3.py --datos datos --salida verificacion
    python3 ensamble_v3.py --datos datos --salida verificacion --final --mejor blend_90_10
    python3 ensamble_v3.py --datos DIR --salida DIR --max-filas 20000   # prueba de humo

Requiere ensamble_v2.py y fraude_tarjetas.py en el mismo directorio.
Sin --final no escribe ningún submission. Con --final escribe
salida-fraude/submission_v3.csv (nunca sobrescribe otro envío). Nada se
envía a Kaggle: eso lo decide Elán. Cada corrida deja su log en
<salida>/ensamble-v3-corrida.log.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fraude_tarjetas as ft  # noqa: E402
import ensamble_v2 as v2  # noqa: E402

import lightgbm as lgb  # noqa: E402

SEED_CV = 42
N_SPLITS = 5
SEEDS_TE = [17, 23, 31]          # semillas del cross-fit interno (bagging)
log = logging.getLogger("ensamble_v3")

NOMBRES = ["glm_lgb4_base", "glm_lgb4_pares", "glm_lgb4_pares_bag3", "mlp",
           "blend_90_10", "blend_80_20"]
CONTROL = "glm_lgb4_base"

# Pares de columnas para llaves conjuntas (cada lado se discretiza).
PARES = [
    ("merchant_category", "merchant_risk_score"),
    ("is_foreign_transaction", "ip_country_mismatch"),
    ("auth_method", "cvv_retry_count"),
    ("is_ai_generated_scam_attempt", "used_vpn"),
    ("merchant_category", "auth_method"),
    ("billing_shipping_mismatch", "is_new_merchant"),
    ("cvv_retry_count", "prior_disputes"),
    ("merchant_category", "is_foreign_transaction"),
    ("auth_method", "is_ai_generated_scam_attempt"),
    ("velocity_score", "txn_count_last_24h"),
    ("amount_usd", "merchant_category"),
    ("amount_usd", "account_balance_usd"),
]
_LOG5 = {"amount_usd", "account_balance_usd"}      # floor(log1p(v) * 5)
_DIV2 = {"velocity_score"}                          # floor(v / 2)
_NUM = set(ft.COLS_NUMERICAS)


def _disc(df, c):
    """Discretiza una columna a texto para formar llaves conjuntas."""
    if c in _NUM:
        v = pd.to_numeric(df[c], errors="coerce").fillna(-1.0).values.astype(float)
        if c in _LOG5:
            v = np.floor(np.log1p(np.clip(v, 0, None)) * 5)
        elif c in _DIV2:
            v = np.floor(v / 2.0)
        else:
            v = np.floor(v)
        return v.astype(np.int64).astype(str)
    return df[c].astype(str).values


def llaves_pares(df):
    out = {}
    for a, b in PARES:
        out[f"par__{a}__{b}"] = np.char.add(np.char.add(_disc(df, a).astype(str), "|"),
                                            _disc(df, b).astype(str))
    return pd.DataFrame(out, index=df.index)


def _auc(y, p):
    return float(roc_auc_score(y, p))


def _rank(p):
    return rankdata(p) / len(p)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def ajustar_glm_boost(XT_tr, XT_va, y_tr, seed):
    """GLM (newton-cholesky) sobre crudas+TE y árboles de 4 hojas que parten
    de su logit (init_score). Devuelve (logit_va, XS_tr, XS_va)."""
    sc = StandardScaler()
    XS_tr = sc.fit_transform(XT_tr).astype(np.float32)
    XS_va = sc.transform(XT_va).astype(np.float32)
    g = v2.glm()
    g.fit(XS_tr, y_tr)
    init_tr = g.decision_function(XS_tr)
    init_va = g.decision_function(XS_va)
    m = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=4,
        min_child_samples=100, reg_lambda=5.0, colsample_bytree=0.7,
        subsample=0.8, subsample_freq=1, objective="binary",
        verbose=-1, n_jobs=2, random_state=seed)
    m.fit(XT_tr, y_tr, init_score=init_tr)
    raw_va = m.predict(XT_va, raw_score=True) + init_va
    return raw_va, XS_tr, XS_va


def ajustar_mlp(XS_tr, y_tr, seed=42, max_epocas=20, paciencia=3):
    """MLP (sklearn) con early stopping propio por AUC en un hold-out
    interno estratificado del 5% del entrenamiento. NO se usa el
    early_stopping de sklearn: monitorea accuracy, que con ~98% de
    negativos casi no cambia y detiene el ajuste antes de aprender.
    Conserva los pesos de la mejor época. Devuelve (modelo, mejor_epoca)."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y_tr))
    i_fit, i_ho = train_test_split(idx, test_size=0.05, stratify=y_tr,
                                   random_state=seed)
    m = MLPClassifier(
        hidden_layer_sizes=(256, 128), activation="relu", solver="adam",
        batch_size=4096, learning_rate_init=1e-3, alpha=1e-4,
        shuffle=True, random_state=seed)
    clases = np.array([0, 1])
    mejor, mejor_ep, sin_mejora = -1.0, 0, 0
    mejores_pesos = None
    for ep in range(1, max_epocas + 1):
        m.partial_fit(XS_tr[i_fit], y_tr[i_fit].astype(int), classes=clases)
        auc = _auc(y_tr[i_ho], m.predict_proba(XS_tr[i_ho])[:, 1])
        log.info("      mlp época %d: AUC hold-out %.5f", ep, auc)
        if auc > mejor + 1e-5:
            mejor, mejor_ep, sin_mejora = auc, ep, 0
            mejores_pesos = ([w.copy() for w in m.coefs_],
                             [b.copy() for b in m.intercepts_])
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    m.coefs_, m.intercepts_ = mejores_pesos
    return m, mejor_ep


def ajustar_predecir(d_tr, d_va, y_tr):
    """Ajusta todos los candidatos en un pliegue. Devuelve dict nombre->p."""
    X_tr, enc = ft.construir_features_crudas(d_tr, fit=True)
    X_va, _ = ft.construir_features_crudas(d_va, encoder=enc)
    X_tr = X_tr.values.astype(np.float32)
    X_va = X_va.values.astype(np.float32)
    base_tr, base_va = v2.construir_llaves(d_tr), v2.construir_llaves(d_va)
    n_base = base_tr.shape[1]
    ll_tr = pd.concat([base_tr, llaves_pares(d_tr)], axis=1)
    ll_va = pd.concat([base_va, llaves_pares(d_va)], axis=1)
    log.info("    llaves: %d base + %d pares", n_base, ll_tr.shape[1] - n_base)

    p, raws, XS_tr0, XS_va0 = {}, [], None, None
    for i, s in enumerate(SEEDS_TE):
        t = time.time()
        v2.SEED_INNER = s
        te_tr, te_va = v2.target_encoding_anidado(ll_tr, y_tr, ll_va)
        XT_tr = np.hstack([X_tr, te_tr])
        XT_va = np.hstack([X_va, te_va])
        if i == 0:
            # Control: llaves base solamente (equivale al glm_lgb4 de v2).
            XTb_tr = np.hstack([X_tr, te_tr[:, :n_base]])
            XTb_va = np.hstack([X_va, te_va[:, :n_base]])
            raw_b, _, _ = ajustar_glm_boost(XTb_tr, XTb_va, y_tr, seed=42)
            p[CONTROL] = _sigmoid(raw_b)
            del XTb_tr, XTb_va
        raw_p, XS_tr, XS_va = ajustar_glm_boost(XT_tr, XT_va, y_tr, seed=s)
        raws.append(raw_p)
        if i == 0:
            p["glm_lgb4_pares"] = _sigmoid(raw_p)
            XS_tr0, XS_va0 = XS_tr, XS_va
        log.info("    semilla TE %d: %.0fs", s, time.time() - t)
        del XT_tr, XT_va, te_tr, te_va
        gc.collect()
    v2.SEED_INNER = SEEDS_TE[0]
    p["glm_lgb4_pares_bag3"] = _sigmoid(np.mean(raws, axis=0))

    t = time.time()
    m, mejor_ep = ajustar_mlp(XS_tr0, y_tr, seed=42)
    p["mlp"] = m.predict_proba(XS_va0)[:, 1]
    log.info("    mlp %.0fs (mejor época %d)", time.time() - t, mejor_ep)
    rb, rm = _rank(p["glm_lgb4_pares_bag3"]), _rank(p["mlp"])
    p["blend_90_10"] = 0.9 * rb + 0.1 * rm
    p["blend_80_20"] = 0.8 * rb + 0.2 * rm
    return p


def cv(tr):
    y = tr[ft.TARGET].values
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED_CV)
    oof = {n: np.zeros(len(tr)) for n in NOMBRES}
    aucs = {n: [] for n in NOMBRES}
    for i, (a, b) in enumerate(skf.split(tr, y)):
        log.info("pliegue %d/%d", i + 1, N_SPLITS)
        p = ajustar_predecir(tr.iloc[a], tr.iloc[b], y[a].astype(float))
        for n in NOMBRES:
            oof[n][b] = p[n]
            aucs[n].append(_auc(y[b], p[n]))
            log.info("  %-20s AUC pliegue = %.5f", n, aucs[n][-1])
        gc.collect()
    return y, oof, aucs


def resumen(y, oof, aucs):
    base = np.array(aucs[CONTROL])
    res = {"protocolo": "StratifiedKFold(5, shuffle, 42), ROC AUC, TE anidado",
           "control": CONTROL,
           "regla": "delta pareado > 0 en 5/5, media >= 0.0001 y media > 2*std(delta)",
           "modelos": {}}
    for n in NOMBRES:
        a = np.array(aucs[n])
        d = a - base
        gana = bool(n != CONTROL and (d > 0).all() and d.mean() >= 0.0001
                    and d.mean() > 2 * d.std())
        res["modelos"][n] = {
            "aucs_por_pliegue": [round(float(x), 5) for x in a],
            "auc_media": round(float(a.mean()), 5),
            "auc_std": round(float(a.std()), 5),
            "auc_oof_agrupado": round(_auc(y, oof[n]), 5),
            "delta_pareado_vs_control": round(float(d.mean()), 5),
            "delta_std": round(float(d.std()), 5),
            "positivo_en_pliegues": int((d > 0).sum()),
            "cumple_regla_adopcion": gana,
        }
        log.info("%-20s media %.5f ± %.5f | Δ pareado %+.5f ± %.5f | +%d/5 | adopta=%s",
                 n, a.mean(), a.std(), d.mean(), d.std(), int((d > 0).sum()), gana)
    return res


def entrenar_final_y_escribir(tr, te, salida_sub, mejor):
    y = tr[ft.TARGET].values.astype(float)
    p = ajustar_predecir(tr, te.assign(**{ft.TARGET: 0}), y)[mejor]
    os.makedirs(salida_sub, exist_ok=True)
    ruta = os.path.join(salida_sub, "submission_v3.csv")
    sub = pd.DataFrame({"id": te["id"].values, "is_fraud": p})
    sub.to_csv(ruta, index=False)
    log.info("submission_v3 escrito: %s (%d filas, media %.5f)", ruta, len(sub),
             sub["is_fraud"].mean())
    return ruta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default=os.path.join(HERE, "datos"))
    ap.add_argument("--salida", default=os.path.join(HERE, "verificacion"))
    ap.add_argument("--max-filas", type=int, default=0)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--mejor", default="blend_90_10", choices=NOMBRES)
    a = ap.parse_args()
    os.makedirs(a.salida, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(os.path.join(a.salida, "ensamble-v3-corrida.log"),
                                      mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])
    t0 = time.time()
    tr = pd.read_csv(os.path.join(a.datos, "train.csv"))
    if a.max_filas:
        tr = tr.sample(a.max_filas, random_state=0).reset_index(drop=True)
    log.info("train %s fraude %.5f", tr.shape, tr[ft.TARGET].mean())
    y, oof, aucs = cv(tr)
    res = resumen(y, oof, aucs)
    res["tiempo_s"] = round(time.time() - t0, 1)
    res["filas"] = int(len(tr))
    with open(os.path.join(a.salida, "ensamble_v3.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(a.salida, "oof_ensamble_v3.npz"), **oof)
    if a.final:
        te = pd.read_csv(os.path.join(a.datos, "test.csv"))
        entrenar_final_y_escribir(tr, te, os.path.join(HERE, "salida-fraude"), a.mejor)
    log.info("listo en %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
