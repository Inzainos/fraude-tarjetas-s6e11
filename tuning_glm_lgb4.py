#!/usr/bin/env python3
"""Tuning ligero del glm_lgb4 del consultor (fraude S6E11).

Reusa el codigo exacto de ensamble_v2.py: por pliegue se calculan una sola
vez el TE anidado y el logit del GLM (C=1.0), y sobre esos mismos datos se
entrenan 6 variantes LightGBM. Solo mide; no escribe ningun submission.
Si alguna variante gana, construir su submission es otro paso con GO aparte.

Protocolo: StratifiedKFold(5, shuffle=True, random_state=42), igual que
ensamble_v2.py. Cada variante se juzga con NUESTRA regla vs lgb4_raw
(delta pareado media >= +0.0005 y >= 4/5 positivos) y, lo decisivo, contra
la variante A (el glm_lgb4 actual, 0.95066 CV / 0.95112 publico).

Variantes (todas LightGBM 4 hojas salvo D, sobre XT = crudas + TE, con
init_score = logit del GLM):
  A control     400 arboles, lr 0.05, leaves 4, reg_lambda 5.0 (el actual;
                debe reproducir ~0.95066 o el harness esta roto)
  B fino        800 arboles, lr 0.025, leaves 4, reg_lambda 5.0
  C mas_boost   800 arboles, lr 0.05, leaves 4, reg_lambda 5.0
  D mas_hojas   400 arboles, lr 0.05, leaves 7, reg_lambda 5.0
  E mas_reg     400 arboles, lr 0.05, leaves 4, reg_lambda 10.0
  lgb4_raw      control de referencia (sin TE, sin init), debe dar ~0.94665
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ensamble_v2 as ev  # noqa: E402
import fraude_tarjetas as ft  # noqa: E402
import lightgbm as lgb  # noqa: E402

log = logging.getLogger("tuning_glm_lgb4")

CONFIGS = [
    ("A_control", dict(n_estimators=400, learning_rate=0.05, num_leaves=4, reg_lambda=5.0)),
    ("B_fino", dict(n_estimators=800, learning_rate=0.025, num_leaves=4, reg_lambda=5.0)),
    ("C_mas_boost", dict(n_estimators=800, learning_rate=0.05, num_leaves=4, reg_lambda=5.0)),
    ("D_mas_hojas", dict(n_estimators=400, learning_rate=0.05, num_leaves=7, reg_lambda=5.0)),
    ("E_mas_reg", dict(n_estimators=400, learning_rate=0.05, num_leaves=4, reg_lambda=10.0)),
]


def _lgbm(**kw):
    return lgb.LGBMClassifier(
        min_child_samples=100, colsample_bytree=0.7, subsample=0.8,
        subsample_freq=1, objective="binary", verbose=-1, n_jobs=2,
        random_state=42, **kw)


def _auc(y, p):
    return float(roc_auc_score(y, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default="/home/deamon/wt_deamonx/CASMI-sandbox/pruebas/fraude-s6e11/datos")
    ap.add_argument("--salida", default=os.path.join(HERE, "verificacion"))
    a = ap.parse_args()
    os.makedirs(a.salida, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(os.path.join(a.salida, "tuning-glm-lgb4.log"),
                                      mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])
    t0 = time.time()
    tr = pd.read_csv(os.path.join(a.datos, "train.csv"))
    y = tr[ft.TARGET].values.astype(float)
    log.info("train %s fraude %.5f", tr.shape, tr[ft.TARGET].mean())
    nombres = ["lgb4_raw"] + [c[0] for c in CONFIGS]
    oof = {n: np.zeros(len(tr)) for n in nombres}
    aucs = {n: [] for n in nombres}
    skf = StratifiedKFold(n_splits=ev.N_SPLITS, shuffle=True, random_state=ev.SEED_CV)
    for i, (a_idx, b_idx) in enumerate(skf.split(tr, y)):
        log.info("pliegue %d/%d", i + 1, ev.N_SPLITS)
        d_tr, d_va = tr.iloc[a_idx], tr.iloc[b_idx]
        y_tr = y[a_idx].astype(float)
        Xr_tr, Xr_va, XT_tr, XT_va, XS_tr, XS_va = ev.preparar_pliegue(d_tr, d_va)
        # control lgb4_raw (sin TE, sin init)
        t = time.time()
        m = ev.lgb4()
        m.fit(Xr_tr, y_tr)
        p = m.predict_proba(Xr_va)[:, 1]
        oof["lgb4_raw"][b_idx] = p
        aucs["lgb4_raw"].append(_auc(y[b_idx], p))
        log.info("  lgb4_raw auc=%.5f %.0fs", aucs["lgb4_raw"][-1], time.time() - t)
        # GLM init (una sola vez por pliegue)
        g = ev.glm()
        g.fit(XS_tr, y_tr)
        init_tr = g.decision_function(XS_tr)
        init_va = g.decision_function(XS_va)
        for nombre, kw in CONFIGS:
            t = time.time()
            m = _lgbm(**kw)
            m.fit(XT_tr, y_tr, init_score=init_tr)
            raw_va = m.predict(XT_va, raw_score=True) + init_va
            p = 1.0 / (1.0 + np.exp(-raw_va))
            oof[nombre][b_idx] = p
            aucs[nombre].append(_auc(y[b_idx], p))
            log.info("  %-12s auc=%.5f %.0fs", nombre, aucs[nombre][-1], time.time() - t)
        gc.collect()
    base = np.array(aucs["lgb4_raw"])
    control = np.array(aucs["A_control"])
    res = {"protocolo": "StratifiedKFold(5, shuffle, 42); TE anidado + GLM init compartidos por pliegue; regla nuestra vs lgb4_raw",
           "lgb4_raw_media": round(float(base.mean()), 5),
           "A_control_media": round(float(control.mean()), 5),
           "modelos": {}}
    for nombre in nombres:
        arr = np.array(aucs[nombre])
        d = arr - base
        dA = arr - control
        gana = bool(d.mean() >= 0.0005 and int((d > 0).sum()) >= 4)
        res["modelos"][nombre] = {
            "aucs_por_pliegue": [round(float(x), 5) for x in arr],
            "auc_media": round(float(arr.mean()), 5),
            "auc_std": round(float(arr.std()), 5),
            "auc_oof_agrupado": round(_auc(y, oof[nombre]), 5),
            "delta_pareado_vs_lgb4_raw": round(float(d.mean()), 5),
            "delta_std": round(float(d.std()), 5),
            "positivo_en_pliegues": int((d > 0).sum()),
            "delta_vs_A_control": round(float(dA.mean()), 5),
            "gana_regla_nuestra": gana,
            "supera_A_control": bool(dA.mean() >= 0.0005 and int((dA > 0).sum()) >= 4),
        }
        log.info("%-12s media %.5f | Δ vs raw %+.5f | Δ vs A %+.5f | +%d/5 | gana=%s",
                 nombre, arr.mean(), d.mean(), dA.mean(), int((d > 0).sum()), gana)
    res["tiempo_s"] = round(time.time() - t0, 1)
    res["filas"] = int(len(tr))
    with open(os.path.join(a.salida, "tuning_glm_lgb4.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    log.info("listo en %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
