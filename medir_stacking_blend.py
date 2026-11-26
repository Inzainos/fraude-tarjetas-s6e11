#!/usr/bin/env python3
"""Medir stacking y blend ponderado sobre el OOF del ensamble_v2 del consultor.

Usa SOLO el OOF ya calculado (verificacion/oof_ensamble_v2_veredicto_2026-11-25.npz)
y train.csv para y y los pliegues. No reentrena los modelos base. No escribe
ningun submission: solo mide. Si algo gana, construir su submission es otro
paso con GO aparte de Elan.

Protocolo: StratifiedKFold(5, shuffle=True, random_state=42) sobre train en
orden de archivo (igual que ensamble_v2.py). Cada candidato se juzga con
NUESTRA regla: delta pareado por pliegue vs lgb4_raw con media >= +0.0005
y >= 4/5 pliegues positivos (delta > 0).

Candidatos:
  blend_pond_nested : pesos elegidos por pliegue con los otros 4 pliegues
                      (anidado, honesto) maximizando AUC OOF.
  blend_pond_global : pesos elegidos con todo el OOF (optimista, referencia).
  stack_lr          : LogisticRegression sobre las 6 columnas OOF, anidado.
  stack_lr_rank     : igual pero sobre rangos de las 6 columnas OOF.
"""
import argparse
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fraude_tarjetas as ft  # noqa: E402

SEED_CV = 42
NOMBRES = ["lgb31_raw", "lgb4_raw", "lgb4_te", "glm", "glm_lgb4", "blend_rank"]
log = logging.getLogger("medir_stacking_blend")


def _auc(y, p):
    return float(roc_auc_score(y, p))


def _pesos_grid():
    for a in (0.4, 0.5, 0.6, 0.7, 0.8):       # glm_lgb4
        for b in (0.0, 0.1, 0.2):              # glm
            for c in (0.0, 0.1, 0.2, 0.3):     # lgb4_te
                for d in (0.0, 0.1):           # lgb4_raw
                    s = a + b + c + d
                    if s <= 0:
                        continue
                    yield (a / s, b / s, c / s, d / s)


def _guardar(res, nombre, aucs, oof_pred, y, base_auc, extra):
    d = aucs - base_auc
    gana = bool(d.mean() >= 0.0005 and int((d > 0).sum()) >= 4)
    res["modelos"][nombre] = {
        "aucs_por_pliegue": [round(float(x), 5) for x in aucs],
        "auc_media": round(float(aucs.mean()), 5),
        "auc_std": round(float(aucs.std()), 5),
        "auc_oof_agrupado": round(_auc(y, oof_pred), 5),
        "delta_pareado_vs_lgb4_raw": round(float(d.mean()), 5),
        "delta_std": round(float(d.std()), 5),
        "positivo_en_pliegues": int((d > 0).sum()),
        "pliegues_con_delta_ge_0005": int((d >= 0.0005).sum()),
        "gana_regla_nuestra": gana,
        **extra,
    }
    log.info("%-18s media %.5f | Δ pareado %+.5f | +%d/5 | gana=%s",
             nombre, aucs.mean(), d.mean(), int((d > 0).sum()), gana)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default="/home/deamon/wt_deamonx/CASMI-sandbox/pruebas/fraude-s6e11/datos")
    ap.add_argument("--salida", default=os.path.join(HERE, "verificacion"))
    ap.add_argument("--oof", default=os.path.join(HERE, "verificacion", "oof_ensamble_v2_veredicto_2026-11-25.npz"))
    a = ap.parse_args()
    os.makedirs(a.salida, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(os.path.join(a.salida, "medir-stacking-blend.log"),
                                      mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])
    t0 = time.time()
    tr = pd.read_csv(os.path.join(a.datos, "train.csv"), usecols=[ft.TARGET])
    y = tr[ft.TARGET].values.astype(float)
    oof = dict(np.load(a.oof))
    for n in NOMBRES:
        assert n in oof, f"falta {n} en el OOF"
    log.info("filas %d fraude %.5f", len(y), y.mean())
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED_CV)
    folds = list(skf.split(np.zeros(len(y)), y))
    base_auc = np.array([_auc(y[b], oof["lgb4_raw"][b]) for _, b in folds])
    res = {"protocolo": "StratifiedKFold(5, shuffle, 42) sobre OOF del veredicto; regla nuestra: delta pareado vs lgb4_raw media >= +0.0005 y >= 4/5 positivos",
           "lgb4_raw_media": round(float(base_auc.mean()), 5),
           "lgb4_raw_aucs": [round(float(x), 5) for x in base_auc],
           "modelos": {}}
    cols = ["glm_lgb4", "glm", "lgb4_te", "lgb4_raw"]
    M = np.column_stack([oof[c] for c in cols])
    # --- blend ponderado anidado (honesto) ---
    aucs_nested, oof_nested, pesos_elegidos = [], np.zeros(len(y)), []
    for k, (tr_idx, va_idx) in enumerate(folds):
        best = (-1.0, None)
        for w in _pesos_grid():
            s = _auc(y[tr_idx], M[tr_idx] @ np.array(w))
            if s > best[0]:
                best = (s, w)
        w = best[1]
        pesos_elegidos.append([round(float(x), 4) for x in w])
        p_va = M[va_idx] @ np.array(w)
        oof_nested[va_idx] = p_va
        aucs_nested.append(_auc(y[va_idx], p_va))
        log.info("blend_nested pliegue %d/5 pesos=%s auc=%.5f",
                 k + 1, pesos_elegidos[-1], aucs_nested[-1])
    _guardar(res, "blend_pond_nested", np.array(aucs_nested), oof_nested, y,
             base_auc, {"pesos_por_pliegue": pesos_elegidos, "columnas": cols})
    # --- blend ponderado global (optimista, solo referencia) ---
    best = (-1.0, None)
    for w in _pesos_grid():
        s = _auc(y, M @ np.array(w))
        if s > best[0]:
            best = (s, w)
    w = best[1]
    oof_global = M @ np.array(w)
    aucs_global = np.array([_auc(y[b], oof_global[b]) for _, b in folds])
    _guardar(res, "blend_pond_global", aucs_global, oof_global, y, base_auc,
             {"pesos": [round(float(x), 4) for x in w], "columnas": cols,
              "nota": "pesos elegidos con todo el OOF; optimista"})
    log.info("blend_global pesos=%s media=%.5f",
             [round(float(x), 4) for x in w], aucs_global.mean())
    # --- stacking LR anidado ---
    X = np.column_stack([oof[n] for n in NOMBRES]).astype(np.float64)
    Xr = np.column_stack([rankdata(oof[n]) / len(y) for n in NOMBRES])
    for nombre, XX in (("stack_lr", X), ("stack_lr_rank", Xr)):
        aucs, oof_pred = [], np.zeros(len(y))
        for k, (tr_idx, va_idx) in enumerate(folds):
            m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            m.fit(XX[tr_idx], y[tr_idx])
            p = m.predict_proba(XX[va_idx])[:, 1]
            oof_pred[va_idx] = p
            aucs.append(_auc(y[va_idx], p))
            log.info("%s pliegue %d/5 auc=%.5f", nombre, k + 1, aucs[-1])
        _guardar(res, nombre, np.array(aucs), oof_pred, y, base_auc, {})
    res["tiempo_s"] = round(time.time() - t0, 1)
    res["filas"] = int(len(y))
    with open(os.path.join(a.salida, "medir_stacking_blend.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    log.info("listo en %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
