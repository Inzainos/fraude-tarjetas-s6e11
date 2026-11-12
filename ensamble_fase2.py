#!/usr/bin/env python3
"""Ronda de ensamble, fase 2 — features COMBINADAS (crudas + señales)
solo para las dos mejores familias de la fase 1 (HGB g5 y LightGBM),
2 semillas cada una. Mismo protocolo: CV estratificada 5 pliegues,
semilla 42, ROC AUC; tablas de referencia solo con el pliegue de
entrenamiento. Checkpoint por pliegue (la caja se reinicia).
"""
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fraude_tarjetas as m  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

BASE = HERE
VERIF = os.path.join(BASE, "verificacion")
os.makedirs(VERIF, exist_ok=True)

SEED_CV = 42
N_SPLITS = 5


def hgb_g5(seed):
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=0.1, random_state=seed)


def lgbm(seed):
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.08, num_leaves=31,
        min_child_samples=50, reg_lambda=1.0, objective="binary",
        verbose=-1, n_jobs=4, random_state=seed)


VARIANTES = [
    ("hgb_g5_comb_s42", lambda: hgb_g5(42)),
    ("hgb_g5_comb_s1", lambda: hgb_g5(1)),
    ("lgbm_comb_s42", lambda: lgbm(42)),
    ("lgbm_comb_s1", lambda: lgbm(1)),
]


def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] fase 2: combinadas, "
          f"g5 + lgbm", flush=True)
    tr = pd.read_csv(os.path.join(BASE, "datos", "train.csv"))
    y = tr[m.TARGET].values
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=SEED_CV)
    oof = {nombre: np.zeros(len(tr), dtype=np.float64)
           for nombre, _ in VARIANTES}
    aucs = {nombre: [] for nombre, _ in VARIANTES}
    tiempos = {nombre: 0.0 for nombre, _ in VARIANTES}
    ckpt_npz = os.path.join(VERIF, "oof_ensamble_comb_parcial.npz")
    ckpt_json = os.path.join(VERIF, "ensamble_fase2_parcial.json")
    fold_inicio = 0
    if os.path.exists(ckpt_npz) and os.path.exists(ckpt_json):
        _d = np.load(ckpt_npz)
        _j = json.load(open(ckpt_json, encoding="utf-8"))
        fold_inicio = int(_j["pliegues_completos"])
        for nombre in oof:
            oof[nombre] = _d[nombre]
            aucs[nombre] = list(_j["variantes"][nombre]["aucs_por_pliegue"])
            tiempos[nombre] = float(
                _j["variantes"][nombre].get("tiempo_s", 0.0))
        print(f"  reanuda desde checkpoint: {fold_inicio} pliegues",
              flush=True)

    for i, (tr_idx, va_idx) in enumerate(skf.split(tr, y)):
        if i < fold_inicio:
            continue
        print(f"[{time.strftime('%H:%M:%S')}] pliegue {i + 1}/{N_SPLITS} "
              f"(features combinadas)", flush=True)
        d_tr, d_va = tr.iloc[tr_idx], tr.iloc[va_idx]
        ref = m._tablas_referencia(d_tr)
        X_tr, enc = m.construir_features(d_tr, "combinadas", ref,
                                         fit_encoder=True)
        X_va, _ = m.construir_features(d_va, "combinadas", ref,
                                       encoder=enc)
        print(f"  X_tr {X_tr.shape} X_va {X_va.shape}", flush=True)
        for nombre, fab in VARIANTES:
            t1 = time.time()
            modelo = fab()
            modelo.fit(X_tr, y[tr_idx])
            p = modelo.predict_proba(X_va)[:, 1]
            oof[nombre][va_idx] = p
            auc = float(roc_auc_score(y[va_idx], p))
            aucs[nombre].append(round(auc, 4))
            tiempos[nombre] += time.time() - t1
            print(f"    {nombre}: AUC pliegue = {auc:.4f} "
                  f"({time.time() - t1:.0f}s)", flush=True)
            del modelo, p
            gc.collect()
        del X_tr, X_va, ref, d_tr, d_va
        gc.collect()
        print(f"  pliegue {i + 1} listo ({time.time() - t0:.0f}s total)",
              flush=True)
        np.savez_compressed(ckpt_npz, **oof)
        parcial = {
            "pliegues_completos": i + 1,
            "variantes": {n: {"aucs_por_pliegue": aucs[n],
                              "tiempo_s": round(tiempos[n], 1)}
                          for n in aucs},
        }
        with open(ckpt_json, "w", encoding="utf-8") as fh:
            json.dump(parcial, fh, indent=2, ensure_ascii=False)

    np.savez_compressed(os.path.join(VERIF, "oof_ensamble_comb.npz"),
                        **oof)
    resumen = {
        "fase": 2,
        "features": "combinadas",
        "protocolo": "CV estratificada 5 pliegues, semilla 42, ROC AUC",
        "variantes": {},
    }
    for nombre in aucs:
        a = aucs[nombre]
        resumen["variantes"][nombre] = {
            "aucs_por_pliegue": a,
            "auc_media": round(float(np.mean(a)), 4),
            "auc_std": round(float(np.std(a)), 4),
            "auc_oof_agrupado": round(
                float(roc_auc_score(y, oof[nombre])), 4),
            "tiempo_s": round(tiempos[nombre], 1),
        }
        print(f"{nombre}: media {np.mean(a):.4f} ± {np.std(a):.4f} "
              f"| OOF agrupado {roc_auc_score(y, oof[nombre]):.4f}",
              flush=True)
    with open(os.path.join(VERIF, "ensamble_fase2.json"), "w",
              encoding="utf-8") as fh:
        json.dump(resumen, fh, indent=2, ensure_ascii=False)
    print(f"[{time.strftime('%H:%M:%S')}] fase 2 completa "
          f"({time.time() - t0:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
