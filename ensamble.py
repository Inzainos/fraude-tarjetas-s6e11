#!/usr/bin/env python3
"""Ronda de ensamble — fraude con tarjeta (S6E11, ROC AUC).

Fase 1: variantes sobre features CRUDAS con el protocolo exacto del
informe (§5, §12): CV estratificada de 5 pliegues, semilla 42, ROC AUC.
Las tablas de referencia se calculan SOLO con el pliegue de
entrenamiento de cada fold (sin fuga), reutilizando
fraude_tarjetas.construir_features.

Variantes (todas sobre crudas):
  hgb_base_s{42,1,7}   HistGradientBoosting base §5 (200 it, lr 0.08,
                       31 hojas, min_samples_leaf 50, l2 1.0)
  hgb_g5_s{42,1,7}     mejor config del ajuste ligero (200 it, lr 0.08,
                       15 hojas, min_samples_leaf 20, l2 0.1)
  lgbm_s{42,1,7}       LightGBM (200 est, lr 0.08, 31 hojas,
                       min_child_samples 50, reg_lambda 1.0)
  logreg               regresión logística sobre features escaladas
                       (familia diversa, lineal)

NOTA 2026-11-09 ~00:05: ExtraTrees (300 árboles, profundidad 12) se
SACÓ de la lista tras medir >12 min por ajuste en un solo pliegue en
esta caja (1.5 CPU compartidas): su aporte marginal a un blend de 11
miembros no compensa ~60 min extra. LogReg queda como familia diversa.

Por pliegue se construyen las features UNA vez y se ajustan todos los
modelos sobre las mismas X_tr/X_va (mismos folds para todos: la
comparación pareada es válida). Se guardan las probabilidades
out-of-fold por variante en verificacion/oof_ensamble.npz y las AUC
por pliegue en verificacion/ensamble_fase1.json.

Un ajuste pesado a la vez; se libera memoria tras cada pliegue.
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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

BASE = HERE
VERIF = os.path.join(BASE, "verificacion")
os.makedirs(VERIF, exist_ok=True)

SEED_CV = 42
N_SPLITS = 5


def hgb_base(seed):
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=50, l2_regularization=1.0, random_state=seed)


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


def logreg():
    return LogisticRegression(C=0.5, max_iter=500, solver="lbfgs")


VARIANTES = [
    ("hgb_base_s42", lambda: hgb_base(42)),
    ("hgb_base_s1", lambda: hgb_base(1)),
    ("hgb_base_s7", lambda: hgb_base(7)),
    ("hgb_g5_s42", lambda: hgb_g5(42)),
    ("hgb_g5_s1", lambda: hgb_g5(1)),
    ("hgb_g5_s7", lambda: hgb_g5(7)),
    ("lgbm_s42", lambda: lgbm(42)),
    ("lgbm_s1", lambda: lgbm(1)),
    ("lgbm_s7", lambda: lgbm(7)),
    ("logreg", logreg),
]


def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] carga train.csv", flush=True)
    tr = pd.read_csv(os.path.join(BASE, "datos", "train.csv"))
    y = tr[m.TARGET].values
    print(f"  train {tr.shape}, fraude {y.mean():.5f}", flush=True)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=SEED_CV)
    oof = {nombre: np.zeros(len(tr), dtype=np.float64)
           for nombre, _ in VARIANTES}
    aucs = {nombre: [] for nombre, _ in VARIANTES}
    tiempos = {nombre: 0.0 for nombre, _ in VARIANTES}
    # Reanudación: si hay checkpoint de un pliegue completo, se retoma.
    ckpt_npz = os.path.join(VERIF, "oof_ensamble_parcial.npz")
    ckpt_json = os.path.join(VERIF, "ensamble_fase1_parcial.json")
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
        print(f"  reanuda desde checkpoint: {fold_inicio} pliegues "
              f"completos", flush=True)

    for i, (tr_idx, va_idx) in enumerate(skf.split(tr, y)):
        if i < fold_inicio:
            continue
        print(f"[{time.strftime('%H:%M:%S')}] pliegue {i + 1}/{N_SPLITS} "
              f"(features crudas)", flush=True)
        d_tr, d_va = tr.iloc[tr_idx], tr.iloc[va_idx]
        ref = m._tablas_referencia(d_tr)
        X_tr, enc = m.construir_features(d_tr, "crudas", ref,
                                         fit_encoder=True)
        X_va, _ = m.construir_features(d_va, "crudas", ref, encoder=enc)
        print(f"  X_tr {X_tr.shape} X_va {X_va.shape}", flush=True)
        # Escalado solo para logreg (se ajusta en el pliegue de entreno).
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        for nombre, fab in VARIANTES:
            t1 = time.time()
            modelo = fab()
            if nombre == "logreg":
                modelo.fit(X_tr_s, y[tr_idx])
                p = modelo.predict_proba(X_va_s)[:, 1]
            else:
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
        del X_tr, X_va, X_tr_s, X_va_s, scaler, ref, d_tr, d_va
        gc.collect()
        print(f"  pliegue {i + 1} listo ({time.time() - t0:.0f}s total)",
              flush=True)
        # Checkpoint por pliegue (la caja se reinicia: no perder trabajo).
        np.savez_compressed(ckpt_npz, **oof)
        parcial = {
            "pliegues_completos": i + 1,
            "variantes": {n: {"aucs_por_pliegue": aucs[n],
                              "tiempo_s": round(tiempos[n], 1)}
                          for n in aucs},
        }
        with open(ckpt_json, "w", encoding="utf-8") as fh:
            json.dump(parcial, fh, indent=2, ensure_ascii=False)

    np.savez_compressed(os.path.join(VERIF, "oof_ensamble.npz"), **oof)
    resumen = {
        "fase": 1,
        "features": "crudas",
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
    with open(os.path.join(VERIF, "ensamble_fase1.json"), "w",
              encoding="utf-8") as fh:
        json.dump(resumen, fh, indent=2, ensure_ascii=False)
    print(f"[{time.strftime('%H:%M:%S')}] fase 1 completa "
          f"({time.time() - t0:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
