#!/usr/bin/env python3
"""Análisis de blends a partir de las predicciones out-of-fold de la
fase 1 (verificacion/oof_ensamble.npz).

Blends (pesos iguales, sin ajuste a OOF):
  - promedio de semillas por familia (hgb_base, hgb_g5, lgbm)
  - gran promedio GBM (9 variantes)
  - gran promedio GBM + logreg + extratrees (11 variantes)
  - GBM g5+lgbm (6 variantes)
  - versiones rank-mean de los grandes blends
Para cada blend: AUC media por pliegue (mismos folds, semilla 42),
std, AUC OOF agrupado, y delta pareado por pliegue contra la
baseline hgb_base_s42 (mismos folds).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fraude_tarjetas as m  # noqa: E402

from scipy.stats import rankdata  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

BASE = HERE
VERIF = os.path.join(BASE, "verificacion")

FAMILIAS = {
    "hgb_base": ["hgb_base_s42", "hgb_base_s1", "hgb_base_s7"],
    "hgb_g5": ["hgb_g5_s42", "hgb_g5_s1", "hgb_g5_s7"],
    "lgbm": ["lgbm_s42", "lgbm_s1", "lgbm_s7"],
}
GBM = sum(FAMILIAS.values(), [])
TODOS = GBM + ["logreg"]


def media(lista_nombres, oof):
    return np.mean([oof[n] for n in lista_nombres], axis=0)


def rank_mean(lista_nombres, oof):
    ranks = [rankdata(oof[n]) / len(oof[n]) for n in lista_nombres]
    return np.mean(ranks, axis=0)


def main():
    tr = pd.read_csv(os.path.join(BASE, "datos", "train.csv"))
    y = tr[m.TARGET].values
    oof = dict(np.load(os.path.join(VERIF, "oof_ensamble.npz")))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(tr, y))

    blends = {}
    for fam, miembros in FAMILIAS.items():
        blends[f"promedio_{fam}"] = media(miembros, oof)
    blends["gran_promedio_gbm"] = media(GBM, oof)
    blends["gran_promedio_todos"] = media(TODOS, oof)
    blends["g5_lgbm"] = media(FAMILIAS["hgb_g5"] + FAMILIAS["lgbm"], oof)
    blends["rankmean_gran_gbm"] = rank_mean(GBM, oof)
    blends["rankmean_todos"] = rank_mean(TODOS, oof)

    base_folds = []
    for _tr, va in folds:
        base_folds.append(float(roc_auc_score(y[va], oof["hgb_base_s42"][va])))

    out = {"blends": {}, "baseline_hgb_base_s42_por_pliegue": base_folds}
    for nombre, p in blends.items():
        aucs_fold = [round(float(roc_auc_score(y[va], p[va])), 4)
                     for _tr, va in folds]
        delta = [round(a - b, 4) for a, b in zip(aucs_fold, base_folds)]
        out["blends"][nombre] = {
            "aucs_por_pliegue": aucs_fold,
            "auc_media": round(float(np.mean(aucs_fold)), 4),
            "auc_std": round(float(np.std(aucs_fold)), 4),
            "auc_oof_agrupado": round(float(roc_auc_score(y, p)), 4),
            "delta_vs_hgb_base_s42_por_pliegue": delta,
            "delta_media": round(float(np.mean(delta)), 4),
            "delta_std": round(float(np.std(delta)), 4),
        }
        print(f"{nombre}: media {np.mean(aucs_fold):.4f} ± "
              f"{np.std(aucs_fold):.4f} | OOF {roc_auc_score(y, p):.4f} | "
              f"Δ vs base {np.mean(delta):+.4f} ± {np.std(delta):.4f}",
              flush=True)
    with open(os.path.join(VERIF, "ensamble_blends.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("OK: verificacion/ensamble_blends.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
