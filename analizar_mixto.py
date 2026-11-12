#!/usr/bin/env python3
"""Blends mixtos: combina las OOF de la fase 1 (crudas) con las de la
fase 2 (combinadas). Mismos folds, semilla 42. Actualiza
verificacion/ensamble_blends.json con la sección "blends_mixtos".
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

GBM_CRUDAS = ["hgb_base_s42", "hgb_base_s1", "hgb_base_s7",
              "hgb_g5_s42", "hgb_g5_s1", "hgb_g5_s7",
              "lgbm_s42", "lgbm_s1", "lgbm_s7"]
COMB = ["hgb_g5_comb_s42", "hgb_g5_comb_s1",
        "lgbm_comb_s42", "lgbm_comb_s1"]


def media(nombres, oof):
    return np.mean([oof[n] for n in nombres], axis=0)


def rank_mean(nombres, oof):
    ranks = [rankdata(oof[n]) / len(oof[n]) for n in nombres]
    return np.mean(ranks, axis=0)


def main():
    tr = pd.read_csv(os.path.join(BASE, "datos", "train.csv"))
    y = tr[m.TARGET].values
    oof1 = dict(np.load(os.path.join(VERIF, "oof_ensamble.npz")))
    oof2 = dict(np.load(os.path.join(VERIF, "oof_ensamble_comb.npz")))
    oof = {**oof1, **oof2}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(tr, y))
    base_folds = [float(roc_auc_score(y[va], oof["hgb_base_s42"][va]))
                  for _tr, va in folds]

    gran_gbm = media(GBM_CRUDAS, oof)
    prom_comb = media(COMB, oof)
    blends = {
        "promedio_comb": prom_comb,
        "mixto_gbm": media(GBM_CRUDAS + COMB, oof),
        "mixto_todos": media(GBM_CRUDAS + COMB + ["logreg"], oof),
        "mixto_modos": 0.5 * gran_gbm + 0.5 * prom_comb,
        "rankmean_mixto_gbm": rank_mean(GBM_CRUDAS + COMB, oof),
        "rankmean_mixto_todos": rank_mean(GBM_CRUDAS + COMB + ["logreg"], oof),
    }
    out = json.load(open(os.path.join(VERIF, "ensamble_blends.json"),
                         encoding="utf-8"))
    out["blends_mixtos"] = {}
    for nombre, p in blends.items():
        aucs_fold = [round(float(roc_auc_score(y[va], p[va])), 4)
                     for _tr, va in folds]
        delta = [round(a - b, 4) for a, b in zip(aucs_fold, base_folds)]
        out["blends_mixtos"][nombre] = {
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
    print("OK: ensamble_blends.json actualizado con blends_mixtos",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
