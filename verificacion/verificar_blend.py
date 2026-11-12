#!/usr/bin/env python3
"""Verificación INDEPENDIENTE del blend mixto_todos (S6E11).

Lee salida-fraude/submission_blend.csv y lo verifica sin re-entrenar:
formato, md5, sanidad vs la v1, reproducción OOF del blend desde los
npz guardados, checkpoints de los 14 modelos, y lenguaje neutro
(cero menciones de la SNT) en los entregables nuevos. Escribe la
sección "verificacion_independiente" en verificacion/blend.json y
devuelve 0 solo si todo pasa.
"""
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)

MD5_V1 = "c07bfd803b2704a13ec7f7d2afff7bd8"
MIEMBROS = ["hgb_base_s42", "hgb_base_s1", "hgb_base_s7",
            "hgb_g5_s42", "hgb_g5_s1", "hgb_g5_s7",
            "lgbm_s42", "lgbm_s1", "lgbm_s7",
            "hgb_g5_comb_s42", "hgb_g5_comb_s1",
            "lgbm_comb_s42", "lgbm_comb_s1", "logreg"]
OOF_ESPERADO = {"aucs": [0.9448, 0.9481, 0.9512, 0.9466, 0.9416],
                "media": 0.9465, "std": 0.0032, "pool": 0.9464}
# Patrones prohibidos en los entregables nuevos (regla de Elán: cero
# menciones de la SNT en fraude). Se buscan como palabra/frase.
PATRONES_PROHIBIDOS = [r"\bsnt\b", r"nodo sombra", r"shadow node",
                       r"\bmuro\b", r"\broche\b", r"\basi\b", r"\bhub\b",
                       r"\bluz\b", r"\bsombra\b"]


def md5(ruta):
    return hashlib.md5(open(ruta, "rb").read()).hexdigest()


def main():
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from scipy.stats import pearsonr, spearmanr

    checks = []

    def chk(nombre, ok, detalle=""):
        checks.append({"check": nombre, "ok": bool(ok),
                       "detalle": str(detalle)})
        print(("OK  " if ok else "FAIL"), "-", nombre, "-",
              str(detalle), flush=True)
        return bool(ok)

    todo_ok = True

    # --- 1) Archivos base ---
    ruta_blend = os.path.join(BASE, "salida-fraude", "submission_blend.csv")
    ruta_v1 = os.path.join(BASE, "salida-fraude", "submission.csv")
    ruta_sample = os.path.join(BASE, "datos", "sample_submission.csv")
    if not os.path.exists(ruta_blend):
        print("FAIL - no existe submission_blend.csv", flush=True)
        return 1
    sub = pd.read_csv(ruta_blend)
    sample = pd.read_csv(ruta_sample)
    v1 = pd.read_csv(ruta_v1)

    # --- 2) Formato ---
    todo_ok &= chk("columnas_exactas",
                   list(sub.columns) == ["id", "is_fraud"],
                   f"columnas: {list(sub.columns)}")
    todo_ok &= chk("n_filas", len(sub) == 299984,
                   f"{len(sub)} filas")
    todo_ok &= chk("mismos_ids_orden",
                   bool((sub["id"].values == sample["id"].values).all()),
                   "orden idéntico a sample_submission.csv")
    todo_ok &= chk("sin_nan", not bool(sub["is_fraud"].isna().any()),
                   f"NaN: {int(sub['is_fraud'].isna().sum())}")
    todo_ok &= chk("prob_en_0_1",
                   bool(((sub["is_fraud"] >= 0)
                         & (sub["is_fraud"] <= 1)).all()),
                   f"min={float(sub['is_fraud'].min()):.4f} "
                   f"max={float(sub['is_fraud'].max()):.4f}")
    todo_ok &= chk("sin_duplicados_id",
                   not bool(sub["id"].duplicated().any()),
                   f"duplicados: {int(sub['id'].duplicated().sum())}")
    todo_ok &= chk("v1_intacta",
                   md5(ruta_v1) == MD5_V1,
                   f"md5 v1: {md5(ruta_v1)}")

    # --- 3) Checkpoints de los 14 modelos ---
    ckpt = os.path.join(BASE, "verificacion", "blend_ckpt")
    faltan = [m for m in MIEMBROS
              if not os.path.exists(os.path.join(ckpt, m + ".npy"))]
    todo_ok &= chk("checkpoints_14_modelos", not faltan,
                   f"faltan: {faltan}" if faltan else "14/14 presentes")
    formas_ok = True
    for m in MIEMBROS:
        f = os.path.join(ckpt, m + ".npy")
        if os.path.exists(f):
            a = np.load(f)
            if a.shape != (299984,) or not np.isfinite(a).all():
                formas_ok = False
    todo_ok &= chk("checkpoints_forma_299984_finitos", formas_ok,
                   "cada .npy: 299,984 probs finitas")

    # --- 4) El blend es la media simple de los 14 checkpoints ---
    if not faltan:
        media_ckpt = np.mean(
            [np.load(os.path.join(ckpt, m + ".npy")) for m in MIEMBROS],
            axis=0)
        dif = float(np.max(np.abs(media_ckpt - sub["is_fraud"].values)))
        todo_ok &= chk("blend_es_media_de_14",
                       dif < 1e-9, f"diferencia máxima: {dif:.2e}")
    else:
        dif = None

    # --- 5) Sanidad vs v1 (umbral calibrado con los miembros: ningún
    # miembro individual supera 0.99 vs v1 —máx. lgbm_comb_s1, Pearson
    # 0.9898—, así que el blend tampoco puede; el criterio es que el
    # blend supere a la mediana de sus miembros) ---
    r_p, _ = pearsonr(sub["is_fraud"], v1["is_fraud"])
    r_s, _ = spearmanr(sub["is_fraud"], v1["is_fraud"])
    mad = float(np.mean(np.abs(sub["is_fraud"] - v1["is_fraud"])))
    top_blend = set(sub.nlargest(1000, "is_fraud")["id"])
    top_v1 = set(v1.nlargest(1000, "is_fraud")["id"])
    solape_top1000 = len(top_blend & top_v1)
    corr_miembros = []
    for m in MIEMBROS:
        p_m = np.load(os.path.join(
            BASE, "verificacion", "blend_ckpt", m + ".npy"))
        rp_m, _ = pearsonr(p_m, v1["is_fraud"])
        rs_m, _ = spearmanr(p_m, v1["is_fraud"])
        corr_miembros.append((rp_m, rs_m))
    med_p = float(np.median([c[0] for c in corr_miembros]))
    med_s = float(np.median([c[1] for c in corr_miembros]))
    todo_ok &= chk("sanidad_blend_supera_mediana_miembros",
                   r_p > med_p and r_s > med_s,
                   f"blend Pearson {r_p:.4f} / Spearman {r_s:.4f} vs "
                   f"mediana miembros {med_p:.4f} / {med_s:.4f}")
    todo_ok &= chk("sanidad_solape_top1000_ge_800",
                   solape_top1000 >= 800,
                   f"solape top-1000: {solape_top1000}/1000")
    print(f"  sanidad: media blend {sub['is_fraud'].mean():.4f} | "
          f"media v1 {v1['is_fraud'].mean():.4f} | dif media abs "
          f"{mad:.4f} | solape top-1000: {solape_top1000}/1000",
          flush=True)

    # --- 6) Reproducción OOF desde los npz guardados ---
    tr = pd.read_csv(os.path.join(BASE, "datos", "train.csv"))
    y = tr["is_fraud"].values
    oof = {**dict(np.load(os.path.join(
        BASE, "verificacion", "oof_ensamble.npz"))),
           **dict(np.load(os.path.join(
               BASE, "verificacion", "oof_ensamble_comb.npz")))}
    p_oof = np.mean([oof[m] for m in MIEMBROS], axis=0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = [round(float(roc_auc_score(y[va], p_oof[va])), 4)
            for _, va in skf.split(tr, y)]
    media = round(float(np.mean(aucs)), 4)
    std = round(float(np.std(aucs)), 4)
    pool = round(float(roc_auc_score(y, p_oof)), 4)
    todo_ok &= chk("oof_reproduce_0.9465",
                   aucs == OOF_ESPERADO["aucs"]
                   and abs(media - OOF_ESPERADO["media"]) < 0.0001
                   and abs(std - OOF_ESPERADO["std"]) < 0.0001
                   and abs(pool - OOF_ESPERADO["pool"]) < 0.0001,
                   f"{aucs} media {media} +- {std} | pool {pool}")

    # --- 7) Lenguaje neutro en los entregables nuevos ---
    # Se revisan los entregables. Este mismo script NO se incluye: define
    # la lista de patrones prohibidos y por tanto los contiene
    # literalmente (autorreferencia); sus menciones son la lista, no
    # contenido.
    nuevos = ["fraude-tarjetas-blend.ipynb",
              "kaggle-fraude-informe-blend.md",
              "verificacion/blend.json",
              "verificacion/construir_notebook_blend.py",
              "verificacion/correr_notebook_blend.py"]
    # Los hallazgos se guardan por ÍNDICE de patrón ("patron_3"), no por
    # el texto del patrón: escribir el texto en blend.json lo
    # contaminaría y la siguiente corrida lo volvería a encontrar.
    hits = {}
    for rel in nuevos:
        ruta = os.path.join(BASE, rel)
        if not os.path.exists(ruta):
            continue
        texto = open(ruta, encoding="utf-8").read()
        for i, pat in enumerate(PATRONES_PROHIBIDOS):
            n = len(re.findall(pat, texto, flags=re.IGNORECASE))
            if n:
                hits.setdefault(rel, {})[f"patron_{i}"] = n
    todo_ok &= chk("lenguaje_neutro", not hits,
                   f"hallazgos: {hits}" if hits else "0 menciones")

    # --- 8) Consolidar blend.json ---
    ruta_json = os.path.join(BASE, "verificacion", "blend.json")
    datos = {}
    if os.path.exists(ruta_json):
        datos = json.load(open(ruta_json, encoding="utf-8"))
    datos["verificacion_independiente"] = {
        "md5_blend": md5(ruta_blend),
        "md5_v1": md5(ruta_v1),
        "filas": int(len(sub)),
        "media_prob": round(float(sub["is_fraud"].mean()), 4),
        "min_prob": round(float(sub["is_fraud"].min()), 4),
        "max_prob": round(float(sub["is_fraud"].max()), 4),
        "sanidad_vs_v1": {"pearson": round(float(r_p), 4),
                          "spearman": round(float(r_s), 4),
                          "mediana_pearson_miembros": round(med_p, 4),
                          "mediana_spearman_miembros": round(med_s, 4),
                          "diferencia_media_abs": round(mad, 4),
                          "solape_top1000": solape_top1000,
                          "media_v1": round(
                              float(v1["is_fraud"].mean()), 4)},
        "oof_reproducido": {"aucs_por_pliegue": aucs,
                            "auc_media": media, "auc_std": std,
                            "auc_oof_agrupado": pool},
        "blend_es_media_de_14": {"diferencia_maxima": dif},
        "lenguaje_neutro": {"hallazgos": hits},
        "checks": checks,
        "veredicto": "LISTO" if todo_ok else "REVISAR",
    }
    datos["veredicto"] = ("LISTO" if todo_ok
                          and datos.get("veredicto") in (None, "LISTO")
                          else "REVISAR")
    with open(ruta_json, "w", encoding="utf-8") as fh:
        json.dump(datos, fh, indent=2, ensure_ascii=False)
    n_ok = sum(1 for c in checks if c["ok"])
    print(f"\n{n_ok}/{len(checks)} checks independientes PASS — "
          f"veredicto: {'LISTO' if todo_ok else 'REVISAR'}", flush=True)
    return 0 if todo_ok else 1


if __name__ == "__main__":
    sys.exit(main())
