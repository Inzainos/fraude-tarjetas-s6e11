#!/usr/bin/env python3
"""Stacking de glm_lgb4 (propio, v2) con el GPT-2 target-first público.

Miembros:
  propio : glm_lgb4. OOF = verificacion/oof_ensamble_v2.npz[glm_lgb4];
           test = salida-fraude/submission_consultor_v2.csv (refit completo,
           el envío de público 0.95112).
  público: goodpjw2008/s6e11-target-first-gpt2-seed-2026 (CC0):
           gpt2_s2026_oof_5fold.npy (train, log-odds, 699,960 filas) y
           gpt2_s2026_test_5fold.npy (test, 299,984 filas).

Stacker (pre-registrado, no se ajusta al resultado):
  - rank-logit suave por miembro: log((r+e)/(1-r+e)), e = 0.5/k, k = nº de
    miembros, r = rango/n (rangos del OOF sobre todo el OOF; rangos del test
    sobre todo el test).
  - regresión logística (C=0.01 por defecto, solver newton-cholesky).
  - CV anidada del stacker con StratifiedKFold(5, shuffle, 42+...=7), una
    partición DISTINTA de la de los miembros (42). El delta pareado se mide
    por los 5 pliegues ORIGINALES (semilla 42) contra glm_lgb4 solo.
  - referencia: blend ponderado de rank-logits con peso w del GPT-2 elegido
    por pliegue en la partición de entrenamiento (rejilla 0..0.60).

Regla de adopción: delta pareado > 0 en 5/5 pliegues originales, media
>= +0.0001 y media > 2x la std del delta. Si no se cumple, el envío de
0.95112 queda como está.

Verificaciones de alineación (se detiene si fallan las duras):
  - longitudes de OOF/test contra train/test;
  - AUC del OOF del GPT-2 por pliegue original (>= 0.90 o aborta: si el
    orden no coincide con train.csv el AUC cae a ~0.5);
  - Spearman OOF propio-vs-GPT-2 y test propio-vs-GPT-2 (aviso si difieren
    en más de 0.02: señal de desalineación del test);
  - md5 de los archivos de entrada en el log.

Uso (en la Kali, desde la raíz del repo):
    python3 stack_publico.py \
        --gpt-oof RUTA/gpt2_s2026_oof_5fold.npy \
        --gpt-test RUTA/gpt2_s2026_test_5fold.npy
    # con --final escribe salida-fraude/submission_stack_gpt2.csv
Nada se envía a Kaggle: eso lo decide Elán. Log: <salida>/stack-publico.log.
"""
import argparse
import hashlib
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fraude_tarjetas as ft  # noqa: E402

SEED_MIEMBROS = 42      # partición con la que se generaron los OOF
SEED_STACK = 7          # partición distinta para evaluar el stacker
log = logging.getLogger("stack_publico")


def md5(ruta):
    h = hashlib.md5()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def auc(y, p):
    return float(roc_auc_score(y, p))


def rank_logit(x, k):
    r = rankdata(x) / len(x)
    e = 0.5 / k
    return np.log((r + e) / (1.0 - r + e))


def cargar_npy(ruta):
    v = np.load(ruta)
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 2 and 1 in v.shape:
        v = v.reshape(-1)
    if v.ndim != 1:
        raise SystemExit(f"{ruta}: forma {v.shape}; se esperaba un vector 1D")
    return v


def stacker(C):
    return LogisticRegression(C=C, solver="newton-cholesky", max_iter=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.path.join(HERE, "datos", "train.csv"))
    ap.add_argument("--test", default=os.path.join(HERE, "datos", "test.csv"))
    ap.add_argument("--sample", default="")
    ap.add_argument("--oof-v2", default=os.path.join(HERE, "verificacion",
                                                     "oof_ensamble_v2.npz"))
    ap.add_argument("--oof-key", default="glm_lgb4")
    ap.add_argument("--test-v2", default=os.path.join(
        HERE, "salida-fraude", "submission_consultor_v2.csv"))
    ap.add_argument("--gpt-oof", required=True)
    ap.add_argument("--gpt-test", required=True)
    ap.add_argument("--salida", default=os.path.join(HERE, "verificacion"))
    ap.add_argument("--salida-sub", default=os.path.join(HERE, "salida-fraude"))
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.salida, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(os.path.join(a.salida, "stack-publico.log"),
                                      mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])
    t0 = time.time()

    for ruta in (a.train, a.test, a.oof_v2, a.test_v2, a.gpt_oof, a.gpt_test):
        if not os.path.exists(ruta):
            raise SystemExit(f"falta el archivo: {ruta}")
        log.info("md5 %s  %s", md5(ruta), ruta)

    tr = pd.read_csv(a.train, usecols=["id", ft.TARGET])
    te_ids = pd.read_csv(a.test, usecols=["id"])["id"].values
    y = tr[ft.TARGET].values.astype(int)
    own_oof = np.load(a.oof_v2)[a.oof_key].astype(np.float64)
    sub_v2 = pd.read_csv(a.test_v2)
    gpt_oof = cargar_npy(a.gpt_oof)
    gpt_te = cargar_npy(a.gpt_test)
    log.info("train %d | test %d | fraude %.5f", len(y), len(te_ids), y.mean())

    # --- verificaciones duras de longitud e ids ---------------------------
    if len(own_oof) != len(y) or len(gpt_oof) != len(y):
        raise SystemExit(f"longitud OOF distinta de train: propio {len(own_oof)}, "
                         f"gpt {len(gpt_oof)}, train {len(y)}")
    if len(gpt_te) != len(te_ids) or len(sub_v2) != len(te_ids):
        raise SystemExit(f"longitud test distinta: gpt {len(gpt_te)}, "
                         f"propio {len(sub_v2)}, test {len(te_ids)}")
    if not (sub_v2["id"].values == te_ids).all():
        raise SystemExit("los ids de submission_consultor_v2.csv no siguen el orden de test.csv")
    own_te = sub_v2["is_fraud"].values.astype(np.float64)

    # --- pliegues originales (los de los miembros) ------------------------
    skf0 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED_MIEMBROS)
    folds0 = [b for _, b in skf0.split(np.zeros(len(y)), y)]
    a_own = [auc(y[b], own_oof[b]) for b in folds0]
    a_gpt = [auc(y[b], gpt_oof[b]) for b in folds0]
    log.info("AUC por pliegue original  propio: %s  media %.5f",
             [round(x, 5) for x in a_own], np.mean(a_own))
    log.info("AUC por pliegue original  GPT-2 : %s  media %.5f",
             [round(x, 5) for x in a_gpt], np.mean(a_gpt))
    if np.mean(a_gpt) < 0.90:
        raise SystemExit("el OOF del GPT-2 tiene AUC < 0.90 contra train: casi seguro "
                         "el orden NO coincide con train.csv. Abortando.")
    rho_oof = spearmanr(own_oof, gpt_oof).correlation
    rho_te = spearmanr(own_te, gpt_te).correlation
    log.info("Spearman propio-vs-GPT-2: OOF %.4f | test %.4f", rho_oof, rho_te)
    if abs(rho_oof - rho_te) > 0.02:
        log.info("AVISO: la correlación del test difiere de la del OOF en más de "
                 "0.02; revisa el orden/alineación de los archivos de test.")

    # --- rank-logits ------------------------------------------------------
    k = 2
    Z = np.column_stack([rank_logit(own_oof, k), rank_logit(gpt_oof, k)])
    Zt = np.column_stack([rank_logit(own_te, k), rank_logit(gpt_te, k)])

    # --- CV anidada del stacker (partición distinta) ----------------------
    skf1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED_STACK)
    oof_lr = np.zeros(len(y))
    oof_bl = np.zeros(len(y))
    pesos_w = []
    rejilla = np.round(np.arange(0.0, 0.61, 0.05), 2)
    for i, (tr_i, va_i) in enumerate(skf1.split(Z, y)):
        m = stacker(a.C)
        m.fit(Z[tr_i], y[tr_i])
        oof_lr[va_i] = m.decision_function(Z[va_i])
        mejor = max(rejilla, key=lambda w: auc(y[tr_i], (1 - w) * Z[tr_i, 0] + w * Z[tr_i, 1]))
        pesos_w.append(float(mejor))
        oof_bl[va_i] = (1 - mejor) * Z[va_i, 0] + mejor * Z[va_i, 1]
        log.info("  stacker pliegue %d/5: coef=%s | w_gpt (blend)=%.2f",
                 i + 1, np.round(m.coef_[0], 4).tolist(), mejor)

    res = {"protocolo": "stacker anidado, partición semilla 7; delta pareado por "
                        "pliegues originales (semilla 42) vs glm_lgb4 solo",
           "C": a.C, "rho_oof": round(float(rho_oof), 5), "rho_test": round(float(rho_te), 5),
           "modelos": {}}
    base = np.array(a_own)
    for nombre, oof in (("glm_lgb4_solo", own_oof), ("gpt2_solo", gpt_oof),
                        ("stack_logistico", oof_lr), ("blend_ponderado", oof_bl)):
        aucs = np.array([auc(y[b], oof[b]) for b in folds0])
        d = aucs - base
        gana = bool(nombre in ("stack_logistico", "blend_ponderado")
                    and (d > 0).all() and d.mean() >= 0.0001 and d.mean() > 2 * d.std())
        res["modelos"][nombre] = {
            "aucs_por_pliegue": [round(float(x), 5) for x in aucs],
            "auc_media": round(float(aucs.mean()), 5),
            "auc_oof_agrupado": round(auc(y, oof), 5),
            "delta_pareado_vs_glm_lgb4": round(float(d.mean()), 5),
            "delta_std": round(float(d.std()), 5),
            "positivo_en_pliegues": int((d > 0).sum()),
            "cumple_regla_adopcion": gana}
        log.info("%-16s media %.5f | Δ pareado %+.5f ± %.5f | +%d/5 | adopta=%s",
                 nombre, aucs.mean(), d.mean(), d.std(), int((d > 0).sum()), gana)
    res["pesos_w_gpt_por_pliegue"] = pesos_w

    # --- ajuste final sobre todo el OOF y escritura -----------------------
    m_final = stacker(a.C)
    m_final.fit(Z, y)
    res["coef_final"] = [round(float(c), 5) for c in m_final.coef_[0]]
    log.info("stacker final: coef %s, intercepto %.4f", res["coef_final"],
             float(m_final.intercept_[0]))
    if a.final:
        p = 1.0 / (1.0 + np.exp(-m_final.decision_function(Zt)))
        os.makedirs(a.salida_sub, exist_ok=True)
        ruta = os.path.join(a.salida_sub, "submission_stack_gpt2.csv")
        sub = pd.DataFrame({"id": te_ids, "is_fraud": p})
        sub.to_csv(ruta, index=False)
        log.info("escrito %s (%d filas, media %.5f)", ruta, len(sub), p.mean())
        rr = spearmanr(p, own_te).correlation
        log.info("Spearman submission nuevo vs envío de 0.95112: %.5f", rr)
        res["spearman_vs_v2_test"] = round(float(rr), 5)
        if a.sample:
            checks = ft.verificar_submission(sub, pd.read_csv(a.sample))
            log.info("checks de formato: %d/%d OK %s", sum(c["ok"] for c in checks),
                     len(checks), [c["check"] for c in checks if not c["ok"]])
    res["tiempo_s"] = round(time.time() - t0, 1)
    with open(os.path.join(a.salida, "stack_publico.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    log.info("listo en %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
