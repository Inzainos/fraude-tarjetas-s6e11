#!/usr/bin/env python3
"""Entrenamiento final del blend ponderado de v2 (segundo final propuesto).

Blend de PROBABILIDADES, pesos fijos tomados de medir_stacking_blend.py
(blend_pond_global, columnas [glm_lgb4, glm, lgb4_te, lgb4_raw]):
    0.7 * glm_lgb4 + 0.1 * glm + 0.2 * lgb4_te + 0.0 * lgb4_raw
CV medido de ese blend: 0.95072 anidado (Δ +0.00006 vs glm_lgb4: empate
técnico); su valor es de piso/cobertura, no de AUC adicional.

Reutiliza ensamble_v2.ajustar_predecir (el mismo código validado que produjo
el envío de 0.95112): entrena sobre TODO train y predice test. No hace CV.
Los pesos NO se reajustan aquí.

Controles que imprime y deja en el log:
  - media de probabilidad (debe rondar la tasa de fraude ~0.0165)
  - correlación de Spearman con el glm_lgb4 puro de esta misma corrida
  - si se pasa --referencia (p. ej. salida-fraude/submission_consultor_v2.csv),
    correlación de Spearman con ese envío (debe ser ~0.99+)
  - checks de formato contra sample_submission.csv si se pasa --sample

Uso:
    python3 final_blend_v2.py --datos datos --salida salida-fraude \
        --referencia salida-fraude/submission_consultor_v2.csv \
        --sample datos/sample_submission.csv

Escribe <salida>/submission_blend_v2.csv y verificacion/final-blend-v2.log.
Nunca sobrescribe otro envío. Nada se envía a Kaggle: eso lo decide Elán.
Requiere ensamble_v2.py y fraude_tarjetas.py en el mismo directorio.
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fraude_tarjetas as ft  # noqa: E402
import ensamble_v2 as v2  # noqa: E402

PESOS = {"glm_lgb4": 0.7, "glm": 0.1, "lgb4_te": 0.2, "lgb4_raw": 0.0}
log = logging.getLogger("final_blend_v2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default=os.path.join(HERE, "datos"))
    ap.add_argument("--salida", default=os.path.join(HERE, "salida-fraude"))
    ap.add_argument("--log", default=os.path.join(HERE, "verificacion"))
    ap.add_argument("--max-filas", type=int, default=0)
    ap.add_argument("--referencia", default="")
    ap.add_argument("--sample", default="")
    a = ap.parse_args()
    os.makedirs(a.salida, exist_ok=True)
    os.makedirs(a.log, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(os.path.join(a.log, "final-blend-v2.log"),
                                      mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])
    # ensamble_v2 escribe su detalle de tiempos en su propio logger
    logging.getLogger("ensamble_v2").setLevel(logging.INFO)
    t0 = time.time()
    tr = pd.read_csv(os.path.join(a.datos, "train.csv"))
    if a.max_filas:
        tr = tr.sample(a.max_filas, random_state=0).reset_index(drop=True)
    te = pd.read_csv(os.path.join(a.datos, "test.csv"))
    y = tr[ft.TARGET].values.astype(float)
    log.info("train %s test %s fraude %.5f", tr.shape, te.shape, y.mean())
    log.info("pesos del blend: %s (suma %.2f)", PESOS, sum(PESOS.values()))

    p = v2.ajustar_predecir(tr, te.assign(**{ft.TARGET: 0}), y)
    blend = sum(w * p[n] for n, w in PESOS.items())
    sub = pd.DataFrame({"id": te["id"].values, "is_fraud": blend})
    ruta = os.path.join(a.salida, "submission_blend_v2.csv")
    sub.to_csv(ruta, index=False)
    log.info("escrito %s (%d filas, media %.5f, min %.3g, max %.5f)", ruta,
             len(sub), sub["is_fraud"].mean(), sub["is_fraud"].min(),
             sub["is_fraud"].max())

    rho = spearmanr(blend, p["glm_lgb4"]).correlation
    log.info("Spearman blend vs glm_lgb4 (misma corrida): %.5f", rho)
    if a.referencia:
        ref = pd.read_csv(a.referencia)
        if (ref["id"].values == sub["id"].values).all():
            rr = spearmanr(blend, ref["is_fraud"].values).correlation
            log.info("Spearman blend vs referencia %s: %.5f", a.referencia, rr)
            rg = spearmanr(p["glm_lgb4"], ref["is_fraud"].values).correlation
            log.info("Spearman glm_lgb4 reentrenado vs referencia: %.5f "
                     "(~1.0 = reproduce el envío de 0.95112)", rg)
        else:
            log.info("AVISO: los ids de la referencia no coinciden con test")
    if a.sample:
        s = pd.read_csv(a.sample)
        checks = ft.verificar_submission(sub, s)
        ok = sum(c["ok"] for c in checks)
        log.info("checks de formato contra sample_submission: %d/%d OK %s", ok,
                 len(checks), [c["check"] for c in checks if not c["ok"]])
    log.info("listo en %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
