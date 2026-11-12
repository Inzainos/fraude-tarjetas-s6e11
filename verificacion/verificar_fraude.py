#!/usr/bin/env python3
"""Verificación independiente de la solución de fraude con tarjeta (S6E11).

Re-calcula desde los CSV (sin usar los resultados guardados) y escribe
verificacion-fraude.json con N checks. Criterio: todo check debe ser
True; los números de referencia son los medidos y re-medidos aquí
(no se dan por buenos sin re-medir).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)
import fraude_tarjetas as m  # noqa: E402

SALIDA = os.path.join(BASE, "salida-fraude")
CHECKS = []


def check(nombre, ok, detalle=""):
    CHECKS.append({"check": nombre, "ok": bool(ok), "detalle": str(detalle)})
    print(("OK   " if ok else "FAIL "), "-", nombre, "-", detalle)
    return bool(ok)


def main():
    tr = pd.read_csv(os.path.join(BASE, "datos", "train.csv"))
    te = pd.read_csv(os.path.join(BASE, "datos", "test.csv"))
    ss = pd.read_csv(os.path.join(BASE, "datos", "sample_submission.csv"))

    # Datos
    check("train 699960x26", tr.shape == (699960, 26), tr.shape)
    check("test 299984x25", te.shape == (299984, 25), te.shape)
    check("sample 299984x2", ss.shape == (299984, 2), ss.shape)
    check("tasa fraude 0.0167",
          abs(tr["is_fraud"].mean() - 0.0167) < 0.0001,
          f"{tr['is_fraud'].mean():.5f}")
    check("fraudes 11689", int(tr["is_fraud"].sum()) == 11689,
          int(tr["is_fraud"].sum()))
    check("cero nulos", int(tr.isna().sum().sum()) == 0
          and int(te.isna().sum().sum()) == 0,
          f"train {int(tr.isna().sum().sum())} test {int(te.isna().sum().sum())}")

    # Señales medidas, re-medidas aquí
    t = tr.groupby("auth_method")["is_fraud"].mean()
    check("Sin Autenticación 6.44%",
          abs(t["No Authentication"] - 0.0644) < 0.0005,
          f"{t['No Authentication']:.4f}")
    check("Biométrico 0.50%",
          abs(t["Biometric"] - 0.0050) < 0.0005, f"{t['Biometric']:.4f}")
    check("intento IA 15.22%",
          abs(tr.loc[tr["is_ai_generated_scam_attempt"], "is_fraud"].mean()
              - 0.1522) < 0.0005,
          f"{tr.loc[tr['is_ai_generated_scam_attempt'], 'is_fraud'].mean():.4f}")
    tc = tr.groupby("merchant_category")["is_fraud"].mean()
    check("Gift Cards 7.30%", abs(tc["Gift Cards"] - 0.0730) < 0.0005,
          f"{tc['Gift Cards']:.4f}")
    check("Crypto 6.53%", abs(tc["Crypto Exchange"] - 0.0653) < 0.0005,
          f"{tc['Crypto Exchange']:.4f}")
    check("Gaming 5.51%", abs(tc["Gaming"] - 0.0551) < 0.0005,
          f"{tc['Gaming']:.4f}")
    check("Groceries 1.11%", abs(tc["Groceries"] - 0.0111) < 0.0005,
          f"{tc['Groceries']:.4f}")

    # Conteo de señales
    t_senales, res = m.tabla_conteo_senales(tr)
    tasas = t_senales["tasa"].values
    check("conteo de señales: tasa no decrece (satura en 100% con 7-8 señales)",
          bool(all(tasas[i] <= tasas[i + 1] for i in range(len(tasas) - 1))),
          [round(x, 4) for x in tasas])
    check("umbral >= 5 señales: n=549", res["n_umbral"] == 549,
          res["n_umbral"])
    check("umbral >= 5 señales: tasa 87.61%",
          abs(res["tasa_umbral"] - 0.8761) < 0.0005,
          f"{res['tasa_umbral']:.4f}")
    check("umbral >= 5 señales: captura 4.1% de fraudes",
          abs(res["pct_fraudes_capturados"] - 4.1) < 0.1,
          f"{res['pct_fraudes_capturados']:.1f}%")
    check("0 señales: tasa 0.19%",
          abs(tasas[0] - 0.0019) < 0.0005, f"{tasas[0]:.4f}")

    # Cuota de fraude
    t_cat = m.tabla_cuota_fraude(tr, "merchant_category")
    rg = t_cat.set_index("merchant_category")["cuota_fraude"]
    check("cuota Gift Cards 0.0787", abs(rg["Gift Cards"] - 0.0787) < 0.0001,
          f"{rg['Gift Cards']:.4f}")
    check("cuota Crypto 0.0698", abs(rg["Crypto Exchange"] - 0.0698) < 0.0001,
          f"{rg['Crypto Exchange']:.4f}")
    check("cuota Groceries 0.0112", abs(rg["Groceries"] - 0.0112) < 0.0001,
          f"{rg['Groceries']:.4f}")
    n0 = set(t_cat.loc[t_cat["nivel_concentracion"] == "Nivel 0",
                       "merchant_category"])
    check("Nivel 0 = 4 (Online Retail, Groceries, Electronics, Gaming)",
          n0 == {"Online Retail", "Groceries", "Electronics", "Gaming"}, n0)
    check("Electronics 46.9% del monto de fraude",
          abs(float(t_cat.loc[t_cat["merchant_category"] == "Electronics",
                              "cuota_fraude_usd_pct"].iloc[0]) - 46.9) < 0.1,
          f"{float(t_cat.loc[t_cat['merchant_category'] == 'Electronics', 'cuota_fraude_usd_pct'].iloc[0]):.2f}%")

    # Serie temporal insuficiente
    diag = m.diagnostico_temporal(tr)
    check("temporal: datos_insuficientes",
          diag["estado"] == "datos_insuficientes", diag["estado"])
    check("id no temporal (|corr| < 0.01)",
          abs(diag["evidencia"]["corr_id_hora"]) < 0.01
          and abs(diag["evidencia"]["corr_id_fraude"]) < 0.01,
          diag["evidencia"])

    # Distancia plana
    ev = m.calcular_senales_riesgo(tr)
    p90 = tr["distance_from_home_km"].quantile(0.9)
    check("distancia plana (p90 1.72% vs resto 1.66%)",
          abs(tr.loc[tr["distance_from_home_km"] >= p90, "is_fraud"].mean()
              - 0.0172) < 0.001
          and abs(tr.loc[tr["distance_from_home_km"] < p90, "is_fraud"].mean()
                  - 0.0166) < 0.001,
          f"p90={p90:.2f} km, "
          f"alta={tr.loc[tr['distance_from_home_km'] >= p90, 'is_fraud'].mean():.4f}, "
          f"baja={tr.loc[tr['distance_from_home_km'] < p90, 'is_fraud'].mean():.4f}")

    # Módulo y notebook
    check("módulo versión 1.0.0", m.__version__ == "1.0.0", m.__version__)
    nb = json.load(open(os.path.join(BASE, "fraude-tarjetas.ipynb"),
                        encoding="utf-8"))
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    check("notebook 12 celdas (6 código + 6 markdown)",
          len(nb["cells"]) == 12 and n_code == 6 and n_md == 6,
          f"{len(nb['cells'])} celdas ({n_code} código, {n_md} markdown)")

    # Envío
    ruta_sub = os.path.join(SALIDA, "submission.csv")
    if os.path.exists(ruta_sub):
        sub = pd.read_csv(ruta_sub)
        for c in m.verificar_submission(sub, ss):
            check(f"submission: {c['check']}", c["ok"], c["detalle"])
    else:
        check("submission.csv existe", False, "no se encontró")

    # Informe y figuras
    check("informe-fraude.md existe",
          os.path.exists(os.path.join(SALIDA, "informe-fraude.md")))
    for f in ("fig-conteo-senales.png", "fig-cuota-fraude-categoria.png",
              "fig-auth.png"):
        check(f"figura {f} existe",
              os.path.exists(os.path.join(SALIDA, f)))

    ok = sum(1 for c in CHECKS if c["ok"])
    resumen = {"n_checks": len(CHECKS), "ok": ok,
               "fallos": [c["check"] for c in CHECKS if not c["ok"]],
               "checks": CHECKS}
    with open(os.path.join(HERE, "verificacion-fraude.json"), "w",
              encoding="utf-8") as fh:
        json.dump(resumen, fh, indent=2, ensure_ascii=False)
    print(f"\n{ok}/{len(CHECKS)} checks OK")
    return 0 if ok == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
