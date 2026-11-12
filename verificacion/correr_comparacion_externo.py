#!/usr/bin/env python3
"""Corre la variante comparada con datos externos (original CC0) para S6E11.

- Carga train/test/sample + el externo (datos/credit-card-fraud-detection-2026.csv).
- Corre ft.evaluar_con_externo() (CV 5 pliegues en los 3 modos, validando
  SOLO sobre pliegues de la competencia; el externo solo entra como
  entrenamiento extra).
- Aplica la decisión con números (ft.decidir_externo): una variante con
  externo gana solo si supera al mejor AUC sin externo por más de la std
  entre pliegues de esa variante. Si gana, regenera submission.csv con
  train + externo en el modo ganador y re-verifica 7/7. Si no gana, deja
  el submission actual intacto y documenta el descarte.
- Guarda verificacion/comparacion_externo.json y añade los checks de la
  comparación a verificacion/verificacion-fraude.json (mantiene los 39
  existentes). No envía nada a Kaggle. No modifica los datos.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)
import fraude_tarjetas as ft  # noqa: E402

DATOS = os.path.join(BASE, "datos")
SALIDA = os.path.join(BASE, "salida-fraude")
VERIF = os.path.join(HERE, "verificacion-fraude.json")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _md5(ruta):
    h = hashlib.md5()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def main():
    t0 = time.time()
    train = pd.read_csv(os.path.join(DATOS, "train.csv"))
    test = pd.read_csv(os.path.join(DATOS, "test.csv"))
    sample = pd.read_csv(os.path.join(DATOS, "sample_submission.csv"))
    ext = pd.read_csv(os.path.join(DATOS, "credit-card-fraud-detection-2026.csv"))
    print(f"externo: {ext.shape}, fraude {ext['is_fraud'].mean():.4f}, "
          f"nulos {int(ext.isna().sum().sum())}")

    ruta_sub = os.path.join(SALIDA, "submission.csv")
    md5_antes = _md5(ruta_sub)
    print("submission actual md5:", md5_antes)

    comp = ft.evaluar_con_externo(train, ext)
    print("\n== Comparación con externo (CV 5 pliegues, validación solo "
          "competencia) ==")
    print(f"Mejor sin externo: {comp['mejor_modo_sin_externo']} "
          f"{comp['mejor_sin_externo']:.4f}")
    for modo in ("crudas", "senales", "combinadas"):
        r = comp["por_modo"][modo]
        delta = r["auc_con_externo_media"] - comp["mejor_sin_externo"]
        print(f"{modo:11s} con externo {r['auc_con_externo_media']:.4f} ± "
              f"{r['auc_con_externo_std']:.4f}  sin externo "
              f"{r['auc_sin_externo_media']:.4f} ± {r['auc_sin_externo_std']:.4f}  "
              f"Δ vs mejor sin externo {delta:+.4f}")
    print("Veredicto:", comp["veredicto"])

    ganador = comp["modo_ganador"]
    envio = {"modo_ganador": ganador, "regenerado": False,
             "md5_antes": md5_antes, "md5_despues": md5_antes}
    if ganador:
        print(f"\nEl externo GANA en modo {ganador}: regenerando envío con "
              "train + externo...")
        modelo, enc, ref = ft.entrenar_final_con_externo(train, ext, ganador)
        probs = ft.predecir_test(modelo, enc, ref, test, ganador)
        sub = ft.escribir_submission(test, probs, ruta_sub)
        checks = ft.verificar_submission(sub, sample)
        for c in checks:
            print(("OK  " if c["ok"] else "FAIL"), "-", c["check"], "-",
                  c["detalle"])
        assert all(c["ok"] for c in checks), "submission regenerado FALLA checks"
        envio["regenerado"] = True
        envio["checks_ok"] = sum(1 for c in checks if c["ok"])
        envio["checks_total"] = len(checks)
        envio["media_prob"] = round(float(sub["is_fraud"].mean()), 4)
        envio["md5_despues"] = _md5(ruta_sub)
        print("submission regenerado md5:", envio["md5_despues"],
              "| media prob:", envio["media_prob"])
    else:
        print("\nEl externo NO gana: submission actual intacto.")
        envio["md5_despues"] = _md5(ruta_sub)
        assert envio["md5_despues"] == md5_antes, "submission cambió sin ganar"

    with open(os.path.join(HERE, "comparacion_externo.json"), "w",
              encoding="utf-8") as fh:
        json.dump(_jsonable({"comparacion": comp, "envio": envio,
                             "tiempo_s": round(time.time() - t0, 1)}),
                  fh, indent=1, ensure_ascii=False)
    print("comparacion_externo.json escrito.")

    # Añadir checks de la comparación a verificacion-fraude.json.
    with open(VERIF, encoding="utf-8") as fh:
        ver = json.load(fh)
    nuevos = []
    def _check(nombre, ok, detalle=""):
        nuevos.append({"check": nombre, "ok": bool(ok), "detalle": str(detalle)})
        print(("OK   " if ok else "FAIL "), "-", nombre, "-", detalle)
    _check("externo 20000x26", ext.shape == (20000, 26), ext.shape)
    _check("externo cero nulos", int(ext.isna().sum().sum()) == 0,
           int(ext.isna().sum().sum()))
    _check("externo tasa fraude 0.0169 (339)",
           abs(float(ext["is_fraud"].mean()) - 0.0169) < 0.0001 and
           int(ext["is_fraud"].sum()) == 339,
           f"{float(ext['is_fraud'].mean()):.4f} / {int(ext['is_fraud'].sum())}")
    _check("externo cambio distribución: monto medio 132.42 vs 79.89",
           abs(comp["medias_numericas_externo"]["amount_usd"] - 132.42) < 0.01 and
           abs(comp["medias_numericas_competencia"]["amount_usd"] - 79.89) < 0.01,
           f"{comp['medias_numericas_externo']['amount_usd']} vs "
           f"{comp['medias_numericas_competencia']['amount_usd']}")
    _check("externo sin auth 0.0444 vs train 0.0644",
           abs(comp["tasa_por_auth_externo"]["No Authentication"] - 0.0444) < 0.0001 and
           abs(comp["tasa_por_auth_competencia"]["No Authentication"] - 0.0644) < 0.0001,
           "0.0444 vs 0.0644")
    _check("externo intento IA 0.0907 vs train 0.1522",
           abs(comp["tasa_intento_ia_externo"] - 0.0907) < 0.0001 and
           abs(comp["tasa_intento_ia_competencia"] - 0.1522) < 0.0001,
           "0.0907 vs 0.1522")
    for modo in ("crudas", "senales", "combinadas"):
        r = comp["por_modo"][modo]
        _check(f"externo CV {modo}: con {r['auc_con_externo_media']:.4f} ± "
               f"{r['auc_con_externo_std']:.4f} vs sin "
               f"{r['auc_sin_externo_media']:.4f} ± {r['auc_sin_externo_std']:.4f}",
               len(r["aucs_con"]) == 5 and len(r["aucs_sin"]) == 5,
               f"con {r['aucs_con']} / sin {r['aucs_sin']}")
    _check("externo decisión documentada (modo_ganador + veredicto)",
           ("modo_ganador" in comp and "veredicto" in comp and
            comp["veredicto"].startswith(("AYUDA", "NO AYUDA"))),
           f"ganador={ganador} / {comp['veredicto'][:60]}...")
    if ganador:
        _check(f"externo envío regenerado modo {ganador} 7/7 checks OK",
               envio["regenerado"] and envio["checks_ok"] == 7,
               f"md5 {envio['md5_despues'][:8]} media {envio['media_prob']}")
    else:
        _check("externo descartado: submission intacto (md5 igual)",
               envio["md5_despues"] == md5_antes, md5_antes[:12])
    ver["checks"].extend(nuevos)
    ver["n_checks"] = len(ver["checks"])
    ver["ok"] = sum(1 for c in ver["checks"] if c["ok"])
    ver["fallos"] = [c["check"] for c in ver["checks"] if not c["ok"]]
    with open(VERIF, "w", encoding="utf-8") as fh:
        json.dump(ver, fh, indent=1, ensure_ascii=False)
    print(f"verificacion-fraude.json: {ver['ok']}/{ver['n_checks']} OK, "
          f"fallos: {ver['fallos']}")
    print(f"TIEMPO TOTAL: {time.time() - t0:.0f}s")
    assert not ver["fallos"], "hay checks con fallo"
    print("COMPARACIÓN COMPLETADA")


if __name__ == "__main__":
    main()
