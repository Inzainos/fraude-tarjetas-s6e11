#!/usr/bin/env python3
"""
ASI por tipo de tarjeta — Kaggle Playground Series S6E11 (fraude con tarjeta).

Calcula el ASI (indice de autonomia) por cada tipo de tarjeta con proxies
DOCUMENTADOS, tomados de la definicion del repo (ASI = dH * alfa / F):

  dH  : entropia de Shannon normalizada [0,1] de la distribucion de
        merchant_category DENTRO de cada tipo de tarjeta (diversidad de
        contextos que enfrenta la tarjeta). H = -sum(p * log2(p)),
        normalizada por log2(K) con K = 12 categorias.
  alfa: proporcion de transacciones con autenticacion iniciada por el
        usuario (Biometric / PIN / OTP) sobre el total de ese tipo.
        3D Secure NO se cuenta como iniciada por el usuario (es un reto
        del protocolo / del comercio) y "No Authentication" tampoco.
  alfa2 (sensibilidad): proporcion de transacciones en comercios conocidos
        (is_new_merchant == False) sobre el total de ese tipo.
  F   : coeficiente de variacion (std / media) de amount_usd dentro de
        cada tipo de tarjeta (std muestral, ddof=1).

  ASI  = dH * alfa  / F
  ASI2 = dH * alfa2 / F   (sensibilidad con alfa2)

ADVERTENCIAS (tambien en el informe, seccion 11):
  1. El umbral ASI > 1 fue calibrado en el cohorte Elite de HackerEarth
     2026; NO se transfiere mecanicamente a fraude. Aqui el ASI es
     descriptivo/comparativo, no un veredicto de soberania.
  2. Como feature del modelo, el ASI por tipo de tarjeta son solo 5
     valores constantes (uno por tipo) y no aporta nada mas alla de lo
     que card_type ya da al modelo; su valor es interpretativo, no de AUC.
     No entra al submission salvo que la CV demuestre lo contrario bajo
     la regla de decision (superar 0.9449 por mas de la std ~0.0035).
  3. El proxy de alfa es interpretativo: en el repo alfa es eventos
     iniciados por el usuario vs impuestos por el sistema; aqui no hay
     log de eventos, asi que se usa el metodo de autenticacion como proxy.

Solo lectura de datos/train.csv. No toca el submission ni la fase 1.
"""
import json
import math
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(BASE, "datos", "train.csv")
OUT_JSON = os.path.join(BASE, "verificacion", "asi_por_tarjeta.json")

AUTH_USUARIO = ["Biometric", "PIN", "OTP"]  # autenticacion iniciada por el usuario
K_CATEGORIAS = 12  # merchant_category distintas en train (verificado)


def entropia_normalizada(conteos):
    """Entropia de Shannon normalizada [0,1]: H / log2(K)."""
    total = float(sum(conteos))
    h = 0.0
    for c in conteos:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h / math.log2(K_CATEGORIAS), h


def main():
    df = pd.read_csv(
        TRAIN,
        usecols=["card_type", "auth_method", "merchant_category",
                 "is_new_merchant", "amount_usd", "is_fraud"],
    )
    assert len(df) == 699960, len(df)
    assert df["merchant_category"].nunique() == K_CATEGORIAS
    assert set(df["auth_method"].unique()) == {
        "Biometric", "PIN", "OTP", "3D Secure", "No Authentication"}

    filas = []
    detalle = {}
    for tipo, g in df.groupby("card_type"):
        n = len(g)
        conteos_cat = g["merchant_category"].value_counts()
        dh, h_bits = entropia_normalizada(
            [int(conteos_cat.get(c, 0)) for c in sorted(df["merchant_category"].unique())])
        n_auth_usuario = int(g["auth_method"].isin(AUTH_USUARIO).sum())
        alfa = n_auth_usuario / n
        n_conocido = int((~g["is_new_merchant"]).sum())
        alfa2 = n_conocido / n
        media = float(g["amount_usd"].mean())
        std = float(g["amount_usd"].std(ddof=1))
        f = std / media
        asi = dh * alfa / f
        asi2 = dh * alfa2 / f
        tasa_fraude = float(g["is_fraud"].mean())
        filas.append({
            "card_type": tipo, "n": n,
            "delta_H": round(dh, 6), "H_bits": round(h_bits, 6),
            "alfa": round(alfa, 6), "alfa2": round(alfa2, 6),
            "amount_media": round(media, 4), "amount_std": round(std, 4),
            "F": round(f, 6),
            "ASI": round(asi, 6), "ASI2": round(asi2, 6),
            "tasa_fraude": round(tasa_fraude, 6),
        })
        detalle[tipo] = {
            "n": n,
            "merchant_category_conteos": {str(k): int(v) for k, v in conteos_cat.items()},
            "auth_method_conteos": {str(k): int(v) for k, v in g["auth_method"].value_counts().items()},
            "n_auth_usuario_biometric_pin_otp": n_auth_usuario,
            "n_comercio_conocido_is_new_merchant_False": n_conocido,
            "amount_media": media, "amount_std_ddof1": std,
            "tasa_fraude": tasa_fraude,
        }

    # Fila global de referencia (todo train).
    conteos_cat = df["merchant_category"].value_counts()
    dh_g, h_g = entropia_normalizada(
        [int(conteos_cat.get(c, 0)) for c in sorted(df["merchant_category"].unique())])
    alfa_g = float(df["auth_method"].isin(AUTH_USUARIO).mean())
    alfa2_g = float((~df["is_new_merchant"]).mean())
    f_g = float(df["amount_usd"].std(ddof=1) / df["amount_usd"].mean())
    filas.append({
        "card_type": "TODAS", "n": len(df),
        "delta_H": round(dh_g, 6), "H_bits": round(h_g, 6),
        "alfa": round(alfa_g, 6), "alfa2": round(alfa2_g, 6),
        "amount_media": round(float(df["amount_usd"].mean()), 4),
        "amount_std": round(float(df["amount_usd"].std(ddof=1)), 4),
        "F": round(f_g, 6),
        "ASI": round(dh_g * alfa_g / f_g, 6),
        "ASI2": round(dh_g * alfa2_g / f_g, 6),
        "tasa_fraude": round(float(df["is_fraud"].mean()), 6),
    })

    resultado = {
        "definicion": "ASI = delta_H * alfa / F (definicion del repo)",
        "proxies": {
            "delta_H": "Entropia de Shannon normalizada [0,1] de merchant_category dentro de cada tipo de tarjeta; H=-sum(p*log2(p)), normalizada por log2(12).",
            "alfa": "Proporcion de transacciones con autenticacion iniciada por el usuario (Biometric/PIN/OTP) sobre el total del tipo. 3D Secure NO se cuenta (reto del protocolo/comercio) ni No Authentication.",
            "alfa2_sensibilidad": "Proporcion de transacciones en comercios conocidos (is_new_merchant=False) sobre el total del tipo.",
            "F": "Coeficiente de variacion std/media de amount_usd dentro de cada tipo (std muestral ddof=1).",
        },
        "advertencias": [
            "El umbral ASI > 1 fue calibrado en el cohorte Elite de HackerEarth 2026; NO se transfiere mecanicamente a fraude. Aqui el ASI es descriptivo/comparativo, no un veredicto de soberania.",
            "Como feature del modelo, el ASI por tipo de tarjeta son solo 5 valores constantes (uno por tipo) y no aporta nada mas alla de lo que card_type ya da al modelo; su valor es interpretativo, no de AUC. No entra al submission salvo que la CV demuestre lo contrario bajo la regla de decision (superar 0.9449 por mas de la std ~0.0035).",
            "El proxy de alfa es interpretativo: en el repo alfa es eventos iniciados por el usuario vs impuestos por el sistema; aqui no hay log de eventos, asi que se usa el metodo de autenticacion como proxy.",
        ],
        "n_train": len(df),
        "k_merchant_category": K_CATEGORIAS,
        "tabla": filas,
        "detalle_por_tipo": detalle,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(resultado, fh, indent=1, ensure_ascii=False)
    print(json.dumps(filas, indent=1, ensure_ascii=False))
    print("JSON:", OUT_JSON)


if __name__ == "__main__":
    main()
