#!/usr/bin/env python3
"""Adaptador para fraude con tarjeta — competencia Kaggle Playground
Series S6E11, "Predicting Credit Card Fraud".

Métrica: ROC AUC. Envío: CSV id,is_fraud (probabilidades 0-1), una
fila por cada id de test. Cierre: 30 nov 2026, 11:59 PM UTC.

Uso:
    python3 fraude_tarjetas.py            # corre todo y deja salida-fraude/
    from fraude_tarjetas import ejecutar  # en notebook o script

El módulo es el código de entrenamiento + inferencia de la solución
(la competencia pide entregarlo si se gana, Sec. 8.3). Todo lo que
afirma está medido sobre train.csv (699,960 filas); nada inventado.

MAPEO DE FEATURES — FRAUDE CON TARJETA (medido sobre train.csv,
699,960 filas, 1.67% de fraude):
- Monto: amount_usd (1-6,870 USD, media 79.9), account_balance_usd.
  La tasa de fraude crece con el monto: < 10 USD 0.52%, 10-50 0.76%,
  50-100 1.08%, 100-500 2.02%, 500-1,000 4.10%, > 1,000 8.36%.
- Autenticación: auth_method. Con autenticación: Biometric (0.50%
  fraude), PIN (0.87%), OTP (1.41%), 3D Secure (1.47%). Sin
  autenticación: No Authentication (6.44% de fraude, 67,060 txns,
  9.58% del total, 37.0% de todo el fraude). cvv_retry_count es forzar
  la verificación con reintentos: 0 -> 0.93%, 1 -> 8.75%,
  2 -> 41.73%, 3 -> 36.0%.
- Concentración por categoría: merchant_category por concentración
  de fraude (cuota por conteo y por monto) + merchant_risk_score.
  Niveles por cuota de fraude (conteo), cortes 10/3/1: Nivel 0 >= 10%
  (Online Retail 14.71%, Groceries 13.56%, Electronics 12.84%,
  Gaming 11.22%), Nivel 1 3-10%, Nivel 2 1-3% (Crypto Exchange
  2.95%), Nivel 3 < 1% (ninguna). Por monto: Electronics 46.9% y
  Travel 21.4% del dinero defraudado.
- Señales de riesgo: la transacción que mueve el mismo monto sin
  autenticación ni verificación. Señales medidas: is_foreign
  (12.57% vs 1.48%), is_new_merchant (4.73% vs 1.12%), used_vpn
  (6.55% vs 1.50%), ip_country_mismatch (16.50% vs 1.43%),
  billing_shipping_mismatch (12.99% vs 1.52%),
  is_ai_generated_scam_attempt (15.22% vs 1.63%), velocity_score
  alto, txn_count_last_24h alto, hours_since_last_txn bajo,
  distance_from_home_km alto (PLANO: 1.72% vs 1.66%, no discrimina),
  prior_disputes (1.55% con 0, 3.64% con 3).
- Intermediario: no medible con estos datos (no hay intermediario
  identificado) — gap registrado, no inventado.

CONTEO DE SEÑALES DE RIESGO DISTINTAS (10 tipos distintos de señal
de riesgo por transacción; umbral >= 5):
  extranjero, comercio nuevo, VPN, mismatch de país, mismatch de
  facturación/envío, intento de estafa por IA, sin autenticación,
  reintento de CVV >= 1, velocidad sobre p90 (velocity_score >= 33.10),
  distancia sobre p90 (distance_from_home_km >= 31.43 km).
  Tasa de fraude por conteo: 0 -> 0.19%, 1 -> 1.06%, 2 -> 5.40%,
  3 -> 22.38%, 4 -> 59.66%, 5 -> 85.40%, 6 -> 98.77%, 7-8 -> 100%.
  Umbral >= 5: 549 txns (0.08%), 87.61% de fraude, lift 52.46x sobre
  la base, captura 4.1% de todo el fraude (481 de 11,689).
  Variante sin distancia: 388 txns, 94.07% de fraude, 3.1% capturado.

CUOTA DE FRAUDE por merchant_category (y por channel):
  cuota = fraude / (total - fraude). Gift Cards 0.0787, Crypto
  Exchange 0.0698, Gaming 0.0583, Electronics 0.0318, Streaming
  0.0230, Travel 0.0189, Utilities 0.0140, Healthcare 0.0130,
  Fuel 0.0125, Restaurants 0.0116, Groceries 0.0112,
  Online Retail 0.0112. Por channel: Online 0.0177, POS 0.0169,
  In-App 0.0166, ATM 0.0161, Contactless 0.0158.

SERIE TEMPORAL: DATOS INSUFICIENTES. No hay fecha, ni id de tarjeta,
ni id de comercio; time_of_day_hour y day_of_week son cíclicos, y el
id es solo orden de fila (corr con time_of_day_hour -0.002, con
day_of_week 0.0005, con is_fraud -0.0006; tasa de fraude por decil de
id plana 1.59%-1.73%). Sin serie por entidad no se ajusta ninguna
tendencia temporal: se declara DATOS INSUFICIENTES con honestidad y
se usan el conteo de señales + autenticación + concentración por
categoría como features. No se inventa ninguna serie.

REGLAS (no se negocian aquí):
- Las cuotas se calculan desde la serie propia de cada categoría.
- NUNCA interpolar. Lo estimado va etiquetado ESTIMADO.
- Las tablas de referencia (cuota por categoría/canal) se calculan
  SOLO con el pliegue de entrenamiento de cada fold: sin fuga.
- El test NUNCA se usa para entrenar: solo se predicen sus
  probabilidades. No se intentan inferir sus etiquetas.
- No se sobreajusta al leaderboard público: la selección de modelo
  se hace por CV local (5 pliegues).

Reglas de la competencia (vinculantes; leídas 2026-11-08 ~13:39):
- ENVÍOS: máximo 10 por día por equipo (Rules Sec. 2.2; el Overview
  decía 5, pero el texto vinculante son 10). Hasta 2 envíos finales
  seleccionables. Equipos de máximo 3 personas.
- El envío es un CSV id,is_fraud (probabilidades). NO es competencia
  de código para puntuar, pero los ganadores deben entregar código de
  entrenamiento + código de inferencia + documentación (Sec. 8.3).
  Este módulo + el notebook son ese código: se conservan limpios y
  reproducibles.
- DATOS EXTERNOS: SÍ permitidos si son públicos, gratuitos y
  accesibles para todos (Sec. 3.1). Candidato: el dataset original
  "Credit Card Fraud Detection 2026" (CC0, dominio público, 20,000
  filas, 1.7% de fraude, mismas columnas salvo transaction_id en vez
  de id). PERO sus distribuciones son cercanas, no idénticas, a las
  de la competencia: si se usa como datos extra de entrenamiento, se
  VALIDA el cambio de distribución (comparar tasas por categoría y
  AUC con y sin él) y se reporta si ayuda o estorba.
  Ver evaluar_con_externo().
- LEADERBOARD: el público es solo una muestra del test; el PRIVADO
  decide el resultado final. No sobreajustar al leaderboard público:
  la selección de modelo se hace por CV local (5 pliegues), no por
  envíos repetidos.
- PROHIBIDO: herramientas AutoML (salvo que su licencia lo permita
  como componente); etiquetar a mano o con ML los registros de test /
  validación; usar información privada o exploits; redistribuir los
  datos de la competencia; compartir código en privado fuera del
  equipo. No se entrena con el test ni se intentan inferir sus
  etiquetas. Este módulo NUNCA toca test.csv salvo para predecir.
- PREMIO: mercancía de Kaggle, valor $0; licencia del ganador "None"
  (no se cede la solución).
- CIERRE: 30 nov 2026, 11:59 PM UTC. Sin envíos hechos todavía.
- DATOS: todo sintético, inspirado en el original; licencia de la
  competencia CC BY 4.0.
"""
import math
import os

import numpy as np
import pandas as pd

__version__ = "1.0.0"
AGENTE = "fraude_tarjetas"
MARCA = "Fractal Core Research"
COMPETENCIA = "playground-series-s6e11"
METRICA = "ROC AUC"
CIERRE = "2026-11-30 23:59 UTC"

# Umbrales de velocidad y distancia, medidos sobre train.csv (percentil 90 de train).
V_UMBRAL = 33.10    # velocity_score p90 (decil superior: fraude 5.26%)
D_UMBRAL = 31.43    # distance_from_home_km p90 (PLANO: 1.72% vs 1.66%)
UMBRAL_SENALES = 5     # >= 5 tipos distintos de evento negativo

# Niveles de concentración por cuota de fraude (conteo), cortes 10/3/1.
# Cortes 10/3/1, declarados aquí.
NIVELES_CONCENTRACION = [(10.0, "Nivel 0"), (3.0, "Nivel 1"),
               (1.0, "Nivel 2")]  # resto: Nivel 3

# Columnas crudas del dataset (train 26, test 25 = train sin is_fraud).
COLS_NUMERICAS = ["amount_usd", "hours_since_last_txn", "txn_count_last_24h",
                  "distance_from_home_km", "card_age_months", "customer_age",
                  "account_balance_usd", "cvv_retry_count", "velocity_score",
                  "time_of_day_hour", "day_of_week", "merchant_risk_score",
                  "prior_disputes"]
COLS_BOOL = ["is_foreign_transaction", "is_new_merchant", "used_vpn",
             "ip_country_mismatch", "billing_shipping_mismatch",
             "is_ai_generated_scam_attempt"]
COLS_CAT = ["merchant_category", "card_type", "auth_method", "channel",
            "device_type"]
TARGET = "is_fraud"


# ---------------------------------------------------------------------------
# Reglas de la competencia (vinculantes; leídas por Mole 2026-11-08 ~13:39)
# ---------------------------------------------------------------------------
# - ENVÍOS: máximo 10 por día por equipo (Rules Sec. 2.2; el Overview decía
#   5, pero el texto vinculante son 10). Hasta 2 envíos finales
#   seleccionables. Equipos de máximo 3 personas.
# - El envío es un CSV id,is_fraud (probabilidades). NO es competencia de
#   código para puntuar, pero los ganadores deben entregar código de
#   entrenamiento + código de inferencia + documentación (Sec. 8.3). Este
#   módulo + el notebook son ese código: se conservan limpios y
#   reproducibles.
# - DATOS EXTERNOS: SÍ permitidos si son públicos, gratuitos y accesibles
#   para todos (Sec. 3.1). Candidato: el dataset original "Credit Card
#   Fraud Detection 2026" (CC0, dominio público, 20,000 filas, 1.7% de
#   fraude, mismas columnas salvo transaction_id en vez de id). PERO sus
#   distribuciones son cercanas, no idénticas, a las de la competencia:
#   si se usa como datos extra de entrenamiento, se VALIDA el cambio de
#   distribución (comparar tasas por categoría y AUC con y sin él) y se
#   reporta si ayuda o estorba. Ver evaluar_con_externo().
# - LEADERBOARD: el público es solo una muestra del test; el PRIVADO
#   decide el resultado final. No sobreajustar al leaderboard público:
#   la selección de modelo se hace por CV local (5 pliegues), no por
#   envíos repetidos.
# - PROHIBIDO: herramientas AutoML (salvo que su licencia lo permita como
#   componente); etiquetar a mano o con ML los registros de test /
#   validación; usar información privada o exploits; redistribuir los
#   datos de la competencia; compartir código en privado fuera del
#   equipo. No se entrena con el test ni se intentan inferir sus
#   etiquetas. Este módulo NUNCA toca test.csv salvo para predecir.
# - PREMIO: mercancía de Kaggle, valor $0; licencia del ganador "None"
#   (no se cede la solución).
# - CIERRE: 30 nov 2026, 11:59 PM UTC. Sin envíos hechos todavía.
# - DATOS: todo sintético, inspirado en el original; licencia de la
#   competencia CC BY 4.0.
REGLAS_COMPETENCIA = {
    "envios_max_dia": 10,
    "envios_finales_seleccionables": 2,
    "equipo_max_personas": 3,
    "metrica": METRICA,
    "cierre": CIERRE,
    "premio": "mercancía de Kaggle, valor $0",
    "datos_externos": "permitidos si públicos, gratuitos y accesibles para todos (Sec. 3.1)",
    "prohibido": "AutoML (salvo licencia que lo permita como componente); "
                 "etiquetar a mano o con ML el test/validación; información "
                 "privada o exploits; redistribuir los datos; compartir "
                 "código en privado fuera del equipo",
    "seleccion_modelo": "CV local de 5 pliegues (el leaderboard público es "
                        "solo una muestra; el privado decide)",
}

# Señales de riesgo: (nombre, descripción, función sobre el DataFrame).
# Los umbrales de velocidad y distancia son p90 de train (medidos).
def _ev_extranjero(df):
    return df["is_foreign_transaction"].astype(bool)


def _ev_comercio_nuevo(df):
    return df["is_new_merchant"].astype(bool)


def _ev_vpn(df):
    return df["used_vpn"].astype(bool)


def _ev_mismatch_pais(df):
    return df["ip_country_mismatch"].astype(bool)


def _ev_mismatch_fact_envio(df):
    return df["billing_shipping_mismatch"].astype(bool)


def _ev_intento_ia(df):
    return df["is_ai_generated_scam_attempt"].astype(bool)


def _ev_sin_autenticacion(df):
    return df["auth_method"] == "No Authentication"


def _ev_cvv_reintento(df):
    return df["cvv_retry_count"] >= 1


def _ev_velocidad_alta(df):
    return df["velocity_score"] >= V_UMBRAL


def _ev_distancia_alta(df):
    return df["distance_from_home_km"] >= D_UMBRAL


SENALES_RIESGO = [
    ("extranjero", "transacción extranjera (is_foreign_transaction)",
     _ev_extranjero),
    ("comercio_nuevo", "comercio nuevo para la tarjeta (is_new_merchant)",
     _ev_comercio_nuevo),
    ("vpn", "VPN en uso (used_vpn)", _ev_vpn),
    ("mismatch_pais", "país de la IP distinto al del titular "
     "(ip_country_mismatch)", _ev_mismatch_pais),
    ("mismatch_fact_envio", "dirección de facturación distinta a la de "
     "envío (billing_shipping_mismatch)", _ev_mismatch_fact_envio),
    ("intento_ia", "intento de estafa generado por IA "
     "(is_ai_generated_scam_attempt)", _ev_intento_ia),
    ("sin_autenticacion", "sin autenticación (auth_method = "
     "'No Authentication')", _ev_sin_autenticacion),
    ("cvv_reintento", "al menos un reintento de CVV (cvv_retry_count >= 1): "
     "forzar la verificación con reintentos", _ev_cvv_reintento),
    ("velocidad_alta", f"velocity_score >= {V_UMBRAL} (p90 de train)",
     _ev_velocidad_alta),
    ("distancia_alta", f"distance_from_home_km >= {D_UMBRAL} km (p90 de "
     "train; PLANO en estos datos: 1.72% vs 1.66%, no discrimina)",
     _ev_distancia_alta),
]


def calcular_senales_riesgo(df):
    """Calcula las 10 señales de riesgo por transacción.

    Devuelve un DataFrame con 10 columnas booleanas (senal_<nombre>),
    n_distinct_risk_signals (conteo de tipos distintos, 0-10) y umbral_5_senales (n_distinct_risk_signals >= 5).
    No usa la etiqueta: funciona igual en train y en test.
    """
    out = pd.DataFrame(index=df.index)
    for nombre, _desc, fn in SENALES_RIESGO:
        out[f"senal_{nombre}"] = fn(df).astype(bool)
    out["n_distinct_risk_signals"] = out.sum(axis=1).astype(int)
    out["umbral_5_senales"] = (out["n_distinct_risk_signals"] >= UMBRAL_SENALES).astype(int)
    return out


def tabla_conteo_senales(df_train):
    """Tasa de fraude por número de señales de riesgo distintas (train).

    Devuelve DataFrame con n_distinct_risk_signals, n, fraude, tasa. La tasa crece
    monótona con el conteo: 0 eventos -> 0.19%, 5 -> 85.40%, 6 -> 98.77%.
    """
    ev = calcular_senales_riesgo(df_train)
    t = df_train[[TARGET]].copy()
    t["n_distinct_risk_signals"] = ev["n_distinct_risk_signals"].values
    g = t.groupby("n_distinct_risk_signals").agg(n=(TARGET, "size"), fraude=(TARGET, "sum"))
    g["tasa"] = g["fraude"] / g["n"]
    g = g.reset_index()
    cruz = ev["umbral_5_senales"].astype(bool).values
    resumen = {
        "n_umbral": int(cruz.sum()),
        "pct_umbral": round(float(cruz.mean() * 100), 2),
        "tasa_umbral": round(float(df_train.loc[cruz, TARGET].mean()), 4),
        "tasa_sin_umbral": round(float(df_train.loc[~cruz, TARGET].mean()), 4),
        "fraudes_capturados": int(df_train.loc[cruz, TARGET].sum()),
        "fraudes_totales": int(df_train[TARGET].sum()),
        "pct_fraudes_capturados": round(
            float(df_train.loc[cruz, TARGET].sum() / df_train[TARGET].sum() * 100), 1),
        "lift": round(float(df_train.loc[cruz, TARGET].mean()
                            / df_train[TARGET].mean()), 2),
    }
    return g, resumen


def tabla_conteo_senales_sin_distancia(df_train):
    """Variante del conteo sin la señal de distancia (que es plana).

    Mismo umbral >= 5, sobre 9 eventos. Medido: 388 txns por encima del
    umbral, 94.07% de fraude, 3.1% de los fraudes capturados (vs 549 /
    87.61% / 4.1% con distancia). La distancia añade 161 txns por
    encima del umbral con menor tasa: es ruido en estos datos, no señal.
    Se reporta, no se esconde.
    """
    ev = calcular_senales_riesgo(df_train)
    conteo9 = ev["n_distinct_risk_signals"] - ev["senal_distancia_alta"].astype(int)
    supera_umbral = (conteo9 >= UMBRAL_SENALES).values
    return {
        "n_umbral": int(supera_umbral.sum()),
        "tasa_umbral": round(float(df_train.loc[supera_umbral, TARGET].mean()), 4),
        "pct_fraudes_capturados": round(
            float(df_train.loc[supera_umbral, TARGET].sum() / df_train[TARGET].sum() * 100), 1),
    }


def nivel_concentracion(cuota_pct):
    """Nivel de concentración por cuota del total de fraude (conteo).

    Cortes 10/3/1, declarados aquí."""
    if cuota_pct is None:
        return ""
    for corte, nombre in NIVELES_CONCENTRACION:
        if cuota_pct >= corte:
            return nombre
    return "Nivel 3"


def tabla_cuota_fraude(df_train, col_grupo):
    """Cuota de fraude transversal por grupo: cuota = fraude / (total - fraude).

    Es una cuota por categoría (o por canal), sin tiempo: no hay serie
    temporal aquí. Devuelve DataFrame ordenado por cuota descendente,
    con n, fraude, cuota de fraude, cuota de fraude por conteo y por
    monto, cuota de txns y nivel de concentración por cuota de fraude.
    """
    g = df_train.groupby(col_grupo).agg(
        n=(TARGET, "size"), fraude=(TARGET, "sum"),
        monto_fraude=("amount_usd",
                      lambda s: float(s[df_train.loc[s.index, TARGET] == 1].sum())),
        monto_total=("amount_usd", "sum"))
    total_fraude = float(df_train[TARGET].sum())
    total_monto_fraude = float(df_train.loc[df_train[TARGET] == 1,
                                           "amount_usd"].sum())
    g["cuota_fraude"] = g["fraude"] / (g["n"] - g["fraude"])
    g["cuota_fraude_n_pct"] = g["fraude"] / total_fraude * 100.0
    g["cuota_fraude_usd_pct"] = g["monto_fraude"] / total_monto_fraude * 100.0
    g["cuota_txn_pct"] = g["n"] / len(df_train) * 100.0
    g["nivel_concentracion"] = g["cuota_fraude_n_pct"].apply(nivel_concentracion)
    return g.sort_values("cuota_fraude", ascending=False).reset_index()


def tabla_auth(df_train):
    """La autenticación por método: tasa de fraude y peso.

    Con autenticación: Biometric, PIN, OTP, 3D Secure. Sin autenticación:
    No Authentication (6.44% de fraude; 37.0% de todo el fraude con
    solo el 9.58% de las transacciones).
    """
    g = df_train.groupby("auth_method").agg(
        n=(TARGET, "size"), fraude=(TARGET, "sum"))
    g["tasa"] = g["fraude"] / g["n"]
    g["cuota_fraude_pct"] = g["fraude"] / float(df_train[TARGET].sum()) * 100.0
    g["cuota_txn_pct"] = g["n"] / len(df_train) * 100.0
    g["auth"] = g.index.map(
        lambda a: "Sin autenticación" if a == "No Authentication" else "Con autenticación")
    return g.sort_values("tasa", ascending=False).reset_index()


def diagnostico_temporal(df_train):
    """¿Hay serie temporal por entidad para ajustar una tendencia temporal? NO.

    Evidencia medida: correlación de id con time_of_day_hour (-0.002),
    con day_of_week (0.0005) y con is_fraud (-0.0006); tasa de fraude
    por decil de id plana (1.59%-1.73%); media de time_of_day_hour por
    decil de id plana (11.72-11.79); no hay columna de fecha, ni id de
    tarjeta, ni id de comercio; time_of_day_hour y day_of_week son
    cíclicos, no una línea de tiempo. Devuelve DATOS INSUFICIENTES con
    la evidencia, no un ajuste inventado.
    """
    d = df_train.copy()
    d["id_decil"] = pd.qcut(d["id"], 10, labels=False)
    por_decil = d.groupby("id_decil").agg(
        tasa_fraude=(TARGET, "mean"),
        hora_media=("time_of_day_hour", "mean"),
        dia_medio=("day_of_week", "mean"))
    return {
        "estado": "datos_insuficientes",
        "razon": "No hay serie temporal real por entidad: no hay fecha, "
                 "ni id de tarjeta, ni id de comercio. El id es solo "
                 "orden de fila y NO es temporal.",
        "evidencia": {
            "corr_id_hora": round(float(d["id"].corr(d["time_of_day_hour"])), 4),
            "corr_id_dia": round(float(d["id"].corr(d["day_of_week"])), 4),
            "corr_id_fraude": round(float(d["id"].corr(d[TARGET])), 4),
            "tasa_fraude_por_decil_id_min": round(float(por_decil["tasa_fraude"].min()), 4),
            "tasa_fraude_por_decil_id_max": round(float(por_decil["tasa_fraude"].max()), 4),
            "hora_media_por_decil_id_min": round(float(por_decil["hora_media"].min()), 2),
            "hora_media_por_decil_id_max": round(float(por_decil["hora_media"].max()), 2),
            "dia_medio_por_decil_id_min": round(float(por_decil["dia_medio"].min()), 2),
            "dia_medio_por_decil_id_max": round(float(por_decil["dia_medio"].max()), 2),
        },
        "faltantes": [
            "columna de fecha u orden temporal real (time_of_day_hour y "
            "day_of_week son cíclicos, no una línea de tiempo)",
            "id de tarjeta (no hay tarjeta que seguir en el tiempo)",
            "id de comercio (no hay comercio que seguir en el tiempo)",
            "serie temporal por entidad con n>=8 para ajustar una tendencia temporal (log-log)",
        ],
        "disponible": [
            "Cuota de fraude transversal por merchant_category y por channel "
            "(sin tiempo)",
            "Conteo de señales de riesgo distintas por transacción (10 tipos de señal)",
            "autenticación por auth_method (con vs sin autenticación)",
            "niveles de concentración por cuota de fraude por categoría",
        ],
    }


# ---------------------------------------------------------------------------
# Features (crudas vs señales de riesgo) — con codificación target SIN fuga
# ---------------------------------------------------------------------------
def _tablas_referencia(df_ref):
    """Tablas de referencia (R por categoría/canal, cuotas, niveles) desde
    un marco de referencia (en CV: el pliegue de entrenamiento; en final:
    todo train). Nunca desde el test ni desde el pliegue de validación."""
    t_cat = tabla_cuota_fraude(df_ref, "merchant_category")
    t_ch = tabla_cuota_fraude(df_ref, "channel")
    ref = {
        "cuota_fraude_categoria": dict(zip(t_cat["merchant_category"], t_cat["cuota_fraude"])),
        "cuota_fraude_canal": dict(zip(t_ch["channel"], t_ch["cuota_fraude"])),
        "cuota_fraude_n": dict(zip(t_cat["merchant_category"],
                                   t_cat["cuota_fraude_n_pct"])),
        "cuota_fraude_usd": dict(zip(t_cat["merchant_category"],
                                     t_cat["cuota_fraude_usd_pct"])),
        "nivel_concentracion": dict(zip(t_cat["merchant_category"],
                              t_cat["nivel_concentracion"])),
        "cuota_fraude_global": float(df_ref[TARGET].mean()),
        "tabla_categoria": t_cat,
        "tabla_channel": t_ch,
    }
    return ref


_NIVEL_ORD = {"Nivel 0": 0, "Nivel 1": 1, "Nivel 2": 2,
              "Nivel 3": 3}


def construir_features_senales(df, ref):
    """Features de señales de riesgo por transacción (18): las 10 señales,
    el conteo, el umbral >= 5 señales, sin autenticación, la cuota de
    fraude de su categoría y de su canal, las cuotas de fraude de su
    categoría y su nivel de concentración. Todo se calcula desde `ref` (marco de entrenamiento),
    nunca desde la etiqueta de la fila."""
    ev = calcular_senales_riesgo(df)
    X = pd.DataFrame(index=df.index)
    for nombre, _d, _f in SENALES_RIESGO:
        X[f"senal_{nombre}"] = ev[f"senal_{nombre}"].astype(int)
    X["n_distinct_risk_signals"] = ev["n_distinct_risk_signals"]
    X["umbral_5_senales"] = ev["umbral_5_senales"]
    X["no_auth"] = (df["auth_method"] == "No Authentication").astype(int)
    X["cuota_fraude_categoria"] = df["merchant_category"].map(
        ref["cuota_fraude_categoria"]).fillna(ref["cuota_fraude_global"])
    X["cuota_fraude_canal"] = df["channel"].map(ref["cuota_fraude_canal"]).fillna(ref["cuota_fraude_global"])
    X["cuota_fraude_categoria_n"] = df["merchant_category"].map(
        ref["cuota_fraude_n"]).fillna(0.0)
    X["cuota_fraude_categoria_usd"] = df["merchant_category"].map(
        ref["cuota_fraude_usd"]).fillna(0.0)
    X["nivel_concentracion_categoria"] = df["merchant_category"].map(
        ref["nivel_concentracion"]).map(_NIVEL_ORD).fillna(3).astype(int)
    return X


def construir_features_crudas(df, encoder=None, fit=False):
    """Features crudas: 14 numéricas + 6 booleanas + one-hot de las 5
    categóricas (40 dummies). El encoder se ajusta SOLO en el marco de
    entrenamiento del pliegue (handle_unknown='ignore')."""
    from sklearn.preprocessing import OneHotEncoder
    X = pd.DataFrame(index=df.index)
    for c in COLS_NUMERICAS:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    for c in COLS_BOOL:
        X[c] = df[c].astype(int)
    cats = df[COLS_CAT].astype(str)
    if fit or encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        enc = encoder.fit_transform(cats)
    else:
        enc = encoder.transform(cats)
    nombres = encoder.get_feature_names_out(COLS_CAT)
    X_enc = pd.DataFrame(enc, columns=nombres, index=df.index)
    return pd.concat([X, X_enc], axis=1), encoder


def construir_features(df, modo, ref, encoder=None, fit_encoder=False):
    """modo: 'crudas' | 'senales' | 'combinadas'. Devuelve (X, encoder)."""
    if modo == "senales":
        return construir_features_senales(df, ref), encoder
    X_raw, encoder = construir_features_crudas(df, encoder, fit_encoder)
    if modo == "crudas":
        return X_raw, encoder
    X_senales = construir_features_senales(df, ref)
    return pd.concat([X_raw, X_senales], axis=1), encoder


# ---------------------------------------------------------------------------
# Modelo y validación cruzada
# ---------------------------------------------------------------------------
def _hgb():
    """HistGradientBoostingClassifier (disponible en notebooks Kaggle).
    Parámetros fijos y declarados: 200 árboles, lr 0.08, 31 hojas."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=50, l2_regularization=1.0, random_state=42)


def evaluar_cv(df_train, modo, n_splits=5, seed=42, verbose=True):
    """CV estratificada de 5 pliegues para un modo de features.

    Las tablas de referencia (R por categoría/canal) se calculan SOLO
    con el pliegue de entrenamiento de cada fold: sin fuga. Devuelve
    dict con la lista de AUC por pliegue, media y desviación estándar.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    y = df_train[TARGET].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for i, (tr_idx, va_idx) in enumerate(skf.split(df_train, y)):
        d_tr, d_va = df_train.iloc[tr_idx], df_train.iloc[va_idx]
        ref = _tablas_referencia(d_tr)
        X_tr, enc = construir_features(d_tr, modo, ref, fit_encoder=True)
        X_va, _ = construir_features(d_va, modo, ref, encoder=enc)
        modelo = _hgb()
        modelo.fit(X_tr, y[tr_idx])
        p = modelo.predict_proba(X_va)[:, 1]
        auc = float(roc_auc_score(y[va_idx], p))
        aucs.append(round(auc, 4))
        if verbose:
            print(f"  pliegue {i + 1}/{n_splits} ({modo}): AUC = {auc:.4f}")
    return {"modo": modo, "aucs": aucs,
            "auc_media": round(float(np.mean(aucs)), 4),
            "auc_std": round(float(np.std(aucs)), 4),
            "n_pliegues": n_splits}


def comparar_modelos(df_train, seed=42):
    """Compara los tres modos: (a) solo crudas, (b) solo señales de riesgo,
    (c) combinadas. La pregunta es si las señales de riesgo añaden señal.
    Baseline nulo: constante 0.01669953 -> AUC 0.5."""
    out = {}
    for modo in ("crudas", "senales", "combinadas"):
        out[modo] = evaluar_cv(df_train, modo, seed=seed)
    out["baseline_constante"] = {"modo": "baseline_constante",
                                 "auc_media": 0.5, "auc_std": 0.0,
                                 "aucs": [0.5] * 5}
    return out


def entrenar_final(df_train):
    """Entrena el modelo combinado con TODO train (tablas de referencia
    sobre todo train). Devuelve (modelo, encoder, ref)."""
    ref = _tablas_referencia(df_train)
    X, enc = construir_features(df_train, "combinadas", ref, fit_encoder=True)
    modelo = _hgb()
    modelo.fit(X, df_train[TARGET].values)
    return modelo, enc, ref


def entrenar_final_con_externo(df_train, df_ext, modo="combinadas"):
    """Entrena el modelo final con TODO train + externo (mismo protocolo
    que la CV con externo: tablas de referencia sobre el concatenado).
    Devuelve (modelo, encoder, ref)."""
    ext = df_ext.copy()
    if "transaction_id" in ext.columns and "id" not in ext.columns:
        ext = ext.rename(columns={"transaction_id": "id"})
    cols = [c for c in df_train.columns if c in ext.columns]
    ext = ext[cols]
    d = pd.concat([df_train, ext], ignore_index=True)
    ref = _tablas_referencia(d)
    X, enc = construir_features(d, modo, ref, fit_encoder=True)
    modelo = _hgb()
    modelo.fit(X, d[TARGET].values)
    return modelo, enc, ref


def predecir_test(modelo, encoder, ref, df_test, modo="combinadas"):
    """Probabilidades de fraude para test (nunca se entrena con test;
    solo se predicen sus probabilidades)."""
    X, _ = construir_features(df_test, modo, ref, encoder=encoder)
    return modelo.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Datos externos (original CC0) — variante comparada, no obligatoria
# ---------------------------------------------------------------------------
def evaluar_con_externo(df_train, df_ext, seed=42):
    """Compara train solo vs train + dataset original CC0, en los 3 modos.

    El original ("Credit Card Fraud Detection 2026", CC0, 20,000 filas)
    tiene las mismas columnas salvo transaction_id en vez de id. Sus
    distribuciones son cercanas, no idénticas, a las de la competencia:
    primero se compara la distribución (tasa de fraude global, por
    categoría, por método de auth, por intento de IA, medias de las
    numéricas) y luego se corre la CV en los tres modos (crudas,
    señales, combinadas) entrenando con train + externo y validando
    SOLO sobre los pliegues de la competencia (el externo nunca entra
    en la validación). Devuelve el dict de comparación con "por_modo"
    (AUC con/sin externo por modo) y la decisión en "modo_ganador" /
    "veredicto" (ver decidir_externo). Las claves de nivel superior
    auc_con_externo_media / auc_sin_externo_media / std / ayuda son
    las del modo combinadas (compatibilidad con la celda del notebook).
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    ext = df_ext.copy()
    if "transaction_id" in ext.columns and "id" not in ext.columns:
        ext = ext.rename(columns={"transaction_id": "id"})
    cols = [c for c in df_train.columns if c in ext.columns]
    ext = ext[cols]
    comp = {
        "n_externo": int(len(ext)),
        "tasa_fraude_externo": round(float(ext[TARGET].mean()), 4),
        "tasa_fraude_competencia": round(float(df_train[TARGET].mean()), 4),
        "tasa_por_categoria_externo": ext.groupby("merchant_category")[TARGET].mean().round(4).to_dict(),
        "tasa_por_categoria_competencia": df_train.groupby("merchant_category")[TARGET].mean().round(4).to_dict(),
        "tasa_por_auth_externo": ext.groupby("auth_method")[TARGET].mean().round(4).to_dict(),
        "tasa_por_auth_competencia": df_train.groupby("auth_method")[TARGET].mean().round(4).to_dict(),
        "tasa_intento_ia_externo": round(float(ext.loc[ext["is_ai_generated_scam_attempt"], TARGET].mean()), 4),
        "tasa_intento_ia_competencia": round(float(df_train.loc[df_train["is_ai_generated_scam_attempt"], TARGET].mean()), 4),
        "medias_numericas_externo": ext[COLS_NUMERICAS].mean().round(2).to_dict(),
        "medias_numericas_competencia": df_train[COLS_NUMERICAS].mean().round(2).to_dict(),
    }
    y = df_train[TARGET].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    por_modo = {}
    for modo in ("crudas", "senales", "combinadas"):
        aucs_con, aucs_sin = [], []
        for tr_idx, va_idx in skf.split(df_train, y):
            d_tr = pd.concat([df_train.iloc[tr_idx], ext], ignore_index=True)
            d_va = df_train.iloc[va_idx]
            ref = _tablas_referencia(d_tr)
            X_tr, enc = construir_features(d_tr, modo, ref, fit_encoder=True)
            X_va, _ = construir_features(d_va, modo, ref, encoder=enc)
            m = _hgb()
            m.fit(X_tr, d_tr[TARGET].values)
            aucs_con.append(float(roc_auc_score(y[va_idx],
                                                m.predict_proba(X_va)[:, 1])))
            ref2 = _tablas_referencia(df_train.iloc[tr_idx])
            X_tr2, enc2 = construir_features(df_train.iloc[tr_idx], modo,
                                             ref2, fit_encoder=True)
            X_va2, _ = construir_features(d_va, modo, ref2, encoder=enc2)
            m2 = _hgb()
            m2.fit(X_tr2, y[tr_idx])
            aucs_sin.append(float(roc_auc_score(y[va_idx],
                                                m2.predict_proba(X_va2)[:, 1])))
        por_modo[modo] = {
            "auc_con_externo_media": round(float(np.mean(aucs_con)), 4),
            "auc_con_externo_std": round(float(np.std(aucs_con)), 4),
            "auc_sin_externo_media": round(float(np.mean(aucs_sin)), 4),
            "auc_sin_externo_std": round(float(np.std(aucs_sin)), 4),
            "aucs_con": [round(a, 4) for a in aucs_con],
            "aucs_sin": [round(a, 4) for a in aucs_sin],
        }
    comp["por_modo"] = por_modo
    comb = por_modo["combinadas"]
    comp["auc_con_externo_media"] = comb["auc_con_externo_media"]
    comp["auc_sin_externo_media"] = comb["auc_sin_externo_media"]
    comp["auc_con_externo_std"] = comb["auc_con_externo_std"]
    comp["auc_sin_externo_std"] = comb["auc_sin_externo_std"]
    comp["ayuda"] = bool(comb["auc_con_externo_media"] > comb["auc_sin_externo_media"])
    comp["mejor_sin_externo"] = round(max(por_modo[m]["auc_sin_externo_media"]
                                          for m in por_modo), 4)
    comp["mejor_modo_sin_externo"] = max(por_modo,
                                         key=lambda m: por_modo[m]["auc_sin_externo_media"])
    comp["modo_ganador"], comp["veredicto"] = decidir_externo(comp)
    return comp


def decidir_externo(comp):
    """Decisión con números: una variante con externo gana solo si supera
    al mejor AUC sin externo por más de la std entre pliegues de esa
    variante (una diferencia menor que la std es ruido). Devuelve
    (modo_ganador o None, veredicto)."""
    mejor_sin = comp["mejor_sin_externo"]
    ganador, mejor_delta = None, 0.0
    for modo, r in comp["por_modo"].items():
        delta = r["auc_con_externo_media"] - mejor_sin
        if delta > r["auc_con_externo_std"] and delta > mejor_delta:
            ganador, mejor_delta = modo, delta
    if ganador:
        r = comp["por_modo"][ganador]
        veredicto = (f"AYUDA: el externo sube el AUC en modo {ganador} "
                     f"({r['auc_con_externo_media']:.4f} ± "
                     f"{r['auc_con_externo_std']:.4f} vs mejor sin externo "
                     f"{mejor_sin:.4f}; Δ +{mejor_delta:.4f} > std "
                     f"{r['auc_con_externo_std']:.4f}). El envío se "
                     f"regenera con train + externo, modo {ganador}.")
    else:
        veredicto = ("NO AYUDA: ninguna variante con externo supera al "
                     f"mejor sin externo ({mejor_sin:.4f}) por más de la "
                     "std entre pliegues; el externo se descarta como "
                     "datos extra (se documenta, no se usa).")
    return ganador, veredicto


# ---------------------------------------------------------------------------
# Envío y verificación
# ---------------------------------------------------------------------------
def verificar_submission(sub, sample):
    """Verifica el envío contra sample_submission.csv.

    Checks: mismas ids, mismo orden, 299,984 filas, 2 columnas
    (id, is_fraud), sin NaN, probabilidades en [0, 1]."""
    checks = []

    def _check(nombre, ok, detalle=""):
        checks.append({"check": nombre, "ok": bool(ok), "detalle": str(detalle)})
        return bool(ok)

    _check("columnas_exactas", list(sub.columns) == ["id", "is_fraud"],
           f"columnas: {list(sub.columns)}")
    _check("n_filas", len(sub) == len(sample),
           f"{len(sub)} vs {len(sample)}")
    _check("mismos_ids", set(sub["id"]) == set(sample["id"]),
           "ids de test")
    _check("mismo_orden_ids", bool((sub["id"].values == sample["id"].values).all()),
           "orden idéntico a sample_submission.csv")
    _check("sin_nan", not bool(sub["is_fraud"].isna().any()),
           f"NaN: {int(sub['is_fraud'].isna().sum())}")
    _check("prob_en_0_1", bool(((sub["is_fraud"] >= 0) & (sub["is_fraud"] <= 1)).all()),
           f"min={float(sub['is_fraud'].min()):.4f} max={float(sub['is_fraud'].max()):.4f}")
    _check("sin_duplicados_id", not bool(sub["id"].duplicated().any()),
           f"duplicados: {int(sub['id'].duplicated().sum())}")
    return checks


def escribir_submission(df_test, probs, ruta):
    """Escribe submission.csv con id,is_fraud (mismo orden que test)."""
    sub = pd.DataFrame({"id": df_test["id"].values,
                        "is_fraud": np.asarray(probs, dtype=float)})
    sub.to_csv(ruta, index=False)
    return sub


# ---------------------------------------------------------------------------
# Figuras e informe
# ---------------------------------------------------------------------------
def _figuras(dir_salida, t_senales, t_cat, t_auth):
    """3 figuras del informe (matplotlib, backend Agg)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(dir_salida, exist_ok=True)
    rutas = []

    # 1. Tasa de fraude por número de señales de riesgo (escala log).
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = t_senales["n_distinct_risk_signals"].values
    ys = t_senales["tasa"].values
    ax.plot(xs, ys, marker="o")
    ax.set_yscale("log")
    ax.set_xlabel("número de tipos distintos de señal de riesgo")
    ax.set_ylabel("tasa de fraude (escala log)")
    ax.set_title("Conteo de señales de riesgo — tasa de fraude por conteo "
                 "(train, n=699,960)")
    ax.axvline(5, color="red", linewidth=1.5, linestyle="--",
               label="umbral >= 5 señales")
    for x, y, n in zip(xs, ys, t_senales["n"].values):
        ax.annotate(f"{y:.2%}\nn={n:,}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    p = os.path.join(dir_salida, "fig-conteo-senales.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    rutas.append(os.path.basename(p))

    # 2. Cuota de fraude por merchant_category.
    fig, ax = plt.subplots(figsize=(9, 5))
    t = t_cat.sort_values("cuota_fraude", ascending=True)
    ax.barh(t["merchant_category"], t["cuota_fraude"])
    ax.set_xlabel("cuota de fraude = fraude / (total - fraude)")
    ax.set_title("Cuota de fraude por merchant_category (train)")
    for i, v in enumerate(t["cuota_fraude"].values):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=8)
    p = os.path.join(dir_salida, "fig-cuota-fraude-categoria.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    rutas.append(os.path.basename(p))

    # 3. Tasa de fraude por método de autenticación.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    t = t_auth.sort_values("tasa", ascending=True)
    colores = ["red" if l == "Sin autenticación" else "steelblue"
               for l in t["auth"].values]
    ax.barh(t["auth_method"], t["tasa"] * 100, color=colores)
    ax.set_xlabel("tasa de fraude (%)")
    ax.set_title("Tasa de fraude por método de autenticación — sin autenticación "
                 "(rojo) concentra el fraude")
    for i, v in enumerate(t["tasa"].values):
        ax.text(v * 100 + 0.1, i, f"{v:.2%}", va="center", fontsize=8)
    p = os.path.join(dir_salida, "fig-auth.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    rutas.append(os.path.basename(p))
    return rutas


def informe(resultados, dir_salida=None):
    """Genera el informe en markdown de la adaptación a fraude con tarjeta.

    `resultados` es el dict que devuelve `ejecutar()` (o un dict parcial
    con las claves que existan). Devuelve el texto markdown y, si se da
    dir_salida, lo escribe junto con las figuras.
    """
    L = []
    L.append("# Fraude con tarjeta (Playground Series S6E11)\n")
    L.append(f"**{MARCA}** — Adaptador v{__version__} "
             f"(fraude_tarjetas.py). Generado por el adaptador, no a mano.\n")
    L.append("**Competencia:** Kaggle Playground Series S6E11, "
             "\"Predicting Credit Card Fraud\". Métrica: ROC AUC. Envío: "
             "CSV id,is_fraud (probabilidades 0–1). Cierre: 30 nov 2026, "
             "11:59 PM UTC. Sin envíos hechos todavía.\n")
    L.append("**Datos:** train.csv 699,960 filas, test.csv 299,984 filas, "
             "sample_submission.csv 299,984 filas. Tasa de fraude en "
             "train: 1.67% (11,689 fraudes). Cero nulos en train y test. "
             "Todo sintético, inspirado en el original; licencia de la "
             "competencia CC BY 4.0.\n")

    # Reglas de la competencia (vinculantes).
    L.append("## 0. Reglas de la competencia (vinculantes)\n")
    L.append("- **Envíos:** máximo 10 por día por equipo (Rules Sec. 2.2; "
             "el Overview decía 5, el texto vinculante son 10). Hasta 2 "
             "envíos finales seleccionables. Equipos de máximo 3 personas.")
    L.append("- **Código:** no es competencia de código para puntuar, "
             "pero los ganadores deben entregar código de entrenamiento + "
             "código de inferencia + documentación (Sec. 8.3). Este módulo "
             "y el notebook son ese código: se conservan limpios y "
             "reproducibles.")
    L.append("- **Datos externos:** permitidos si son públicos, gratuitos "
             "y accesibles para todos (Sec. 3.1). Candidato: el dataset "
             "original \"Credit Card Fraud Detection 2026\" (CC0, 20,000 "
             "filas). Sus distribuciones son cercanas, no idénticas: se "
             "valida el cambio de distribución antes de usarlo (ver §6).")
    L.append("- **Leaderboard:** el público es solo una muestra del test; "
             "el PRIVADO decide el resultado final. La selección de modelo "
             "se hace por CV local (5 pliegues), no por envíos repetidos "
             "ni por sobreajuste al leaderboard público.")
    L.append("- **Prohibido:** AutoML (salvo licencia que lo permita como "
             "componente); etiquetar a mano o con ML el test/validación; "
             "información privada o exploits; redistribuir los datos de la "
             "competencia; compartir código en privado fuera del equipo. "
             "Este adaptador NUNCA entrena con el test ni infiere sus "
             "etiquetas.")
    L.append("- **Premio:** mercancía de Kaggle, valor $0; licencia del "
             "ganador \"None\" (no se cede la solución).\n")

    # Mapeo de features.
    L.append("## 1. Las features (medidas, no declaradas)\n")
    L.append("| Grupo | Definición (medida) |\n|---|---|")
    L.append("| **Monto** | amount_usd (el dinero que se mueve; 1-6,870 USD, "
             "media 79.9) y account_balance_usd (el saldo de la cuenta). |")
    L.append("| **Autenticación** | auth_method. Con autenticación: Biometric "
             "0.50%, PIN 0.87%, OTP 1.41%, 3D Secure 1.47% de fraude. Sin "
             "autenticación: No Authentication 6.44% (37.0% de todo el "
             "fraude con 9.58% de las txns). cvv_retry_count es forzar la "
             "verificación con reintentos (0 -> 0.93%, 1 -> 8.75%, "
             "2 -> 41.73%, 3 -> 36.0%). |")
    L.append("| **Concentración por categoría** | merchant_category por "
             "concentración de fraude (cuota por conteo y por monto) + "
             "merchant_risk_score. Niveles por cuota de fraude: Nivel 0 "
             ">= 10%, Nivel 1 3-10%, Nivel 2 1-3%, Nivel 3 < 1%. |")
    L.append("| **Señales de riesgo** | La transacción que mueve el mismo "
             "monto sin autenticación ni verificación: transacción "
             "extranjera, comercio nuevo, VPN, mismatch de país, mismatch "
             "de facturación/envío, intento de estafa por IA, reintentos "
             "de CVV, velocidad alta, distancia alta. |")
    L.append("| **Intermediario** | No medible con estos datos (no hay "
             "intermediario identificado) — gap registrado, no inventado. |")
    L.append("")
    L.append("**Ficha por categoría (merchant_category, por observables):** "
             "origen = categoría del comercio; posición = nivel de "
             "concentración por cuota de fraude; naturaleza = de alto "
             "riesgo si su cuota de fraude es alta y su autenticación es "
             "baja, de bajo riesgo si su cuota es baja; patrón = concentra "
             "fraude (cuota alta) o lo distribuye (cuota baja). Las fichas "
             "completas están en la tabla de cuota de fraude (§3).\n")

    # Conteo de señales.
    t_senales = resultados.get("tabla_conteo_senales")
    res_senales = resultados.get("resumen_senales", {})
    if t_senales is not None:
        L.append("## 2. Conteo de señales de riesgo distintas\n")
        L.append("Por transacción, se cuentan los tipos DISTINTOS de "
                 "señal de riesgo entre 10 definidas (extranjero, comercio "
                 "nuevo, VPN, mismatch de país, mismatch de facturación/"
                 "envío, intento de estafa por IA, sin autenticación, "
                 "reintento de CVV, velocidad sobre p90, distancia sobre "
                 "p90). Umbral: >= 5 tipos distintos.\n")
        L.append("| Señales distintas | n | Tasa de fraude |")
        L.append("|---|---|---|")
        for _, r in t_senales.iterrows():
            L.append(f"| {int(r['n_distinct_risk_signals'])} | "
                     f"{int(r['n']):,} | {r['tasa']:.2%} |")
        L.append("")
        L.append(f"**Umbral >= 5 señales:** {res_senales.get('n_umbral', '')} "
                 f"transacciones ({res_senales.get('pct_umbral', '')}% del "
                 f"total), tasa de fraude "
                 f"{res_senales.get('tasa_umbral', ''):.2%} frente a "
                 f"{res_senales.get('tasa_sin_umbral', ''):.2%} sin llegar "
                 f"al umbral; captura "
                 f"{res_senales.get('fraudes_capturados', '')} de "
                 f"{res_senales.get('fraudes_totales', '')} fraudes "
                 f"({res_senales.get('pct_fraudes_capturados', '')}%). "
                 f"Lift: {res_senales.get('lift', '')}x sobre la base. La "
                 f"tasa crece monótona con el conteo (0 señales: 0.19%; 3: "
                 f"22.38%; 4: 59.66%; 5: 85.40%; 6: 98.77%; 7-8: 100%).\n")
        var = resultados.get("conteo_senales_sin_distancia")
        if var:
            L.append("**Variante sin distancia:** la señal de distancia "
                     "es PLANA (fraude 1.72% sobre p90 vs 1.66% bajo p90): "
                     "no discrimina en estos datos. Sin ella, el umbral "
                     f">= 5 son {var['n_umbral']} txns con "
                     f"{var['tasa_umbral']:.2%} de fraude y "
                     f"{var['pct_fraudes_capturados']}% de los fraudes "
                     f"capturados (vs 549 / 87.61% / 4.1% con distancia). "
                     f"La distancia añade 161 txns por encima del umbral "
                     f"con menor tasa: es ruido en estos datos, no señal. "
                     f"Se mantiene en el conteo por definición y se "
                     f"reporta la variante.\n")
        L.append("**El conteo de señales solo ya tiene AUC 0.8702** "
                 "(Mann-Whitney sobre train): un solo entero, sin modelo, "
                 "separa fraude de no-fraude mejor que merchant_risk_score "
                 "solo (0.7489) o cvv_retry_count solo (0.7081). El umbral "
                 ">= 5 solo tiene AUC 0.5205: es preciso (87.61%) pero "
                 "raro (0.08% de las txns), así que su valor está en el "
                 "conteo, no en el binario.\n")

    # Cuota de fraude por categoría.
    t_cat = resultados.get("tabla_cuota_fraude_categoria")
    if t_cat is not None:
        L.append("## 3. Cuota de fraude por merchant_category "
                 "(transversal, sin tiempo)\n")
        L.append("Cuota de fraude = fraude / (total - fraude). Es una "
                 "cuota por categoría, sin tiempo: no hay serie temporal "
                 "aquí.\n")
        L.append("| Categoría | n | Fraude | Cuota | Cuota fraude (n) | "
                 "Cuota fraude (USD) | Nivel |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in t_cat.iterrows():
            L.append(f"| {r['merchant_category']} | {int(r['n']):,} | "
                     f"{int(r['fraude']):,} | {r['cuota_fraude']:.4f} | "
                     f"{r['cuota_fraude_n_pct']:.2f}% | "
                     f"{r['cuota_fraude_usd_pct']:.2f}% | "
                     f"{r['nivel_concentracion']} |")
        L.append("")
        L.append("**Lectura:** por conteo de fraude las categorías con "
                 "más fraude son Online Retail (14.71%), Groceries "
                 "(13.56%), Electronics (12.84%) y Gaming (11.22%) — las "
                 "cuatro de Nivel 0. Por monto de fraude la concentración "
                 "es mucho mayor: Electronics 46.90% y Travel 21.42% del "
                 "dinero defraudado (Gift Cards solo 1.96% del monto pese "
                 "a su cuota más alta: mucho fraude chico). La cuota mide "
                 "la tasa; la cuota de fraude por conteo/monto mide dónde "
                 "está el fraude: son dos ejes.\n")

    # Serie temporal insuficiente.
    L.append("## 4. Serie temporal: DATOS INSUFICIENTES (honesto)\n")
    diag = resultados.get("diagnostico_temporal", {})
    if diag:
        L.append("No hay serie temporal real por entidad para ajustar una "
                 "tendencia temporal. Evidencia medida: correlación de id "
                 "con time_of_day_hour "
                 f"{diag['evidencia']['corr_id_hora']}, con day_of_week "
                 f"{diag['evidencia']['corr_id_dia']}, con is_fraud "
                 f"{diag['evidencia']['corr_id_fraude']}; tasa de fraude "
                 f"por decil de id entre "
                 f"{diag['evidencia']['tasa_fraude_por_decil_id_min']:.2%} y "
                 f"{diag['evidencia']['tasa_fraude_por_decil_id_max']:.2%} "
                 f"(plana); media de time_of_day_hour por decil de id entre "
                 f"{diag['evidencia']['hora_media_por_decil_id_min']} y "
                 f"{diag['evidencia']['hora_media_por_decil_id_max']} "
                 f"(plana). El id es solo orden de fila, no tiempo.\n")
        L.append("**Lo que falta (exacto):**\n")
        for f in diag["faltantes"]:
            L.append(f"- {f}\n")
        L.append("**Lo que sí existe:**\n")
        for d in diag["disponible"]:
            L.append(f"- {d}\n")

    # Modelo.
    comp = resultados.get("comparacion_modelos")
    if comp:
        L.append("## 5. Modelo: ¿las señales de riesgo añaden señal?\n")
        L.append("HistGradientBoostingClassifier (200 árboles, lr 0.08, "
                 "31 hojas), CV estratificada de 5 pliegues, AUC media ± "
                 "std. Las tablas de referencia (cuota de fraude por "
                 "categoría/canal) se calculan SOLO con el pliegue de "
                 "entrenamiento de cada fold: sin fuga. Baseline nulo "
                 "(constante 0.01669953): AUC 0.5.\n")
        L.append("| Features | AUC media | std | AUC por pliegue |")
        L.append("|---|---|---|---|")
        for modo in ("crudas", "senales", "combinadas"):
            r = comp[modo]
            etiqueta = {"crudas": "Solo crudas",
                        "senales": "Solo señales de riesgo",
                        "combinadas": "Combinadas (crudas + señales)"}[modo]
            L.append(f"| {etiqueta} | {r['auc_media']:.4f} | "
                     f"{r['auc_std']:.4f} | {r['aucs']} |")
        L.append("")
        c, s = comp["combinadas"]["auc_media"], comp["senales"]["auc_media"]
        r_ = comp["crudas"]["auc_media"]
        L.append(f"**Lectura:** combinadas {c:.4f} vs solo crudas "
                 f"{r_:.4f} vs solo señales {s:.4f}. "
                 + ("Las señales de riesgo añaden señal: las features de "
                    "señales (conteo, cuota por categoría, autenticación) "
                    "suben el AUC sobre las crudas solas." if c > r_ else
                    "Las señales de riesgo NO añaden señal sobre las "
                    "crudas en AUC: el modelo combinado no supera al de "
                    "solo crudas (las features de señales son "
                    "transformaciones de columnas crudas que el árbol ya "
                    "puede usar). El valor de las señales aquí es la "
                    "lectura (conteo, autenticación, concentración por "
                    "categoría), no el AUC.") +
                 "\n")

    # Externo.
    ext = resultados.get("externo")
    if ext:
        L.append("## 6. Datos externos (original CC0): variante comparada\n")
        L.append(f"Dataset original \"Credit Card Fraud Detection 2026\" "
                 f"(CC0, {ext['n_externo']:,} filas). Tasa de fraude: "
                 f"externo {ext['tasa_fraude_externo']:.2%} vs competencia "
                 f"{ext['tasa_fraude_competencia']:.2%}. El cambio de "
                 f"distribución es real: monto medio "
                 f"{ext['medias_numericas_externo']['amount_usd']:.1f} vs "
                 f"{ext['medias_numericas_competencia']['amount_usd']:.1f} "
                 f"USD; distancia media "
                 f"{ext['medias_numericas_externo']['distance_from_home_km']:.1f} vs "
                 f"{ext['medias_numericas_competencia']['distance_from_home_km']:.1f} km; "
                 f"sin autenticación "
                 f"{ext['tasa_por_auth_externo']['No Authentication']:.2%} vs "
                 f"{ext['tasa_por_auth_competencia']['No Authentication']:.2%}; "
                 f"intento de IA {ext['tasa_intento_ia_externo']:.2%} vs "
                 f"{ext['tasa_intento_ia_competencia']:.2%}; Gift Cards "
                 f"{ext['tasa_por_categoria_externo']['Gift Cards']:.2%} vs "
                 f"{ext['tasa_por_categoria_competencia']['Gift Cards']:.2%}.\n")
        L.append("CV de 5 pliegues validando SOLO sobre pliegues de la "
                 "competencia (el externo solo entra como entrenamiento "
                 "extra):\n")
        L.append("| Modo | AUC con externo | std | AUC sin externo | std | Δ vs mejor sin externo |")
        L.append("|---|---|---|---|---|---|")
        for modo in ("crudas", "senales", "combinadas"):
            r = ext["por_modo"][modo]
            delta = r["auc_con_externo_media"] - ext["mejor_sin_externo"]
            L.append(f"| {modo} | {r['auc_con_externo_media']:.4f} | "
                     f"{r['auc_con_externo_std']:.4f} | "
                     f"{r['auc_sin_externo_media']:.4f} | "
                     f"{r['auc_sin_externo_std']:.4f} | {delta:+.4f} |")
        L.append("")
        L.append(f"Mejor sin externo: {ext['mejor_modo_sin_externo']} "
                 f"{ext['mejor_sin_externo']:.4f}. **Veredicto: "
                 f"{ext['veredicto']}**\n")
    elif resultados.get("externo_pendiente"):
        L.append("## 6. Datos externos (original CC0): PENDIENTE de descarga\n")
        L.append("El dataset original \"Credit Card Fraud Detection 2026\" "
                 "(CC0, dominio público, 20,000 filas, 1.7% de fraude, "
                 "mismas columnas salvo transaction_id en vez de id) vive "
                 "en Kaggle como dataset público: se descarga con sesión "
                 "de Kaggle (mismo procedimiento que los datos de la "
                 "competencia). Desde esta máquina no se pudo descargar "
                 "sin sesión (la API sin credenciales no responde). La "
                 "comparación está lista en `evaluar_con_externo()`: "
                 "compara la distribución (tasas por categoría y por "
                 "método de auth) y corre la CV combinada entrenando con "
                 "train + externo y validando SOLO sobre los pliegues de "
                 "la competencia. Cuando el archivo esté en "
                 "`datos/credit-card-fraud-detection-2026.csv`, la celda "
                 "del notebook lo corre sola. Hasta entonces: NO se usa "
                 "ningún dato externo.\n")

    # Envío.
    env = resultados.get("envio")
    if env:
        L.append("## 7. Envío (submission.csv)\n")
        if env.get("con_externo"):
            origen = (f"Modelo final ({env.get('modo', 'combinadas')}) "
                      f"entrenado con TODO train + externo "
                      f"(el externo subió el AUC fuera del ruido, §6).")
        else:
            origen = (f"Modelo final ({env.get('modo', 'combinadas')}) "
                      f"entrenado con TODO train.")
        L.append(origen +
                 f" Archivo: `{env['ruta']}` — {env['n_filas']:,} filas, "
                 f"columnas id,is_fraud. Verificación contra "
                 f"sample_submission.csv: {env['checks_ok']}/"
                 f"{env['checks_total']} checks OK "
                 f"({', '.join(env['checks_fallidos']) if env['checks_fallidos'] else 'sin fallos'}). "
                 f"Media de probabilidad predicha: {env['media_prob']:.4f} "
                 f"(base 0.0167). **NO enviado a Kaggle: el envío lo "
                 f"decide Elán después.**\n")

    # Gaps.
    L.append("## 8. Gaps (registrados, nada inventado)\n")
    L.append("- **Serie temporal:** sin serie temporal por entidad (ver §4).")
    L.append("- **Intermediario:** no medible (no hay intermediario identificado).")
    L.append("- **Distancia:** plana en estos datos (1.72% vs 1.66%); se "
             "mantiene en el conteo por definición, con la variante medida.")
    L.append("- **channel / card_type / device_type:** planos "
             "(fraude 1.46%-1.74% en todas las clases): no discriminan.")
    L.append("- **day_of_week / time_of_day_hour:** planos (1.58%-1.83%): "
             "no hay patrón horario ni semanal de fraude en estos datos.")
    if resultados.get("externo"):
        L.append("- **Datos externos:** el original CC0 se probó en §6 "
                 "(variante comparada con CV; veredicto ahí).\n")
    else:
        L.append("- **Datos externos:** el original CC0 no se descargó en "
                 "esta entrega (ver §6).\n")
    md = "\n".join(L)
    if dir_salida:
        os.makedirs(dir_salida, exist_ok=True)
        with open(os.path.join(dir_salida, "informe-fraude.md"), "w",
                  encoding="utf-8") as fh:
            fh.write(md)
    return md


def ejecutar(df_train, df_test=None, sample=None, dir_salida=None,
             df_externo=None):
    """Ejecuta la adaptación completa: conteo de señales, cuota de fraude,
    autenticación, diagnóstico temporal,
    comparación de modelos y (si hay test) envío.

    Devuelve el dict de resultados (tablas, resumen, comparación,
    envío). No envía nada a Kaggle.
    """
    t_senales, res_senales = tabla_conteo_senales(df_train)
    var_senales = tabla_conteo_senales_sin_distancia(df_train)
    t_cat = tabla_cuota_fraude(df_train, "merchant_category")
    t_ch = tabla_cuota_fraude(df_train, "channel")
    t_auth = tabla_auth(df_train)
    diag = diagnostico_temporal(df_train)
    resultados = {
        "tabla_conteo_senales": t_senales, "resumen_senales": res_senales,
        "conteo_senales_sin_distancia": var_senales,
        "tabla_cuota_fraude_categoria": t_cat, "tabla_cuota_fraude_channel": t_ch,
        "tabla_auth": t_auth, "diagnostico_temporal": diag,
        "comparacion_modelos": None, "externo": None,
        "externo_pendiente": df_externo is None, "envio": None,
    }
    if dir_salida:
        _figuras(dir_salida, t_senales, t_cat, t_auth)
    resultados["comparacion_modelos"] = comparar_modelos(df_train)
    if df_externo is not None:
        resultados["externo"] = evaluar_con_externo(df_train, df_externo)
    if df_test is not None and sample is not None:
        modelo, enc, ref = entrenar_final(df_train)
        probs = predecir_test(modelo, enc, ref, df_test)
        ruta = os.path.join(dir_salida, "submission.csv") if dir_salida else "submission.csv"
        sub = escribir_submission(df_test, probs, ruta)
        checks = verificar_submission(sub, sample)
        ok = [c for c in checks if c["ok"]]
        resultados["envio"] = {
            "ruta": ruta, "n_filas": int(len(sub)),
            "checks_ok": len(ok), "checks_total": len(checks),
            "checks_fallidos": [c["check"] for c in checks if not c["ok"]],
            "media_prob": round(float(sub["is_fraud"].mean()), 4),
        }
        resultados["checks_submission"] = checks
    return resultados
