#!/usr/bin/env python3
"""Ajuste ligero (afinación) del modelo de fraude con tarjeta — Kaggle S6E11.

Objetivo: subir el ROC AUC con un ajuste ligero, por orden de Elán.

ESTADO DE PARTIDA (medido, 5 pliegues, HistGradientBoosting 200 árboles,
lr 0.08, 31 hojas, min_samples_leaf 50, l2 1.0, semilla 42):
  crudas 0.9449 ± 0.0035 | señales 0.9075 ± 0.0054 | combinadas 0.9448 ± 0.0031.
Enviado a Kaggle 2026-11-08 ~17:03: público ROC AUC 0.94375 (1027/1142).
El externo CC0 se probó y se descartó (NO AYUDA).

NOTA DE CALIBRACIÓN (importante): calibrar las probabilidades NO mueve el
ROC AUC — el AUC es invariante a transformaciones monótonas del puntaje.
La calibración se mide aquí SOLO como diagnóstico (log-loss, Brier,
pendiente de calibración), nunca como mejora de AUC.

LO QUE SE PRUEBA (todo con la misma CV estratificada de 5 pliegues,
misma semilla 42, mismo protocolo que el informe; tablas de referencia
SOLO del pliegue de entrenamiento, sin fuga):
1. Rejilla pequeña de HistGradientBoosting: 10 configuraciones (base + 9),
   sobre el modo combinadas.
2. LightGBM (4.7.0, instalado por pip): UNA configuración comparable
   (200 árboles, lr 0.08, 31 hojas, min_child_samples 50, reg_lambda 1.0).
   No es AutoML: es un GBM directo, y las reglas prohíben AutoML, no GBMs.
3. Features seguros, cada uno por separado, sobre combinadas + config base:
   (a) ratio amount_usd / account_balance_usd;
   (b) amount_usd relativo a la media de su merchant_category (out-of-fold);
   (c) log1p(amount_usd) y log1p(velocity_score);
   (d) 5 factores de interacción (por orden de Elán):
       velocity_score x txn_count_last_24h;
       distance_from_home_km x is_foreign_transaction;
       amount_usd x merchant_risk_score;
       cvv_retry_count x prior_disputes;
       hours_since_last_txn x velocity_score.
   Todos son productos por fila de columnas crudas (sin fuga); solo (b)
   usa una media por categoría, calculada out-of-fold.
4. Desviación contra el patrón del GRUPO (refinamiento de Elán: los
   usuarios de tarjetas tienen patrones; las anomalías se notan como
   desviación del patrón normal). LÍMITE: no hay ID de usuario en el
   dataset (el id es el ID de la transacción), así que NO hay perfil por
   usuario: son desviaciones contra el patrón del grupo, todas
   out-of-fold: z-score de amount vs su categoría; z-score de amount vs
   (categoría x tipo de tarjeta); z-score de velocity vs su channel;
   z-score de distancia vs su tipo de tarjeta; z-score de horas desde la
   última vs su categoría; distancia circular de la hora a la hora típica
   (media circular) de su categoría.

REGLA DE DECISIÓN: regenera submission.csv SOLO si alguna variante supera
al mejor actual (crudas 0.9449) por más de su std entre pliegues (~0.0035).
Si ninguna supera, el submission actual queda intacto y todo lo probado
se documenta como descartado. Si alguna gana, se regenera con TODO train,
se re-verifica el formato 7/7 y se guarda el anterior como
salida-fraude/submission-anterior-0.94375.csv.

MEMORIA: la máquina está apretada (155 MB libres al arrancar este script;
una corrida anterior murió en silencio con 178 MB libres). Por eso las
matrices de features se pasan a float32 antes de ajustar y se liberan
con gc.collect() tras cada pliegue. La config base se corre con el mismo
protocolo para verificar que float32 no cambia el AUC (debe reproducir
0.9448 en combinadas).

Uso:
    python3 verificacion/ajuste_ligero.py
Escribe verificacion/ajuste_ligero.json (final) y
verificacion/ajuste_ligero_parcial.json (checkpoint tras cada variante).
No envía nada a Kaggle. No modifica los datos (solo lectura).
"""
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fraude_tarjetas as ft

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_DATOS = os.path.join(DIR, "datos")
DIR_SALIDA = os.path.join(DIR, "salida-fraude")
DIR_VERIF = os.path.join(DIR, "verificacion")
MEJOR_ACTUAL = 0.9449          # crudas, CV 5 pliegues
MEJOR_ACTUAL_MODO = "crudas"
UMBRAL_STD_REF = 0.0035         # std entre pliegues de referencia
SEED = 42

# ---------------------------------------------------------------------------
# Rejilla de HistGradientBoosting (10 configuraciones: base + 9)
# ---------------------------------------------------------------------------
GRID = [
    ("base", dict(max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
                  min_samples_leaf=50, l2_regularization=1.0)),
    ("g1_lr05_l31_ml20_l2.1_it300",
     dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
          min_samples_leaf=20, l2_regularization=0.1)),
    ("g2_lr05_l63_ml20_l2.1_it300",
     dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=63,
          min_samples_leaf=20, l2_regularization=0.1)),
    ("g3_lr12_l31_ml20_l2.0_it200",
     dict(max_iter=200, learning_rate=0.12, max_leaf_nodes=31,
          min_samples_leaf=20, l2_regularization=0.0)),
    ("g4_lr12_l63_ml10_l2.0_it200",
     dict(max_iter=200, learning_rate=0.12, max_leaf_nodes=63,
          min_samples_leaf=10, l2_regularization=0.0)),
    ("g5_lr08_l15_ml20_l2.1_it200",
     dict(max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
          min_samples_leaf=20, l2_regularization=0.1)),
    ("g6_lr08_l63_ml40_l2.1_it300",
     dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
          min_samples_leaf=40, l2_regularization=0.1)),
    ("g7_lr05_l15_ml40_l2.0_it200",
     dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
          min_samples_leaf=40, l2_regularization=0.0)),
    ("g8_lr12_l15_ml10_l2.1_it300",
     dict(max_iter=300, learning_rate=0.12, max_leaf_nodes=15,
          min_samples_leaf=10, l2_regularization=0.1)),
    ("g9_lr08_l31_ml10_l2.0_it300",
     dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
          min_samples_leaf=10, l2_regularization=0.0)),
]

# ---------------------------------------------------------------------------
# Features extra (cada uno por separado; sin fuga)
# ---------------------------------------------------------------------------
def _extra_ratio(d_tr, d_va):
    """(a) ratio amount_usd / account_balance_usd (por fila; saldo mínimo
    58.25 USD, sin ceros; el replace es solo seguro)."""
    def f(d):
        saldo = d["account_balance_usd"].replace(0, np.nan)
        return pd.DataFrame(
            {"f_ratio_monto_saldo": (d["amount_usd"] / saldo).fillna(0.0)},
            index=d.index)
    return f(d_tr), f(d_va)


def _extra_rel_categoria(d_tr, d_va):
    """(b) amount_usd relativo a la media de su merchant_category.
    La media se calcula SOLO con el pliegue de entrenamiento (out-of-fold);
    categorías no vistas usan la media global del pliegue."""
    medias = d_tr.groupby("merchant_category", observed=True)["amount_usd"].mean()
    g = float(d_tr["amount_usd"].mean())

    def f(d):
        m = d["merchant_category"].map(medias).astype(float).fillna(g)
        return pd.DataFrame({"f_monto_rel_categoria": d["amount_usd"] / m},
                            index=d.index)
    return f(d_tr), f(d_va)


def _extra_log(d_tr, d_va):
    """(c) log1p de amount_usd y velocity_score (por fila)."""
    def f(d):
        return pd.DataFrame({
            "f_log_monto": np.log1p(d["amount_usd"]),
            "f_log_velocidad": np.log1p(d["velocity_score"])}, index=d.index)
    return f(d_tr), f(d_va)


def _extra_inter_rafaga(d_tr, d_va):
    """(d1) velocity_score x txn_count_last_24h (ráfaga)."""
    def f(d):
        return pd.DataFrame(
            {"f_inter_velocidad_rafaga":
             d["velocity_score"] * d["txn_count_last_24h"]}, index=d.index)
    return f(d_tr), f(d_va)


def _extra_inter_dist_extranjero(d_tr, d_va):
    """(d2) distance_from_home_km x is_foreign_transaction."""
    def f(d):
        return pd.DataFrame(
            {"f_inter_distancia_extranjero":
             d["distance_from_home_km"] * d["is_foreign_transaction"].astype(int)},
            index=d.index)
    return f(d_tr), f(d_va)


def _extra_inter_monto_riesgo(d_tr, d_va):
    """(d3) amount_usd x merchant_risk_score."""
    def f(d):
        return pd.DataFrame(
            {"f_inter_monto_riesgo":
             d["amount_usd"] * d["merchant_risk_score"]}, index=d.index)
    return f(d_tr), f(d_va)


def _extra_inter_cvv_disputas(d_tr, d_va):
    """(d4) cvv_retry_count x prior_disputes."""
    def f(d):
        return pd.DataFrame(
            {"f_inter_cvv_disputas":
             d["cvv_retry_count"] * d["prior_disputes"]}, index=d.index)
    return f(d_tr), f(d_va)


def _extra_inter_horas_velocidad(d_tr, d_va):
    """(d5) hours_since_last_txn x velocity_score."""
    def f(d):
        return pd.DataFrame(
            {"f_inter_horas_velocidad":
             d["hours_since_last_txn"] * d["velocity_score"]}, index=d.index)
    return f(d_tr), f(d_va)


# ---------------------------------------------------------------------------
# Desviación contra el patrón del GRUPO (por orden de Elán: los usuarios
# de tarjetas tienen patrones; las anomalías se notan como desviación
# del patrón normal, no solo como valor crudo).
#
# LÍMITE DEL DATASET (registrado, no inventado): NO hay columna de ID de
# usuario/cliente — el id es el ID de la transacción, no temporal y no
# agrupa por usuario — así que NO se puede construir un perfil por
# usuario. Lo más cercano que el dataset permite es la desviación contra
# el patrón del GRUPO (categoría, canal, tipo de tarjeta). Todas las
# estadísticas del grupo se calculan SOLO con el pliegue de entrenamiento
# (out-of-fold); nunca con validación ni con el test.
# ---------------------------------------------------------------------------
def _zscore_por_grupo(d_tr, d_va, col_grupo, col_valor, nombre):
    """Z-score de col_valor contra su grupo: (x - media_grupo) / std_grupo,
    con media/std del pliegue de entrenamiento. Grupos no vistos o con
    std 0 usan la media/std global del pliegue."""
    g = d_tr.groupby(col_grupo, observed=True)[col_valor].agg(["mean", "std"])
    gm, gs = float(d_tr[col_valor].mean()), float(d_tr[col_valor].std())

    def f(d):
        med = d[col_grupo].map(g["mean"]).astype(float).fillna(gm)
        desv = d[col_grupo].map(g["std"]).astype(float).fillna(gs).replace(0, gs)
        return pd.DataFrame({nombre: (d[col_valor] - med) / desv},
                            index=d.index)
    return f(d_tr), f(d_va)


def _zscore_por_grupo_doble(d_tr, d_va, col_g1, col_g2, col_valor, nombre):
    """Z-score contra el grupo (col_g1 x col_g2). Fallback en cascada:
    grupo doble -> grupo simple col_g1 -> global."""
    clave_tr = (d_tr[col_g1].astype(str) + "|" + d_tr[col_g2].astype(str))
    g2 = d_tr.groupby(clave_tr, observed=True)[col_valor].agg(["mean", "std"])
    g1 = d_tr.groupby(col_g1, observed=True)[col_valor].agg(["mean", "std"])
    gm, gs = float(d_tr[col_valor].mean()), float(d_tr[col_valor].std())

    def f(d):
        clave = d[col_g1].astype(str) + "|" + d[col_g2].astype(str)
        med = clave.map(g2["mean"]).astype(float)
        desv = clave.map(g2["std"]).astype(float)
        med = med.fillna(d[col_g1].map(g1["mean"]).astype(float)).fillna(gm)
        desv = desv.fillna(d[col_g1].map(g1["std"]).astype(float)).fillna(gs).replace(0, gs)
        return pd.DataFrame({nombre: (d[col_valor] - med) / desv},
                            index=d.index)
    return f(d_tr), f(d_va)


def _hora_circular_media(horas):
    """Hora típica circular (media circular, no aritmética: la media
    aritmética de horas cruza mal la medianoche)."""
    ang = 2 * np.pi * np.asarray(horas, dtype=float) / 24.0
    m = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    return float(m * 24.0 / (2 * np.pi)) % 24.0


def _extra_dev_hora_circular(d_tr, d_va):
    """(dev6) Distancia circular de time_of_day_hour a la hora típica de
    su merchant_category: min(|h - h_tipica|, 24 - |h - h_tipica|).
    La hora típica es la MEDIA CIRCULAR de la categoría en el pliegue
    de entrenamiento (la media aritmética cruza mal la medianoche);
    categorías no vistas usan la media circular global."""
    h_tip = d_tr.groupby("merchant_category", observed=True)[
        "time_of_day_hour"].apply(_hora_circular_media)
    h_glob = _hora_circular_media(d_tr["time_of_day_hour"])

    def f(d):
        ht = d["merchant_category"].map(h_tip).astype(float).fillna(h_glob)
        dist = (d["time_of_day_hour"] - ht).abs()
        dist = np.minimum(dist, 24.0 - dist)
        return pd.DataFrame({"f_dev_hora_circular": dist}, index=d.index)
    return f(d_tr), f(d_va)


DESVIACIONES = [
    ("z_monto_categoria",
     "z-score de amount_usd contra media/std de su merchant_category "
     "(out-of-fold; además del ratio ya probado)",
     lambda tr, va: _zscore_por_grupo(
         tr, va, "merchant_category", "amount_usd", "f_z_monto_categoria")),
    ("dev_monto_cat_tarjeta",
     "z-score de amount_usd contra media/std de su "
     "(merchant_category x card_type) (out-of-fold)",
     lambda tr, va: _zscore_por_grupo_doble(
         tr, va, "merchant_category", "card_type", "amount_usd",
         "f_dev_monto_cat_tarjeta")),
    ("dev_velocidad_canal",
     "z-score de velocity_score contra media/std de su channel "
     "(out-of-fold)",
     lambda tr, va: _zscore_por_grupo(
         tr, va, "channel", "velocity_score", "f_dev_velocidad_canal")),
    ("dev_distancia_tarjeta",
     "z-score de distance_from_home_km contra media/std de su card_type "
     "(out-of-fold; se usa la media por tipo de tarjeta, no la mediana "
     "global)",
     lambda tr, va: _zscore_por_grupo(
         tr, va, "card_type", "distance_from_home_km",
         "f_dev_distancia_tarjeta")),
    ("dev_horas_categoria",
     "z-score de hours_since_last_txn contra media/std de su "
     "merchant_category (out-of-fold)",
     lambda tr, va: _zscore_por_grupo(
         tr, va, "merchant_category", "hours_since_last_txn",
         "f_dev_horas_categoria")),
    ("dev_hora_circular",
     "distancia circular de time_of_day_hour a la hora típica (media "
     "circular) de su merchant_category (out-of-fold)",
     _extra_dev_hora_circular),
]


EXTRAS = [
    ("ratio_monto_saldo", "amount_usd / account_balance_usd", _extra_ratio),
    ("monto_rel_categoria",
     "amount_usd / media de amount_usd de su merchant_category (out-of-fold)",
     _extra_rel_categoria),
    ("log_monto_velocidad", "log1p(amount_usd), log1p(velocity_score)", _extra_log),
    ("inter_velocidad_rafaga",
     "velocity_score x txn_count_last_24h (ráfaga)", _extra_inter_rafaga),
    ("inter_distancia_extranjero",
     "distance_from_home_km x is_foreign_transaction", _extra_inter_dist_extranjero),
    ("inter_monto_riesgo",
     "amount_usd x merchant_risk_score", _extra_inter_monto_riesgo),
    ("inter_cvv_disputas",
     "cvv_retry_count x prior_disputes", _extra_inter_cvv_disputas),
    ("inter_horas_velocidad",
     "hours_since_last_txn x velocity_score", _extra_inter_horas_velocidad),
]


# ---------------------------------------------------------------------------
# CV genérica (mismo protocolo que el informe)
# ---------------------------------------------------------------------------
def cv_generico(df_train, modo, fabricar_modelo, extra=None, seed=SEED,
                oof=False):
    """CV estratificada de 5 pliegues. Tablas de referencia SOLO del
    pliegue de entrenamiento; `extra` es (d_tr, d_va) -> (Xe_tr, Xe_va).
    Las matrices se pasan a float32 antes de ajustar (memoria) y se
    liberan tras cada pliegue. Devuelve dict con aucs, media, std y,
    si oof=True, las predicciones out-of-fold."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    y = df_train[ft.TARGET].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs, oof_p, oof_y = [], [], []
    for i, (tr_idx, va_idx) in enumerate(skf.split(df_train, y)):
        d_tr, d_va = df_train.iloc[tr_idx], df_train.iloc[va_idx]
        ref = ft._tablas_referencia(d_tr)
        X_tr, enc = ft.construir_features(d_tr, modo, ref, fit_encoder=True)
        X_va, _ = ft.construir_features(d_va, modo, ref, encoder=enc)
        if extra is not None:
            Xe_tr, Xe_va = extra(d_tr, d_va)
            X_tr = pd.concat([X_tr, Xe_tr], axis=1)
            X_va = pd.concat([X_va, Xe_va], axis=1)
        X_tr = X_tr.astype(np.float32)
        X_va = X_va.astype(np.float32)
        modelo = fabricar_modelo()
        modelo.fit(X_tr, y[tr_idx])
        p = modelo.predict_proba(X_va)[:, 1]
        auc = float(roc_auc_score(y[va_idx], p))
        aucs.append(round(auc, 4))
        if oof:
            oof_p.append(p)
            oof_y.append(y[va_idx])
        print(f"    pliegue {i + 1}/5: AUC = {auc:.4f}", flush=True)
        del X_tr, X_va, modelo, p
        gc.collect()
    out = {"aucs": aucs,
           "auc_media": round(float(np.mean(aucs)), 4),
           "auc_std": round(float(np.std(aucs)), 4)}
    if oof:
        out["oof_p"] = np.concatenate(oof_p)
        out["oof_y"] = np.concatenate(oof_y)
    return out


def _hgb_params(params):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(random_state=SEED, **params)


def _lgbm():
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="binary", n_estimators=200,
                              learning_rate=0.08, num_leaves=31,
                              min_child_samples=50, reg_lambda=1.0,
                              random_state=SEED, verbose=-1, n_jobs=2)


def _mem_disponible_mb():
    try:
        with open("/proc/meminfo") as fh:
            for linea in fh:
                if linea.startswith("MemAvailable:"):
                    return int(linea.split()[1]) // 1024
    except Exception:
        return None
    return None


def _checkpoint(resultados, ruta_parcial):
    with open(ruta_parcial, "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, indent=1, ensure_ascii=False,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def diagnostico_calibracion(oof_p, oof_y):
    """Diagnóstico de calibración de probabilidades (NO mueve el AUC:
    el AUC es invariante a transformaciones monótonas del puntaje).
    Se reporta solo como diagnóstico."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss
    p = np.asarray(oof_p, dtype=float)
    y = np.asarray(oof_y, dtype=int)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(pc / (1 - pc))
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(logit.reshape(-1, 1), y)
    return {
        "log_loss": round(float(log_loss(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "media_predicha": round(float(p.mean()), 4),
        "media_observada": round(float(y.mean()), 4),
        "pendiente_calibracion": round(float(lr.coef_[0][0]), 4),
        "intercepto_calibracion": round(float(lr.intercept_[0]), 4),
        "nota": "Diagnóstico únicamente: calibrar las probabilidades NO "
                "mueve el ROC AUC (invariante a transformaciones monótonas "
                "del puntaje). Pendiente 1.0 e intercepto 0.0 = calibrado.",
    }


def main():
    t0 = time.time()
    print("== Ajuste ligero — Kaggle S6E11 fraude con tarjeta ==", flush=True)
    import sklearn
    print(f"sklearn {sklearn.__version__}", flush=True)
    try:
        import lightgbm as _lgb
        print(f"lightgbm {_lgb.__version__} (instalado por pip 2026-11-08)",
              flush=True)
    except Exception as e:
        print(f"lightgbm NO disponible: {e}", flush=True)
    print(f"memoria disponible al inicio: {_mem_disponible_mb()} MB",
          flush=True)

    train = pd.read_csv(os.path.join(DIR_DATOS, "train.csv"))
    test = pd.read_csv(os.path.join(DIR_DATOS, "test.csv"))
    sample = pd.read_csv(os.path.join(DIR_DATOS, "sample_submission.csv"))
    # NOTA: no se convierten las categóricas a dtype 'category': el módulo
    # usa .map(...).fillna(...) sobre columnas object y con 'category' eso
    # falla (TypeError al rellenar con un valor fuera de las categorías).
    # Se mantiene el mismo dtype que en las corridas del informe.
    print(f"train {train.shape}, test {test.shape}, sample {sample.shape}",
          flush=True)

    ruta_parcial = os.path.join(DIR_VERIF, "ajuste_ligero_parcial.json")
    ruta_final = os.path.join(DIR_VERIF, "ajuste_ligero.json")
    res = {
        "mejor_actual": MEJOR_ACTUAL,
        "mejor_actual_modo": MEJOR_ACTUAL_MODO,
        "umbral_std_ref": UMBRAL_STD_REF,
        "semilla": SEED,
        "rejilla_hgb": {},
        "lightgbm": None,
        "features_extra": {},
        "desviaciones": {},
        "mejor_config_crudas": None,
        "calibracion_diagnostico": None,
        "decision": None,
        "envio": None,
    }
    # Reanudar desde el checkpoint si existe: la corrida anterior murió
    # con el reinicio de la máquina (a mitad de g2) y lo ya medido no se
    # repite. El checkpoint se escribe tras cada variante.
    if os.path.exists(ruta_parcial):
        try:
            with open(ruta_parcial, encoding="utf-8") as fh:
                previo = json.load(fh)
            for clave in ("rejilla_hgb", "features_extra", "desviaciones",
                          "lightgbm", "calibracion_diagnostico",
                          "mejor_config_crudas"):
                if previo.get(clave):
                    res[clave] = previo[clave]
            print(f"checkpoint cargado: rejilla "
                  f"{list(res['rejilla_hgb'])}, features "
                  f"{list(res['features_extra'])}, desviaciones "
                  f"{list(res['desviaciones'])}, lightgbm "
                  f"{'sí' if res['lightgbm'] else 'no'}", flush=True)
        except Exception as e:
            print(f"checkpoint ilegible, se empieza de cero: {e}", flush=True)

    # 1. Rejilla HGB sobre combinadas (base + 9).
    print("\n== 1. Rejilla HistGradientBoosting (modo combinadas) ==",
          flush=True)
    for nombre, params in GRID:
        if nombre in res["rejilla_hgb"]:
            r = res["rejilla_hgb"][nombre]
            print(f"-- {nombre}: YA MEDIDO en checkpoint "
                  f"({r['auc_media']:.4f} ± {r['auc_std']:.4f}), se salta",
                  flush=True)
            continue
        print(f"-- {nombre}: {params} "
              f"(mem {_mem_disponible_mb()} MB)", flush=True)
        r = cv_generico(train, "combinadas",
                        lambda p=params: _hgb_params(p),
                        oof=(nombre == "base"))
        r["params"] = params
        if nombre == "base":
            res["calibracion_diagnostico"] = diagnostico_calibracion(
                r.pop("oof_p"), r.pop("oof_y"))
            print(f"   calibración (diagnóstico, no mueve AUC): "
                  f"{res['calibracion_diagnostico']}", flush=True)
        res["rejilla_hgb"][nombre] = r
        print(f"   {nombre}: {r['auc_media']:.4f} ± {r['auc_std']:.4f} "
              f"{r['aucs']}", flush=True)
        _checkpoint(res, ruta_parcial)

    # 2. LightGBM, una configuración comparable, modo combinadas.
    print("\n== 2. LightGBM (modo combinadas) ==", flush=True)
    if isinstance(res["lightgbm"], dict) and "auc_media" in res["lightgbm"]:
        print(f"   lightgbm: YA MEDIDO en checkpoint "
              f"({res['lightgbm'].get('auc_media')} ± "
              f"{res['lightgbm'].get('auc_std')}), se salta", flush=True)
    else:
        try:
            r = cv_generico(train, "combinadas", _lgbm)
            r["params"] = {"n_estimators": 200, "learning_rate": 0.08,
                           "num_leaves": 31, "min_child_samples": 50,
                           "reg_lambda": 1.0}
            res["lightgbm"] = r
            print(f"   lightgbm: {r['auc_media']:.4f} ± {r['auc_std']:.4f} "
                  f"{r['aucs']}", flush=True)
        except Exception as e:
            res["lightgbm"] = {"error": str(e)}
            print(f"   lightgbm FALLÓ: {e}", flush=True)
        _checkpoint(res, ruta_parcial)

    # 3. Features extra, cada uno por separado, config base, combinadas.
    print("\n== 3. Features extra (config base, modo combinadas) ==",
          flush=True)
    for nombre, descripcion, fn in EXTRAS:
        if nombre in res["features_extra"]:
            r = res["features_extra"][nombre]
            print(f"-- {nombre}: YA MEDIDO en checkpoint "
                  f"({r['auc_media']:.4f} ± {r['auc_std']:.4f}), se salta",
                  flush=True)
            continue
        print(f"-- {nombre}: {descripcion} "
              f"(mem {_mem_disponible_mb()} MB)", flush=True)
        r = cv_generico(train, "combinadas", ft._hgb, extra=fn)
        r["descripcion"] = descripcion
        res["features_extra"][nombre] = r
        print(f"   {nombre}: {r['auc_media']:.4f} ± {r['auc_std']:.4f} "
              f"{r['aucs']}", flush=True)
        _checkpoint(res, ruta_parcial)

    # 3b. Desviaciones contra el patrón del grupo, cada una por separado,
    # config base, combinadas. Sin ID de usuario no hay perfil por usuario.
    print("\n== 3b. Desviación contra el patrón del grupo (config base, "
          "modo combinadas) ==", flush=True)
    for nombre, descripcion, fn in DESVIACIONES:
        if nombre in res["desviaciones"]:
            r = res["desviaciones"][nombre]
            print(f"-- {nombre}: YA MEDIDO en checkpoint "
                  f"({r['auc_media']:.4f} ± {r['auc_std']:.4f}), se salta",
                  flush=True)
            continue
        print(f"-- {nombre}: {descripcion} "
              f"(mem {_mem_disponible_mb()} MB)", flush=True)
        r = cv_generico(train, "combinadas", ft._hgb, extra=fn)
        r["descripcion"] = descripcion
        res["desviaciones"][nombre] = r
        print(f"   {nombre}: {r['auc_media']:.4f} ± {r['auc_std']:.4f} "
              f"{r['aucs']}", flush=True)
        _checkpoint(res, ruta_parcial)

    # 4. Mejor config de la rejilla, también en modo crudas.
    mejor_grid = max(res["rejilla_hgb"].items(),
                     key=lambda kv: kv[1]["auc_media"])
    if res.get("mejor_config_crudas") is not None:
        r = res["mejor_config_crudas"]
        print(f"\n== 4. Mejor config en modo crudas: YA MEDIDO en "
              f"checkpoint ({r['auc_media']:.4f} ± {r['auc_std']:.4f}), "
              f"se salta ==", flush=True)
    else:
        print(f"\n== 4. Mejor config de la rejilla en modo crudas "
              f"({mejor_grid[0]}) ==", flush=True)
        r = cv_generico(train, "crudas",
                        lambda p=mejor_grid[1]["params"]: _hgb_params(p))
        r["params"] = mejor_grid[1]["params"]
        r["config_origen"] = mejor_grid[0]
        res["mejor_config_crudas"] = r
        print(f"   {mejor_grid[0]} en crudas: {r['auc_media']:.4f} ± "
              f"{r['auc_std']:.4f} {r['aucs']}", flush=True)
        _checkpoint(res, ruta_parcial)

    # 5. Decisión.
    print("\n== 5. Decisión ==", flush=True)
    candidatos = []
    for nombre, r in res["rejilla_hgb"].items():
        candidatos.append((f"hgb/{nombre}/combinadas", r["auc_media"],
                           r["auc_std"],
                           {"tipo": "hgb", "modo": "combinadas",
                            "params": r["params"]}))
    if res["lightgbm"] and "auc_media" in res["lightgbm"]:
        r = res["lightgbm"]
        candidatos.append(("lightgbm/combinadas", r["auc_media"],
                           r["auc_std"],
                           {"tipo": "lightgbm", "modo": "combinadas",
                            "params": r["params"]}))
    for nombre, r in res["features_extra"].items():
        candidatos.append((f"feature/{nombre}/combinadas+base", r["auc_media"],
                           r["auc_std"],
                           {"tipo": "feature", "modo": "combinadas",
                            "feature": nombre, "params": "base"}))
    for nombre, r in res["desviaciones"].items():
        candidatos.append((f"desviacion/{nombre}/combinadas+base",
                           r["auc_media"], r["auc_std"],
                           {"tipo": "feature", "modo": "combinadas",
                            "feature": nombre, "params": "base"}))
    if res["mejor_config_crudas"]:
        r = res["mejor_config_crudas"]
        candidatos.append((f"hgb/{r['config_origen']}/crudas", r["auc_media"],
                           r["auc_std"],
                           {"tipo": "hgb", "modo": "crudas",
                            "params": r["params"]}))
    ganadores = []
    for nombre, media, std, receta in candidatos:
        delta = round(media - MEJOR_ACTUAL, 4)
        gana = delta > std
        print(f"   {nombre}: {media:.4f} ± {std:.4f} | Δ vs {MEJOR_ACTUAL}: "
              f"{delta:+.4f} | {'GANA' if gana else 'no gana'} (std {std})",
              flush=True)
        if gana:
            ganadores.append((media, nombre, receta))
    if ganadores:
        ganadores.sort(reverse=True)
        media, nombre, receta = ganadores[0]
        res["decision"] = {
            "veredicto": "GANA",
            "variante": nombre,
            "auc_media": media,
            "receta": receta,
            "nota": f"Supera al mejor actual ({MEJOR_ACTUAL}) por más de "
                    f"la std entre pliegues. El envío se regenera.",
        }
    else:
        res["decision"] = {
            "veredicto": "NO GANA NADA",
            "variante": None,
            "nota": f"Ninguna variante supera al mejor actual "
                    f"({MEJOR_ACTUAL}) por más de la std entre pliegues "
                    f"(~{UMBRAL_STD_REF}). El submission actual queda "
                    f"intacto; todo lo probado se documenta como descartado.",
        }
    print(f"   VEREDICTO: {res['decision']['veredicto']} — "
          f"{res['decision'].get('variante') or res['decision']['nota']}",
          flush=True)

    # 6. Regenerar el envío solo si algo gana.
    if res["decision"]["veredicto"] == "GANA":
        receta = res["decision"]["receta"]
        ruta_vieja = os.path.join(DIR_SALIDA, "submission.csv")
        ruta_backup = os.path.join(
            DIR_SALIDA, "submission-anterior-0.94375.csv")
        import hashlib
        with open(ruta_vieja, "rb") as fh:
            md5_antes = hashlib.md5(fh.read()).hexdigest()
        os.replace(ruta_vieja, ruta_backup)
        print(f"   submission anterior guardado en {ruta_backup} "
              f"(md5 {md5_antes})", flush=True)
        ref = ft._tablas_referencia(train)
        if receta["tipo"] == "hgb":
            X, enc = ft.construir_features(train, receta["modo"], ref,
                                           fit_encoder=True)
            modelo = _hgb_params(receta["params"])
            modelo.fit(X.astype(np.float32), train[ft.TARGET].values)
            X_te, _ = ft.construir_features(test, receta["modo"], ref,
                                            encoder=enc)
            probs = modelo.predict_proba(X_te.astype(np.float32))[:, 1]
        elif receta["tipo"] == "lightgbm":
            X, enc = ft.construir_features(train, receta["modo"], ref,
                                           fit_encoder=True)
            modelo = _lgbm()
            modelo.fit(X.astype(np.float32), train[ft.TARGET].values)
            X_te, _ = ft.construir_features(test, receta["modo"], ref,
                                            encoder=enc)
            probs = modelo.predict_proba(X_te.astype(np.float32))[:, 1]
        else:  # feature (de EXTRAS o de DESVIACIONES)
            todas = dict((n, f) for n, _d, f in EXTRAS + DESVIACIONES)
            fn = todas[receta["feature"]]
            X, enc = ft.construir_features(train, "combinadas", ref,
                                           fit_encoder=True)
            Xe_tr, Xe_te = fn(train, test)
            X = pd.concat([X, Xe_tr], axis=1)
            modelo = ft._hgb()
            modelo.fit(X.astype(np.float32), train[ft.TARGET].values)
            X_te, _ = ft.construir_features(test, "combinadas", ref,
                                            encoder=enc)
            X_te = pd.concat([X_te, Xe_te], axis=1)
            probs = modelo.predict_proba(X_te.astype(np.float32))[:, 1]
        sub = ft.escribir_submission(test, probs, ruta_vieja)
        checks = ft.verificar_submission(sub, sample)
        ok = [c for c in checks if c["ok"]]
        with open(ruta_vieja, "rb") as fh:
            md5_despues = hashlib.md5(fh.read()).hexdigest()
        res["envio"] = {
            "regenerado": True,
            "md5_antes": md5_antes,
            "md5_despues": md5_despues,
            "checks_ok": len(ok), "checks_total": len(checks),
            "checks_fallidos": [c["check"] for c in checks if not c["ok"]],
            "media_prob": round(float(sub["is_fraud"].mean()), 4),
        }
        print(f"   envío regenerado: md5 {md5_despues}, "
              f"{len(ok)}/{len(checks)} checks OK", flush=True)
    else:
        import hashlib
        with open(os.path.join(DIR_SALIDA, "submission.csv"), "rb") as fh:
            md5 = hashlib.md5(fh.read()).hexdigest()
        res["envio"] = {"regenerado": False, "md5_actual": md5,
                        "nota": "Submission intacto (no se regeneró)."}
        print(f"   submission intacto (md5 {md5})", flush=True)

    res["tiempo_s"] = round(time.time() - t0, 1)
    with open(ruta_final, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"\nTIEMPO TOTAL: {res['tiempo_s']}s", flush=True)
    print("AJUSTE LIGERO COMPLETADO", flush=True)
    return res


if __name__ == "__main__":
    main()
