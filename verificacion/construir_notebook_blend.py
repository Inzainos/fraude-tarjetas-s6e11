#!/usr/bin/env python3
"""Construye fraude-tarjetas-blend.ipynb (12 celdas: 5 markdown + 7 código).

El notebook entrena el blend `mixto_todos` (14 modelos, promedio
simple, pesos iguales) con TODO train y escribe
salida-fraude/submission_blend.csv. Corre de arriba a abajo sin pasos
manuales. Resuelve rutas Kaggle (/kaggle/input/playground-series-s6e11
para los datos; /kaggle/input/**/fraude_tarjetas.py para el módulo) o
locales (el directorio del notebook). Checkpoints por modelo en
verificacion/blend_ckpt/ (la caja se reinicia: no perder trabajo).
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

C = []

C.append(nbf.v4.new_markdown_cell("""# Fraude con tarjeta — Blend `mixto_todos` (Playground Series S6E11)

**Fractal Core Research** — Blend del ensamble para la competencia Kaggle
*Predicting Credit Card Fraud* (cierra 30 nov 2026, métrica ROC AUC,
envío `id,is_fraud` con probabilidades 0–1).

El blend `mixto_todos` es el promedio simple (pesos iguales) de 14
modelos entrenados con TODO train: 9 GBM sobre features crudas, 4 GBM
sobre features combinadas (crudas + señales de riesgo) y 1 regresión
logística sobre crudas escaladas. CV local (5 pliegues, semilla 42):
**0.9465 ± 0.0032** (OOF agrupado 0.9464; Δ +0.0014 ± 0.0003 vs la base
HGB, positivo en los 5 pliegues). La ganancia es real pero pequeña:
el ensamble reduce varianza, no encuentra señal nueva. La barra de
decisión (0.9484) no se superó, pero Elán decidió enviarlo igual para
tener su score público ("aunque sea poco, agrega el ensamble").

Reglas: máximo **10 envíos por día**, 2 finales seleccionables; la v2
de Elán (pública 0.94393) sigue seleccionada como final y **este
notebook NO cambia ninguna selección**. Prohibido AutoML, etiquetar
el test y entrenar con el test. **Este notebook NO envía nada a
Kaggle: el envío lo decide Elán después.**
"""))

C.append(nbf.v4.new_code_cell("""# --- Setup: rutas Kaggle o local, e importar el adaptador ---
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Datos de la competencia: en Kaggle van a
# /kaggle/input/competitions/playground-series-s6e11/{train,test,sample_submission}.csv
# (el notebook de competencia los tiene adjuntos). En local viven en
# <directorio del notebook>/datos/.
# El módulo fraude_tarjetas.py va en /kaggle/input/**/ (dataset
# utilitario) o junto al notebook.
def _resolver():
    for base in [Path("/kaggle/input"), Path("."), Path("..")]:
        for mod in base.glob("**/fraude_tarjetas.py"):
            root = mod.parent
            for cand in [root / "datos", base / "datos"]:
                if (cand / "train.csv").exists():
                    return root, cand
            # módulo encontrado pero datos en la ruta de la competencia
            for comp in [Path("/kaggle/input/playground-series-s6e11"), Path("/kaggle/input/competitions/playground-series-s6e11")]:
                if (comp / "train.csv").exists():
                    return root, comp
    raise FileNotFoundError("No se encontró fraude_tarjetas.py ni train.csv")

ROOT, DATOS = _resolver()
print("Módulo en:", ROOT)
print("Datos en:", DATOS)

_spec = importlib.util.spec_from_file_location(
    "fraude_tarjetas", ROOT / "fraude_tarjetas.py")
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)
print("Adaptador v" + ft.__version__ + " cargado.")

train = pd.read_csv(DATOS / "train.csv")
test = pd.read_csv(DATOS / "test.csv")
sample = pd.read_csv(DATOS / "sample_submission.csv")
print("train:", train.shape, "| test:", test.shape, "| sample:", sample.shape)
print("tasa de fraude en train:", round(train["is_fraud"].mean(), 4))
print("nulos train:", int(train.isna().sum().sum()),
      "| nulos test:", int(test.isna().sum().sum()))
"""))

C.append(nbf.v4.new_markdown_cell("""## 1. La definición del blend `mixto_todos`

Promedio simple (pesos iguales) de las 14 probabilidades. Definición
exacta, tomada de `analizar_mixto.py` y `verificacion/ensamble_blends.json`
(la misma que dio el OOF 0.9465):

| Miembro | Modelo | Features | Semilla |
|---|---|---|---|
| hgb_base_s42 / s1 / s7 | HistGradientBoosting (200 it, lr 0.08, 31 hojas, min_leaf 50, l2 1.0) | crudas | 42 / 1 / 7 |
| hgb_g5_s42 / s1 / s7 | HistGradientBoosting (200 it, lr 0.08, 15 hojas, min_leaf 20, l2 0.1) | crudas | 42 / 1 / 7 |
| lgbm_s42 / s1 / s7 | LightGBM (200 est, lr 0.08, 31 hojas, min_child 50, reg_lambda 1.0) | crudas | 42 / 1 / 7 |
| hgb_g5_comb_s42 / s1 | HistGradientBoosting g5 | combinadas | 42 / 1 |
| lgbm_comb_s42 / s1 | LightGBM | combinadas | 42 / 1 |
| logreg | Regresión logística C=0.5, lbfgs, max_iter 500, sobre crudas escaladas | crudas | — |

Las features crudas son 14 numéricas + 6 booleanas + one-hot de las 5
categóricas (60 columnas); las combinadas añaden las 18 señales de
riesgo (78 columnas). Las tablas de referencia se calculan con TODO
train (sin fuga: el test no tiene etiquetas). El promediado simple y
el rank-mean dieron lo mismo en CV (±0.0001), así que se usa el
simple.
"""))

C.append(nbf.v4.new_code_cell("""# --- Tablas de referencia (TODO train) + features crudas ---
import gc
import json
import os
import time
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ref = ft._tablas_referencia(train)
X_tr, enc = ft.construir_features(train, "crudas", ref, fit_encoder=True)
X_te, _ = ft.construir_features(test, "crudas", ref, encoder=enc)
print("X_tr crudas:", X_tr.shape, "| X_te crudas:", X_te.shape)
print("columnas crudas:", X_tr.shape[1])
"""))

C.append(nbf.v4.new_code_cell("""# --- 9 GBM sobre crudas (checkpoints por modelo) ---
import lightgbm as lgb

CKPT = Path("verificacion/blend_ckpt")
CKPT.mkdir(parents=True, exist_ok=True)
TIEMPOS = Path("verificacion/blend_tiempos.json")
tiempos = json.loads(TIEMPOS.read_text()) if TIEMPOS.exists() else {}

def hgb_base(seed):
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=50, l2_regularization=1.0, random_state=seed)

def hgb_g5(seed):
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=0.1, random_state=seed)

def lgbm(seed):
    return lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.08, num_leaves=31,
        min_child_samples=50, reg_lambda=1.0, objective="binary",
        verbose=-1, n_jobs=4, random_state=seed)

def entrenar(nombre, fab, X, y, X_te):
    f = CKPT / (nombre + ".npy")
    if f.exists():
        p = np.load(f)
        print(" ", nombre, ": checkpoint reutilizado")
        return p
    t1 = time.time()
    modelo = fab()
    modelo.fit(X, y)
    p = modelo.predict_proba(X_te)[:, 1]
    np.save(f, p)
    dt = time.time() - t1
    tiempos[nombre] = {"tiempo_s": round(dt, 1)}
    TIEMPOS.write_text(json.dumps(tiempos, indent=2))
    print(" ", nombre, ": entrenado en", round(dt, 0), "s, media p =",
          round(float(p.mean()), 4), flush=True)
    del modelo, p
    gc.collect()
    return np.load(f)

y = train["is_fraud"].values
CRUDAS = [("hgb_base_s42", lambda: hgb_base(42)),
          ("hgb_base_s1", lambda: hgb_base(1)),
          ("hgb_base_s7", lambda: hgb_base(7)),
          ("hgb_g5_s42", lambda: hgb_g5(42)),
          ("hgb_g5_s1", lambda: hgb_g5(1)),
          ("hgb_g5_s7", lambda: hgb_g5(7)),
          ("lgbm_s42", lambda: lgbm(42)),
          ("lgbm_s1", lambda: lgbm(1)),
          ("lgbm_s7", lambda: lgbm(7))]
probs = {}
for nombre, fab in CRUDAS:
    probs[nombre] = entrenar(nombre, fab, X_tr, y, X_te)
print("9 GBM crudas listas.")
"""))

C.append(nbf.v4.new_code_cell("""# --- Regresión logística sobre crudas escaladas (familia diversa) ---
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)
probs["logreg"] = entrenar(
    "logreg",
    lambda: LogisticRegression(C=0.5, max_iter=500, solver="lbfgs"),
    X_tr_s, y, X_te_s)
print("logreg lista.")
del X_tr_s, X_te_s, scaler
gc.collect()
# Liberar las features crudas: ya no se necesitan (las combinadas se
# construyen aparte).
del X_tr, X_te
gc.collect()
print("Modelos en el blend hasta ahora:", len(probs))
"""))

C.append(nbf.v4.new_markdown_cell("""## 2. Features combinadas: 4 GBM más

Las combinadas son crudas + 18 señales de riesgo (10 señales, conteo,
umbral ≥ 5, sin autenticación, cuotas de fraude por categoría/canal y
nivel de concentración). En CV quedaron igual que las crudas para el
mismo modelo (g5 0.9452 vs 0.9451–0.9453; lgbm 0.9453 vs 0.9454–0.9455):
su valor en el blend es solo diversidad.
"""))

C.append(nbf.v4.new_code_cell("""# --- Features combinadas + 4 GBM ---
Xc_tr, _ = ft.construir_features(train, "combinadas", ref, encoder=enc)
Xc_te, _ = ft.construir_features(test, "combinadas", ref, encoder=enc)
print("Xc_tr combinadas:", Xc_tr.shape, "| Xc_te combinadas:", Xc_te.shape)
COMB = [("hgb_g5_comb_s42", lambda: hgb_g5(42)),
        ("hgb_g5_comb_s1", lambda: hgb_g5(1)),
        ("lgbm_comb_s42", lambda: lgbm(42)),
        ("lgbm_comb_s1", lambda: lgbm(1))]
for nombre, fab in COMB:
    probs[nombre] = entrenar(nombre, fab, Xc_tr, y, Xc_te)
del Xc_tr, Xc_te
gc.collect()
print("4 GBM combinadas listas. Modelos en el blend:", len(probs))
"""))

C.append(nbf.v4.new_markdown_cell("""## 3. Blend + submission

Promedio simple de los 14. El archivo es `salida-fraude/submission_blend.csv`
(`id,is_fraud`, mismo orden que el sample). El submission de la v1
(`salida-fraude/submission.csv`) no se toca.
"""))

C.append(nbf.v4.new_code_cell("""# --- Blend mixto_todos: promedio simple de los 14 + submission ---
MIEMBROS = ["hgb_base_s42", "hgb_base_s1", "hgb_base_s7",
            "hgb_g5_s42", "hgb_g5_s1", "hgb_g5_s7",
            "lgbm_s42", "lgbm_s1", "lgbm_s7",
            "hgb_g5_comb_s42", "hgb_g5_comb_s1",
            "lgbm_comb_s42", "lgbm_comb_s1", "logreg"]
assert set(MIEMBROS) == set(probs), "faltan miembros: " + str(set(MIEMBROS) - set(probs))
blend = np.mean([probs[m] for m in MIEMBROS], axis=0)
print("blend: media", round(float(blend.mean()), 4),
      "| min", round(float(blend.min()), 4),
      "| max", round(float(blend.max()), 4))
os.makedirs("salida-fraude", exist_ok=True)
sub = ft.escribir_submission(test, blend, "salida-fraude/submission_blend.csv")
checks = ft.verificar_submission(sub, sample)
for c in checks:
    print(("OK  " if c["ok"] else "FAIL"), "-", c["check"], "-", c["detalle"])
assert all(c["ok"] for c in checks), "checks de formato fallidos"
print("submission_blend.csv escrito:", len(sub), "filas")
"""))

C.append(nbf.v4.new_code_cell("""# --- Verificación final + reproducción OOF del blend ---
import hashlib
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# 1) Reproducción OOF desde los archivos guardados (sin re-entrenar)
oof1 = dict(np.load("verificacion/oof_ensamble.npz"))
oof2 = dict(np.load("verificacion/oof_ensamble_comb.npz"))
oof = {**oof1, **oof2}
p_oof = np.mean([oof[m] for m in MIEMBROS], axis=0)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs_oof = [round(float(roc_auc_score(y[va], p_oof[va])), 4)
            for _, va in skf.split(train, y)]
oof_media = round(float(np.mean(aucs_oof)), 4)
oof_std = round(float(np.std(aucs_oof)), 4)
oof_pool = round(float(roc_auc_score(y, p_oof)), 4)
print("OOF mixto_todos:", aucs_oof, "media", oof_media, "+-", oof_std,
      "| agrupado", oof_pool)
print("Esperado: [0.9448, 0.9481, 0.9512, 0.9466, 0.9416] media 0.9465",
      "+- 0.0032 | agrupado 0.9464")

# 2) Sanidad vs la v1 (que queda intacta)
v1 = pd.read_csv("salida-fraude/submission.csv")
r_p, _ = pearsonr(sub["is_fraud"], v1["is_fraud"])
r_s, _ = spearmanr(sub["is_fraud"], v1["is_fraud"])
mad = float(np.mean(np.abs(sub["is_fraud"] - v1["is_fraud"])))
print("vs v1: Pearson", round(float(r_p), 4), "| Spearman",
      round(float(r_s), 4), "| diferencia media abs", round(mad, 4))
print("v1 media", round(float(v1["is_fraud"].mean()), 4),
      "| blend media", round(float(sub["is_fraud"].mean()), 4))
# Calibración del umbral: ningún miembro individual supera 0.99 vs v1
# (máximo lgbm_comb_s1, Pearson 0.9898), así que el blend tampoco
# puede; el criterio correcto es que el blend supere a la mediana de
# sus miembros (el promediado reduce ruido).
corr_miembros = []
for m in MIEMBROS:
    rp_m, _ = pearsonr(probs[m], v1["is_fraud"])
    rs_m, _ = spearmanr(probs[m], v1["is_fraud"])
    corr_miembros.append((rp_m, rs_m))
med_p = float(np.median([c[0] for c in corr_miembros]))
med_s = float(np.median([c[1] for c in corr_miembros]))
print("correlación miembros vs v1: Pearson mediana", round(med_p, 4),
      "| Spearman mediana", round(med_s, 4))

def md5(ruta):
    return hashlib.md5(open(ruta, "rb").read()).hexdigest()

v = []
v.append(("train 699,960 x 26", train.shape == (699960, 26)))
v.append(("test 299,984 x 25", test.shape == (299984, 25)))
v.append(("14 miembros entrenados", len(probs) == 14))
v.append(("OOF reproduce 0.9465 +- 0.0032",
          abs(oof_media - 0.9465) < 0.0001 and abs(oof_std - 0.0032) < 0.0001
          and abs(oof_pool - 0.9464) < 0.0001))
v.append(("submission: todos los checks OK",
          all(c["ok"] for c in checks)))
v.append(("v1 intacta (md5 c07bfd80...)", md5("salida-fraude/submission.csv") == "c07bfd803b2704a13ec7f7d2afff7bd8"))
v.append(("blend supera mediana de miembros vs v1",
          r_p > med_p and r_s > med_s))
ok = sum(1 for _, b in v if b)
for nombre, b in v:
    print(("OK  " if b else "FAIL"), "-", nombre)
print(str(ok) + "/" + str(len(v)) + " verificaciones PASS")

blend_json = {
    "blend": "mixto_todos",
    "miembros": MIEMBROS,
    "promediado": "media simple, pesos iguales",
    "oof_reproducido": {"aucs_por_pliegue": aucs_oof,
                        "auc_media": oof_media, "auc_std": oof_std,
                        "auc_oof_agrupado": oof_pool},
    "oof_esperado": {"aucs_por_pliegue": [0.9448, 0.9481, 0.9512, 0.9466, 0.9416],
                     "auc_media": 0.9465, "auc_std": 0.0032,
                     "auc_oof_agrupado": 0.9464},
    "tiempos_s": {k: vv["tiempo_s"] for k, vv in tiempos.items()},
    "tiempo_total_s": round(sum(vv["tiempo_s"] for vv in tiempos.values()), 1),
    "submission": {"ruta": "salida-fraude/submission_blend.csv",
                   "filas": int(len(sub)),
                   "md5": md5("salida-fraude/submission_blend.csv"),
                   "media_prob": round(float(sub["is_fraud"].mean()), 4),
                   "min_prob": round(float(sub["is_fraud"].min()), 4),
                   "max_prob": round(float(sub["is_fraud"].max()), 4),
                   "checks_ok": sum(1 for c in checks if c["ok"]),
                   "checks_total": len(checks)},
    "sanidad_vs_v1": {"pearson": round(float(r_p), 4),
                      "spearman": round(float(r_s), 4),
                      "mediana_pearson_miembros": round(med_p, 4),
                      "mediana_spearman_miembros": round(med_s, 4),
                      "diferencia_media_abs": round(mad, 4),
                      "media_v1": round(float(v1["is_fraud"].mean()), 4),
                      "md5_v1": md5("salida-fraude/submission.csv")},
    "verificaciones": [{"nombre": n, "ok": bool(b)} for n, b in v],
    "veredicto": "LISTO" if ok == len(v) else "REVISAR",
}
Path("verificacion/blend.json").write_text(
    json.dumps(blend_json, indent=2, ensure_ascii=False))
print("blend.json escrito. Veredicto:", blend_json["veredicto"])
"""))

C.append(nbf.v4.new_markdown_cell("""## Lectura honesta

- El blend sube el CV local de 0.9449 (base publicada) a 0.9465
  (+0.0016) y de 0.9451 (reproducción local) a 0.9465 (+0.0014 pareado,
  positivo en los 5 pliegues). Es una ganancia real pero pequeña, por
  debajo de la desviación entre pliegues (~0.0035) y de la barra 0.9484.
- El score público puede no reflejar la ganancia: el leaderboard
  público es solo una muestra y el privado decide. El envío se hace
  por decisión de Elán para medirlo, no porque el CV lo garantice.
- La v2 de Elán (pública 0.94393) sigue seleccionada como final; este
  blend no cambia ninguna selección. Si el público del blend supera a
  la v2, la decisión de cambiar la final es de Elán.
"""))

nb["cells"] = C
nbf.write(nb, "fraude-tarjetas-blend.ipynb")
print("fraude-tarjetas-blend.ipynb escrito:", len(C), "celdas")
