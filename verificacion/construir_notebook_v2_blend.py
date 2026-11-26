#!/usr/bin/env python3
"""Construye fraude-tarjetas-v2-blend.ipynb (13 celdas: 7 markdown + 6 código).

El notebook reproduce la solución actual del consultor (ensamble_v2):
target encoding anidado por dígitos + GLM-boost (init_score) + blend
ponderado 0.7/0.1/0.2/0.0. Corre de arriba a abajo sin pasos manuales
y escribe submission.csv (el blend). CV de 5 pliegues (semilla 42)
con los 6 candidatos y deltas pareados vs lgb4_raw; los valores
esperados están en la celda de CV (veredicto 2026-11-25). Resuelve
rutas Kaggle (/kaggle/input/playground-series-s6e11 para los datos;
/kaggle/input/**/fraude_tarjetas.py para el módulo) o locales (el
directorio del notebook). Las funciones del pipeline se copian
verbatim de ensamble_v2.py (solo log.info -> print).
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

C = []

C.append(nbf.v4.new_markdown_cell('''# Fraude con tarjeta — GLM-boost + blend ponderado (Playground Series S6E11)

**Fractal Core Research** — Solución para la competencia Kaggle
*Predicting Credit Card Fraud* (cierra 30 nov 2026, métrica ROC AUC,
envío `id,is_fraud` con probabilidades 0–1).

**Resultado:** el blend ponderado **0.95119 público** (final) y el modelo
solo (glm_lgb4) **0.95112 público** (final). CV local (5 pliegues,
semilla 42, medido 2026-11-25): glm_lgb4 **0.95066 ± 0.0033**, blend
ponderado anidado **0.95072**. Los dos envíos finales son estos dos;
la selección se hizo por CV local, no por el público.

**La idea en una línea:** los datos son casi logísticos, así que un GLM
bien puesto + LightGBM de 4 hojas que parte del logit del GLM
(`init_score`) + target encoding anidado por dígitos le gana al
baseline por +0.004. El blend (0.7 glm_lgb4 + 0.1 glm + 0.2 lgb4_te)
no sube el AUC: su valor es de piso, no de AUC adicional.

**Lectura honesta:** la v3 (llaves de pares, bagging del TE, MLP) no
superó al glm_lgb4 (mejor delta +0.00013, ruido); stacking y tuning
tampoco. El top del leaderboard (0.96–0.97) es sondeo del público, no
un modelo mejor: los participantes honestos con CV ~0.9526 reportan lo
mismo. Todo lo que afirma este notebook está medido con el protocolo
de abajo; nada inventado.

Reglas: máximo **10 envíos por día**, 2 finales seleccionables. **Este
notebook NO envía nada a Kaggle: el envío lo decide Elán después.**
'''))

C.append(nbf.v4.new_code_cell('''# --- Setup: rutas Kaggle o local, e importar el adaptador ---
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
'''))

C.append(nbf.v4.new_markdown_cell('''## 1. Features crudas

Numéricas + booleanas + one-hot de las 5 categóricas, con el encoder
ajustado solo en train. Las crudas solas ya dan 0.9467 (LightGBM 4
hojas, CV). Las 18 señales de riesgo del adaptador (conteo, cuotas)
se midieron aparte y NO suman AUC en este pipeline: el valor está en
el target encoding y en el GLM-boost, no en las señales. La celda de
abajo construye las crudas y reporta su forma.
'''))

C.append(nbf.v4.new_code_cell('''# --- Features crudas (TODO train) ---
X_tr, enc = ft.construir_features_crudas(train, fit=True)
X_te, _ = ft.construir_features_crudas(test, encoder=enc)
print("X_tr crudas:", X_tr.shape, "| X_te crudas:", X_te.shape)
print("columnas crudas:", X_tr.shape[1])
'''))

C.append(nbf.v4.new_markdown_cell('''## 2. Target encoding anidado (por dígitos)

Por cada columna numérica se construyen 6 llaves de dígitos (valor,
entero, entero//10, décimas, centavos, 2 primeros dígitos) + las 5
categóricas como texto: 83 llaves en total. El TE es el logit
suavizado (20 pseudo-conteos) de la tasa de fraude por llave.

**Sin fuga:** para las filas de entrenamiento se usa cross-fitting
interno (5 pliegues, semilla 17): cada fila se codifica con
estadísticas de las demás. Para validación/test se usa todo el
entrenamiento. El TE solo (lgb4_te) da 0.95000 en CV, +0.00335
pareado contra lgb4_raw.
'''))

C.append(nbf.v4.new_code_cell('''# --- Llaves de dígitos y target encoding anidado (sin fuga) ---
from sklearn.model_selection import StratifiedKFold

SEED_INNER = 17
SUAVIZADO = 20.0           # pseudo-conteos del suavizado bayesiano del TE
COLS_TE_NUM = ["amount_usd", "account_balance_usd", "distance_from_home_km",
               "velocity_score", "hours_since_last_txn", "merchant_risk_score",
               "card_age_months", "customer_age", "txn_count_last_24h",
               "time_of_day_hour", "day_of_week", "cvv_retry_count",
               "prior_disputes"]


def construir_llaves(df):
    # Llaves enteras (codificadas como texto) por columna: valor, parte
    # entera, entera//10, décimas, centavos, primeros dígitos. No usa la
    # etiqueta. Devuelve DataFrame de strings, una columna por llave.
    out = {}
    for c in COLS_TE_NUM:
        v = pd.to_numeric(df[c], errors="coerce").fillna(-1.0).values.astype(float)
        ent = np.floor(v)
        frac = v - ent
        out[f"{c}__val"] = np.round(v, 4).astype(str)
        out[f"{c}__ent"] = ent.astype(np.int64).astype(str)
        out[f"{c}__ent10"] = (ent // 10).astype(np.int64).astype(str)
        out[f"{c}__dec"] = np.floor(frac * 10).astype(np.int64).astype(str)
        out[f"{c}__cts"] = (np.round(frac * 100).astype(np.int64) % 100).astype(str)
        out[f"{c}__lead2"] = np.array([s[:2] for s in ent.astype(np.int64).astype(str)])
    for c in ft.COLS_CAT:
        out[f"{c}__cat"] = df[c].astype(str).values
    return pd.DataFrame(out, index=df.index)


def _codificar(llaves_a, llaves_b):
    # Códigos enteros comunes (factorize sobre la concatenación; no usa y).
    cods_a, cods_b = {}, {}
    for c in llaves_a.columns:
        todo = np.concatenate([llaves_a[c].values, llaves_b[c].values])
        cod, uniq = pd.factorize(todo)
        cods_a[c] = cod[:len(llaves_a)]
        cods_b[c] = cod[len(llaves_a):]
    return cods_a, cods_b


def _logit_te(cod_fit, y_fit, cod_apply, prior, n_cod):
    s = np.bincount(cod_fit, weights=y_fit, minlength=n_cod)
    n = np.bincount(cod_fit, minlength=n_cod).astype(float)
    p = (s + SUAVIZADO * prior) / (n + SUAVIZADO)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))[cod_apply]


def target_encoding_anidado(llaves_tr, y_tr, llaves_otro):
    # TE por valor, logit suavizado. Para las filas de entrenamiento se
    # usa cross-fitting interno (5 pliegues): cada fila se codifica con
    # estadísticas de las demás. Para `llaves_otro` (validación/test) se
    # usa TODO el entrenamiento. Devuelve (TE_tr, TE_otro) float32.
    cod_tr, cod_ot = _codificar(llaves_tr, llaves_otro)
    prior = float(y_tr.mean())
    cols = list(llaves_tr.columns)
    te_tr = np.zeros((len(llaves_tr), len(cols)), dtype=np.float32)
    te_ot = np.zeros((len(llaves_otro), len(cols)), dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED_INNER)
    idx_internos = list(skf.split(np.zeros(len(y_tr)), y_tr))
    for j, c in enumerate(cols):
        a, b = cod_tr[c], cod_ot[c]
        n_cod = int(max(a.max(), b.max())) + 1
        for fit_idx, ap_idx in idx_internos:
            te_tr[ap_idx, j] = _logit_te(a[fit_idx], y_tr[fit_idx], a[ap_idx],
                                         prior, n_cod)
        te_ot[:, j] = _logit_te(a, y_tr, b, prior, n_cod)
    return te_tr, te_ot
'''))

C.append(nbf.v4.new_markdown_cell('''## 3. Los modelos

- **lgb31_raw:** LightGBM 31 hojas, 200 árboles (baseline comparable).
- **lgb4_raw:** LightGBM 4 hojas, 800 árboles, lr 0.05. Los datos son
  casi logísticos, así que 4 hojas bastan y 31 sobreajustan.
- **lgb4_te:** lgb4 sobre crudas + TE anidado.
- **glm:** regresión logística (newton-cholesky) sobre crudas + TE,
  escaladas.
- **glm_lgb4:** LightGBM 4 hojas (400 árboles) que parte del logit del
  GLM (`init_score`): el GLM pone la dirección y los árboles corrigen
  lo que queda. Es el mejor solo: 0.95066.
- **blend_rank:** media de rangos de lgb4_te, glm_lgb4 y lgb4_raw
  (referencia, no el final).
'''))

C.append(nbf.v4.new_code_cell('''# --- Modelos + preparar_pliegue + ajustar_predecir ---
import gc
import time

import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SEED_CV = 42
N_SPLITS = 5


def lgb31(seed=42):
    return lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.08, num_leaves=31,
        min_child_samples=50, reg_lambda=1.0, objective="binary",
        verbose=-1, n_jobs=2, random_state=seed)


def lgb4(seed=42):
    return lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.05, num_leaves=4,
        min_child_samples=100, reg_lambda=5.0, colsample_bytree=0.7,
        subsample=0.8, subsample_freq=1, objective="binary",
        verbose=-1, n_jobs=2, random_state=seed)


def glm():
    return LogisticRegression(C=1.0, solver="newton-cholesky", max_iter=100)


def _auc(y, p):
    return float(roc_auc_score(y, p))


def _rank(p):
    return rankdata(p) / len(p)


NOMBRES = ["lgb31_raw", "lgb4_raw", "lgb4_te", "glm", "glm_lgb4", "blend_rank"]


def preparar_pliegue(d_tr, d_va):
    # Matrices de un pliegue: crudas, TE anidado, y versiones escaladas.
    X_tr, enc = ft.construir_features_crudas(d_tr, fit=True)
    X_va, _ = ft.construir_features_crudas(d_va, encoder=enc)
    X_tr = X_tr.astype(np.float32)
    X_va = X_va.astype(np.float32)
    y_tr = d_tr[ft.TARGET].values.astype(float)
    te_tr, te_va = target_encoding_anidado(construir_llaves(d_tr), y_tr,
                                           construir_llaves(d_va))
    XT_tr = np.hstack([X_tr.values, te_tr])
    XT_va = np.hstack([X_va.values, te_va])
    sc = StandardScaler()
    XS_tr = sc.fit_transform(XT_tr).astype(np.float32)
    XS_va = sc.transform(XT_va).astype(np.float32)
    return X_tr.values, X_va.values, XT_tr, XT_va, XS_tr, XS_va


def ajustar_predecir(d_tr, d_va, y_tr):
    # Ajusta los 5 modelos + blend en un pliegue. Devuelve dict nombre->p.
    Xr_tr, Xr_va, XT_tr, XT_va, XS_tr, XS_va = preparar_pliegue(d_tr, d_va)
    p = {}
    t = time.time()
    m = lgb31(); m.fit(Xr_tr, y_tr); p["lgb31_raw"] = m.predict_proba(Xr_va)[:, 1]
    print("    lgb31_raw %.0fs" % (time.time() - t)); t = time.time()
    m = lgb4(); m.fit(Xr_tr, y_tr); p["lgb4_raw"] = m.predict_proba(Xr_va)[:, 1]
    print("    lgb4_raw %.0fs" % (time.time() - t)); t = time.time()
    m = lgb4(); m.fit(XT_tr, y_tr); p["lgb4_te"] = m.predict_proba(XT_va)[:, 1]
    print("    lgb4_te %.0fs" % (time.time() - t)); t = time.time()
    g = glm(); g.fit(XS_tr, y_tr)
    print("    glm n_iter_=%s %.0fs" % (getattr(g, "n_iter_", "?"), time.time() - t))
    p["glm"] = g.predict_proba(XS_va)[:, 1]
    t = time.time()
    init_tr = g.decision_function(XS_tr)
    init_va = g.decision_function(XS_va)
    m = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=4,
        min_child_samples=100, reg_lambda=5.0, colsample_bytree=0.7,
        subsample=0.8, subsample_freq=1, objective="binary",
        verbose=-1, n_jobs=2, random_state=42)
    m.fit(XT_tr, y_tr, init_score=init_tr)
    raw_va = m.predict(XT_va, raw_score=True) + init_va
    p["glm_lgb4"] = 1.0 / (1.0 + np.exp(-raw_va))
    print("    glm_lgb4 %.0fs" % (time.time() - t))
    p["blend_rank"] = (_rank(p["lgb4_te"]) + _rank(p["glm_lgb4"])
                       + _rank(p["lgb4_raw"])) / 3.0
    return p
'''))

C.append(nbf.v4.new_markdown_cell('''## 4. Validación cruzada (el protocolo que decide)

StratifiedKFold, 5 pliegues, shuffle, semilla 42. Todo (TE, encoder,
escalado, GLM) se calcula SOLO con el pliegue de entrenamiento de
cada fold: sin fuga. Se reporta el AUC por pliegue, la media y el
delta pareado contra lgb4_raw. Regla de adopción: delta pareado medio
≥ +0.0005 y positivo en ≥ 4/5 pliegues. La selección se hace por CV,
no por el leaderboard público (que es solo una muestra y tiene ruido
de ±0.003 por envío). La celda tarda ~8 minutos.
'''))

C.append(nbf.v4.new_code_cell('''# --- CV 5 pliegues: los 6 candidatos, deltas pareados vs lgb4_raw ---
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ESPERADO = {
    "lgb31_raw": [0.94341, 0.94776, 0.95009, 0.94563, 0.94013],
    "lgb4_raw": [0.94489, 0.94816, 0.95124, 0.94707, 0.9419],
    "lgb4_te": [0.94897, 0.9506, 0.95503, 0.95084, 0.94456],
    "glm": [0.9495, 0.95087, 0.95551, 0.95081, 0.945],
    "glm_lgb4": [0.94985, 0.95105, 0.95567, 0.95138, 0.94537],
    "blend_rank": [0.94912, 0.95117, 0.95501, 0.95088, 0.94522],
}
y = train[ft.TARGET].values
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED_CV)
oof = {n: np.zeros(len(train)) for n in NOMBRES}
aucs = {n: [] for n in NOMBRES}
t0 = time.time()
for i, (a, b) in enumerate(skf.split(train, y)):
    print(f"pliegue {i + 1}/5", flush=True)
    p = ajustar_predecir(train.iloc[a], train.iloc[b], y[a].astype(float))
    for n in NOMBRES:
        oof[n][b] = p[n]
        aucs[n].append(float(roc_auc_score(y[b], p[n])))
        print(f"  {n:11s} AUC pliegue = {aucs[n][-1]:.5f}", flush=True)
    gc.collect()
print(f"CV completa en {time.time() - t0:.0f}s")
print()
print(f"{'modelo':11s} {'media':>8s} {'std':>7s} {'delta vs lgb4_raw':>16s} {'+/5':>4s} {'reproduce':>10s}")
base = np.array(aucs["lgb4_raw"])
for n in NOMBRES:
    a = np.array(aucs[n])
    d = a - base
    esp = np.array(ESPERADO[n])
    repro = bool(np.max(np.abs(a - esp)) < 0.0005)
    pos = int((d > 0).sum()) if n != "lgb4_raw" else "-"
    print(f"{n:11s} {a.mean():8.5f} {a.std():7.5f} {d.mean():+16.5f} {str(pos):>4s} {str(repro):>10s}")
'''))

C.append(nbf.v4.new_markdown_cell('''## 5. El blend final (ponderado, pesos fijos)

Pesos **0.7 glm_lgb4 + 0.1 glm + 0.2 lgb4_te + 0.0 lgb4_raw**, sobre
probabilidades. Vienen del ajuste global sobre el OOF
(blend_pond_global, 0.95073, `medir_stacking_blend.py` 2026-11-25);
la estimación honesta anidada (pesos por pliegue) es **0.95072**, un
empate técnico con el glm_lgb4 solo (0.95066, Δ +0.00006). No se
adopta por AUC: se adopta por piso — el segundo final solo aporta si
le gana a la v2 en el privado, y el blend es el candidato más cercano
a ella (Spearman 0.999 con la v2). Los pesos NO se reajustan aquí.
'''))

C.append(nbf.v4.new_code_cell('''# --- Ajuste final con TODO train + blend ponderado + submission ---
import os
from scipy.stats import spearmanr

y_full = train[ft.TARGET].values.astype(float)
p_final = ajustar_predecir(train, test.assign(**{ft.TARGET: 0}), y_full)
PESOS = {"glm_lgb4": 0.7, "glm": 0.1, "lgb4_te": 0.2, "lgb4_raw": 0.0}
blend = (PESOS["glm_lgb4"] * p_final["glm_lgb4"]
         + PESOS["glm"] * p_final["glm"]
         + PESOS["lgb4_te"] * p_final["lgb4_te"]
         + PESOS["lgb4_raw"] * p_final["lgb4_raw"])
print("blend: media", round(float(blend.mean()), 4),
      "| min", round(float(blend.min()), 4),
      "| max", round(float(blend.max()), 4))
print("Spearman blend vs glm_lgb4:",
      round(float(spearmanr(blend, p_final["glm_lgb4"]).statistic), 5),
      "(esperado ~0.999)")
sub = ft.escribir_submission(test, blend, "submission.csv")
checks = ft.verificar_submission(sub, sample)
for c in checks:
    print(("OK  " if c["ok"] else "FAIL"), "-", c["check"], "-", c["detalle"])
assert all(c["ok"] for c in checks), "checks de formato fallidos"
print("submission.csv escrito:", len(sub), "filas")

# --- Verificación final del notebook ---
v = []
v.append(("train 699,960 x 26", train.shape == (699960, 26)))
v.append(("test 299,984 x 25", test.shape == (299984, 25)))
v.append(("sample 299,984 x 2", sample.shape == (299984, 2)))
v.append(("tasa fraude train 0.0167", abs(train[ft.TARGET].mean() - 0.0167) < 0.0001))
v.append(("cero nulos train/test",
          int(train.isna().sum().sum()) == 0 and int(test.isna().sum().sum()) == 0))
v.append(("CV glm_lgb4 reproduce 0.95066",
          abs(float(np.mean(aucs["glm_lgb4"])) - 0.95066) < 0.0005))
v.append(("CV lgb4_te reproduce 0.95000",
          abs(float(np.mean(aucs["lgb4_te"])) - 0.95000) < 0.0005))
v.append(("blend Spearman vs glm_lgb4 > 0.99",
          float(spearmanr(blend, p_final["glm_lgb4"]).statistic) > 0.99))
v.append(("blend media prob ~0.0166",
          abs(float(blend.mean()) - 0.0166) < 0.001))
v.append(("submission: todos los checks OK",
          all(c["ok"] for c in checks)))
ok = sum(1 for _, b in v if b)
for nombre, b in v:
    print(("OK  " if b else "FAIL"), "-", nombre)
print(f"\\n{ok}/{len(v)} verificaciones PASS")
'''))

C.append(nbf.v4.new_markdown_cell('''## 6. Resultados y lo que no funcionó

- **CV (5 pliegues, semilla 42, medido 2026-11-25):** lgb31_raw 0.9454,
  lgb4_raw 0.94665, lgb4_te 0.95000, glm 0.95034, **glm_lgb4 0.95066**,
  blend_rank 0.95028. Los valores exactos por pliegue los imprime la
  celda de CV; si difieren de estos en más de 0.0005, algo cambió en
  el entorno.
- **Público (Kaggle):** glm_lgb4 **0.95112** (ref 58917706), blend
  ponderado **0.95119** (ref 58925302), finales 2/2. La v4 anterior
  (0.94723) quedó fuera.
- **Lo que no funcionó (medido, no asumido):** v3 (llaves de pares
  para el TE, bagging de 3 semillas del TE, MLP con early stopping por
  AUC): mejor delta +0.00013, ruido; el MLP solo da 0.94605, peor que
  el control. Stacking con regresión logística: 0.95037. Tuning del
  glm_lgb4 (5 variantes): ninguna supera al control (mejor +0.00001).
  La caza del punch estructural: sin punch; el techo legítimo está en
  ~0.951–0.952.
- **Sobre el top del leaderboard (0.96–0.97):** son cuentas nuevas con
  decenas de envíos; los participantes honestos con CV ~0.9526
  reportan que ≥ 0.96 no está respaldado por CV. Es sondeo del
  público, no una meta.
- **Reproducibilidad:** este notebook corre de arriba a abajo sin
  intervención y escribe `submission.csv` (el blend). El código de
  entrenamiento e inferencia es este notebook + `fraude_tarjetas.py`
  (dataset adjunto). Nada se envía a Kaggle desde aquí: el envío lo
  decide el equipo después.
'''))

nb["cells"] = C
nbf.write(nb, "fraude-tarjetas-v2-blend.ipynb")
print("fraude-tarjetas-v2-blend.ipynb escrito:", len(C), "celdas")
