#!/usr/bin/env python3
"""Construye fraude-tarjetas.ipynb (12 celdas: 6 markdown + 6 código).

El notebook corre de arriba a abajo sin pasos manuales. Resuelve rutas
Kaggle (/kaggle/input/playground-series-s6e11 para los datos de la
competencia; /kaggle/input/**/fraude_tarjetas.py para el módulo) o
locales (el directorio del notebook).
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

C = []

C.append(nbf.v4.new_markdown_cell("""# Fraude con tarjeta (Playground Series S6E11)

**Fractal Core Research** — Solución para la competencia Kaggle
*Predicting Credit Card Fraud* (cierra 30 nov 2026, métrica ROC AUC,
envío `id,is_fraud` con probabilidades 0–1).

La pregunta que responde este notebook: **¿las señales de riesgo
añaden señal para detectar fraude?** El enfoque: el monto es
`amount_usd`; la autenticación es `auth_method` y `cvv_retry_count`;
la concentración de fraude se mide por `merchant_category`; y el
**conteo de señales de riesgo distintas** cuenta los tipos distintos
de señal de riesgo por transacción (umbral: ≥ 5). La serie temporal
es **DATOS INSUFICIENTES**: no hay serie temporal por entidad (el `id`
no es temporal, medido).

Reglas de la competencia (vinculantes): máximo **10 envíos por día**
(Rules Sec. 2.2), 2 finales seleccionables; el leaderboard público es
solo una muestra, el privado decide — la selección de modelo se hace
por CV local, no por envíos repetidos; prohibido AutoML, etiquetar el
test y entrenar con el test. **Este notebook NO envía nada a Kaggle:
el envío lo decide Elán después.**
"""))

C.append(nbf.v4.new_code_cell("""# --- Setup: rutas Kaggle o local, e importar el adaptador ---
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Datos de la competencia: en Kaggle van a
# /kaggle/input/playground-series-s6e11/{train,test,sample_submission}.csv
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
            for comp in [Path("/kaggle/input/playground-series-s6e11")]:
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

C.append(nbf.v4.new_markdown_cell("""## 1. Las features (medidas, no declaradas)

| Grupo | Definición |
|---|---|
| **Monto** | `amount_usd` (1–6,870 USD, media 79.9) y `account_balance_usd`. |
| **Autenticación** | `auth_method` (con autenticación: Biometric, PIN, OTP, 3D Secure; sin autenticación: No Authentication) y `cvv_retry_count` (forzar la verificación con reintentos). |
| **Concentración por categoría** | `merchant_category` por concentración de fraude + `merchant_risk_score`. |
| **Señales de riesgo** | La transacción que mueve el mismo monto sin autenticación ni verificación. |
| **Intermediario** | No medible (gap). |

**Conteo de señales de riesgo distintas:** 10 tipos distintos de
señal de riesgo por transacción (extranjero, comercio nuevo, VPN,
mismatch de país, mismatch de facturación/envío, intento de estafa
por IA, sin autenticación, reintento de CVV, velocidad sobre p90,
distancia sobre p90). Umbral: ≥ 5 tipos distintos.
"""))

C.append(nbf.v4.new_code_cell("""# --- Conteo de señales, cuota de fraude por categoría/canal, autenticación ---
t_senales, res_senales = ft.tabla_conteo_senales(train)
print("== Conteo de señales de riesgo: tasa de fraude por conteo ==")
print(t_senales.to_string(index=False))
print()
print("Umbral >= 5 señales:", res_senales["n_umbral"], "txns",
      f"({res_senales['pct_umbral']}% del total),",
      f"tasa de fraude {res_senales['tasa_umbral']:.2%} vs "
      f"{res_senales['tasa_sin_umbral']:.2%} sin llegar al umbral;",
      f"captura {res_senales['fraudes_capturados']} de "
      f"{res_senales['fraudes_totales']} fraudes "
      f"({res_senales['pct_fraudes_capturados']}%), lift {res_senales['lift']}x")
var_senales = ft.tabla_conteo_senales_sin_distancia(train)
print("Variante sin distancia:", var_senales)

t_cat = ft.tabla_cuota_fraude(train, "merchant_category")
print()
print("== Cuota de fraude por merchant_category "
      "(cuota = fraude/(total-fraude)) ==")
print(t_cat[["merchant_category", "n", "fraude", "cuota_fraude",
             "cuota_fraude_n_pct", "cuota_fraude_usd_pct",
             "nivel_concentracion"]].to_string(index=False))

t_ch = ft.tabla_cuota_fraude(train, "channel")
print()
print("== Cuota de fraude por channel ==")
print(t_ch[["channel", "n", "fraude", "cuota_fraude",
            "cuota_fraude_n_pct"]].to_string(index=False))

t_auth = ft.tabla_auth(train)
print()
print("== Autenticación por auth_method ==")
print(t_auth.to_string(index=False))
"""))

C.append(nbf.v4.new_markdown_cell("""## 2. Serie temporal: DATOS INSUFICIENTES (honesto)

No hay serie temporal real por entidad: no hay fecha, ni id de
tarjeta, ni id de comercio. El `id` es solo orden de fila y **no es
temporal** (medido: correlación con `time_of_day_hour` −0.002, con
`day_of_week` 0.0005, con `is_fraud` −0.0006; tasa de fraude por decil
de id plana). `time_of_day_hour` y `day_of_week` son cíclicos, no una
línea de tiempo. Sin serie por entidad no hay tendencia temporal que
ajustar: se declara DATOS INSUFICIENTES en vez de inventar una serie.
Lo que sí existe: la cuota de fraude transversal por categoría/canal,
el conteo de señales, la autenticación y los niveles de concentración.
"""))

C.append(nbf.v4.new_code_cell("""# --- Diagnóstico temporal + figuras ---
diag = ft.diagnostico_temporal(train)
print("serie temporal:", diag["estado"])
print("Razón:", diag["razon"])
print("Evidencia:", diag["evidencia"])
print("Faltantes:")
for f in diag["faltantes"]:
    print(" -", f)

import os
figs = ft._figuras("salida-fraude", t_senales, t_cat, t_auth)
print("Figuras:", figs)
for f in figs:
    print(" -", os.path.join("salida-fraude", f))
"""))

C.append(nbf.v4.new_markdown_cell("""## 3. Modelo: ¿las señales de riesgo añaden señal?

`HistGradientBoostingClassifier` (200 árboles, lr 0.08, 31 hojas), CV
estratificada de 5 pliegues, métrica ROC AUC. Tres modos: **(a)** solo
features crudas, **(b)** solo features de señales de riesgo (10
señales, conteo, umbral ≥ 5, sin autenticación, cuota de fraude por
categoría/canal, cuotas de fraude, nivel de concentración), **(c)**
combinadas. Las tablas de referencia (cuota de fraude por
categoría/canal) se calculan SOLO con el pliegue de entrenamiento de
cada fold: sin fuga. Baseline nulo (constante 0.01669953): AUC 0.5.
"""))

C.append(nbf.v4.new_code_cell("""# --- Comparación de modelos (CV 5 pliegues; tarda unos minutos) ---
comparacion = ft.comparar_modelos(train)
for modo in ("crudas", "senales", "combinadas"):
    r = comparacion[modo]
    print(f"{modo:11s} AUC media {r['auc_media']:.4f} ± {r['auc_std']:.4f}  "
          f"pliegues {r['aucs']}")
print("baseline constante: AUC 0.5")
"""))

C.append(nbf.v4.new_markdown_cell("""## 4. Datos externos (original CC0)

Las reglas permiten datos externos si son públicos, gratuitos y
accesibles para todos (Sec. 3.1). Candidato: el dataset original
*Credit Card Fraud Detection 2026* (CC0, 20,000 filas, 1.7% de fraude,
mismas columnas salvo `transaction_id` en vez de `id`). Sus
distribuciones son cercanas, no idénticas: antes de usarlo se compara
la distribución y se corre la CV con y sin él, validando SOLO sobre los
pliegues de la competencia. Si el archivo no está en `datos/`, esta
celda lo deja como PENDIENTE y no usa ningún dato externo.
"""))

C.append(nbf.v4.new_code_cell("""# --- Variante con datos externos (si el archivo está) ---
import os
ruta_ext = DATOS / "credit-card-fraud-detection-2026.csv"
if ruta_ext.exists():
    ext = pd.read_csv(ruta_ext)
    comp_ext = ft.evaluar_con_externo(train, ext)
    print("Externo:", comp_ext["n_externo"], "filas, tasa de fraude",
          comp_ext["tasa_fraude_externo"], "vs competencia",
          comp_ext["tasa_fraude_competencia"])
    print("Mejor sin externo:", comp_ext["mejor_modo_sin_externo"],
          comp_ext["mejor_sin_externo"])
    for modo in ("crudas", "senales", "combinadas"):
        r = comp_ext["por_modo"][modo]
        print(f"{modo:11s} con externo {r['auc_con_externo_media']:.4f} ± "
              f"{r['auc_con_externo_std']:.4f}  sin externo "
              f"{r['auc_sin_externo_media']:.4f} ± {r['auc_sin_externo_std']:.4f}  "
              f"pliegues con {r['aucs_con']}")
    print("Veredicto:", comp_ext["veredicto"])
    externo = comp_ext
else:
    print("PENDIENTE: el dataset original CC0 no está en datos/.")
    print("Se descarga con sesión de Kaggle (dataset público) a",
          ruta_ext, "y esta celda corre sola la comparación.")
    print("Hasta entonces: NO se usa ningún dato externo.")
    externo = None
"""))

C.append(nbf.v4.new_markdown_cell("""## 5. Envío (submission.csv)

Modelo final (combinadas) entrenado con TODO train, salvo que la
variante con datos externos gane fuera del ruido (ver celda anterior):
en ese caso el envío se regenera con train + externo en el modo
ganador. El envío es `id,is_fraud` con probabilidades 0–1, una fila
por cada id de test, en el mismo orden que `sample_submission.csv`.
Se verifica contra el sample antes de escribirlo. **Este notebook NO
envía nada a Kaggle: el envío lo decide Elán después.**
"""))

C.append(nbf.v4.new_code_cell("""# --- Entrenamiento final + envío + informe + verificación ---
if "externo" in dir() and externo and externo.get("modo_ganador"):
    modo_final = externo["modo_ganador"]
    modelo, enc, ref = ft.entrenar_final_con_externo(train, ext, modo_final)
    probs = ft.predecir_test(modelo, enc, ref, test, modo_final)
    con_externo = True
    print(f"Envío regenerado con train + externo, modo {modo_final} "
          f"(el externo sube el AUC fuera del ruido).")
else:
    modo_final = "combinadas"
    modelo, enc, ref = ft.entrenar_final(train)
    probs = ft.predecir_test(modelo, enc, ref, test, modo_final)
    con_externo = False
    if "externo" in dir() and externo:
        print("El externo no sube el AUC fuera del ruido: se descarta; "
              "el envío usa el modelo combinado con TODO train.")
    else:
        print("Sin datos externos: el envío usa el modelo combinado "
              "con TODO train.")
sub = ft.escribir_submission(test, probs, "salida-fraude/submission.csv")
checks = ft.verificar_submission(sub, sample)
for c in checks:
    print(("OK  " if c["ok"] else "FAIL"), "-", c["check"], "-", c["detalle"])
print("Media de probabilidad predicha:", round(float(sub["is_fraud"].mean()), 4),
      "(base 0.0167)")

resultados = {
    "tabla_conteo_senales": t_senales, "resumen_senales": res_senales,
    "conteo_senales_sin_distancia": var_senales,
    "tabla_cuota_fraude_categoria": t_cat,
    "tabla_cuota_fraude_channel": t_ch, "tabla_auth": t_auth,
    "diagnostico_temporal": diag, "comparacion_modelos": comparacion,
    "externo": externo if "externo" in dir() else None,
    "externo_pendiente": not ruta_ext.exists(),
    "envio": {"ruta": "salida-fraude/submission.csv",
              "n_filas": int(len(sub)),
              "checks_ok": sum(1 for c in checks if c["ok"]),
              "checks_total": len(checks),
              "checks_fallidos": [c["check"] for c in checks if not c["ok"]],
              "media_prob": round(float(sub["is_fraud"].mean()), 4),
              "modo": modo_final, "con_externo": con_externo},
}
md = ft.informe(resultados, dir_salida="salida-fraude")
print("Informe escrito en salida-fraude/informe-fraude.md")

# --- Verificación final del notebook (12 checks) ---
v = []
v.append(("train 699,960 x 26", train.shape == (699960, 26)))
v.append(("test 299,984 x 25", test.shape == (299984, 25)))
v.append(("sample 299,984 x 2", sample.shape == (299984, 2)))
v.append(("tasa fraude train 0.0167", abs(train["is_fraud"].mean() - 0.0167) < 0.0001))
v.append(("cero nulos train/test",
          int(train.isna().sum().sum()) == 0 and int(test.isna().sum().sum()) == 0))
tasas = t_senales["tasa"].values
v.append(("conteo de señales: tasa no decrece con el conteo (satura en 100%)",
          bool(all(tasas[i] <= tasas[i + 1] for i in range(len(tasas) - 1)))))
v.append(("umbral >= 5 señales: tasa > 0.80",
          res_senales["tasa_umbral"] > 0.80))
v.append(("cuota Gift Cards 0.0787",
          abs(float(t_cat.loc[t_cat["merchant_category"] == "Gift Cards", "cuota_fraude"].iloc[0]) - 0.0787) < 0.0001))
v.append(("auth No Authentication 0.0644",
          abs(float(t_auth.loc[t_auth["auth_method"] == "No Authentication", "tasa"].iloc[0]) - 0.0644) < 0.0001))
v.append(("temporal: datos_insuficientes", diag["estado"] == "datos_insuficientes"))
v.append(("CV: 3 modos con AUC > 0.5",
          all(comparacion[m]["auc_media"] > 0.5 for m in ("crudas", "senales", "combinadas"))))
v.append(("submission: todos los checks OK",
          all(c["ok"] for c in checks)))
ok = sum(1 for _, b in v if b)
for nombre, b in v:
    print(("OK  " if b else "FAIL"), "-", nombre)
print(f"\\n{ok}/{len(v)} verificaciones PASS")
"""))

nb.cells = C
with open("fraude-tarjetas.ipynb", "w", encoding="utf-8") as fh:
    nbf.write(nb, fh)
print(f"Notebook escrito: {len(C)} celdas "
      f"({sum(1 for c in C if c.cell_type == 'code')} código, "
      f"{sum(1 for c in C if c.cell_type == 'markdown')} markdown)")
