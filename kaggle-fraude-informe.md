# Fraude con tarjeta — Kaggle Playground Series S6E11

**Fractal Core Research** — Solución construida 2026-11-08. Competencia
"Predicting Credit Card Fraud": métrica ROC AUC, envío CSV `id,is_fraud`
con probabilidades 0–1, cierre 30 nov 2026, 11:59 PM UTC. **Sin envíos
hechos todavía: el envío lo decide Elán después.**

## Qué se construyó

| Archivo | Qué es |
|---|---|
| `fraude_tarjetas.py` | Adaptador v1.0.0: conteo de señales de riesgo, cuota de fraude por categoría/canal, autenticación, diagnóstico temporal, features, CV, entrenamiento final, envío y verificación. Es el código de entrenamiento + inferencia de la solución (la competencia lo pide a los ganadores, Sec. 8.3). |
| `fraude-tarjetas.ipynb` | Notebook de 12 celdas (6 markdown + 6 código): datos → features → conteo de señales → serie temporal → modelo con CV → datos externos → envío. Resuelve rutas Kaggle (`/kaggle/input/**`) o locales. Corre de arriba a abajo sin pasos manuales. |
| `salida-fraude/submission.csv` | Envío generado por el modelo final (combinadas) entrenado con TODO train: 299,984 filas, 7/7 checks OK contra `sample_submission.csv`, media de probabilidad 0.0166 (base 0.0167). |
| `salida-fraude/informe-fraude.md` | Informe generado por el adaptador (mismo contenido que §1–§8 de este documento, en detalle). |
| `salida-fraude/fig-conteo-senales.png`, `fig-cuota-fraude-categoria.png`, `fig-auth.png` | Las 3 figuras del informe. |
| `verificacion/verificar_fraude.py` + `verificacion-fraude.json` | Verificación independiente: re-calcula desde los CSV y escribe N checks. |
| `verificacion/asi_por_tarjeta.py` + `verificacion/asi_por_tarjeta.json` | ASI por tipo de tarjeta (§11): script + tabla completa con detalle por tipo. |
| `verificacion/construir_notebook.py`, `verificacion/correr_notebook.py` | Constructor del notebook y ejecutor celda por celda (sin jupyter). |
| `datos/` | Datos de la competencia, solo lectura: `train.csv` (699,960 × 26), `test.csv` (299,984 × 25), `sample_submission.csv` (299,984 × 2). |

## 1. Las features (medidas, no declaradas)

- **Monto:** `amount_usd` (1–6,870 USD, media 79.9) y
  `account_balance_usd`. La tasa de fraude crece con el monto: < 10 USD
  0.52%, 10–50 0.76%, 50–100 1.08%, 100–500 2.02%, 500–1,000 4.10%,
  > 1,000 8.36%.
- **Autenticación:** `auth_method`. Con autenticación: Biometric 0.50%,
  PIN 0.87%, OTP 1.41%, 3D Secure 1.47% de fraude. Sin autenticación:
  No Authentication 6.44% (37.0% de todo el fraude con 9.58% de las
  txns). `cvv_retry_count` es forzar la verificación con reintentos
  (0 → 0.93%, 1 → 8.75%, 2 → 41.73%, 3 → 36.0%).

  ![Fraude por método de autenticación](salida-fraude/fig-auth.png)
- **Concentración por categoría:** `merchant_category` por concentración
  de fraude (cuota por conteo y por monto) + `merchant_risk_score`.
  Niveles por cuota de fraude: Nivel 0 ≥ 10% (Online Retail 14.71%,
  Groceries 13.56%, Electronics 12.84%, Gaming 11.22%), Nivel 1 3–10%,
  Nivel 2 1–3% (Crypto Exchange 2.95%), Nivel 3 < 1% (ninguna). Por
  monto: Electronics 46.9% y Travel 21.4% del dinero defraudado.
- **Señales de riesgo:** la transacción que mueve el mismo monto sin
  autenticación ni verificación. Señales medidas: extranjero (12.57% vs
  1.48%), comercio nuevo (4.73% vs 1.12%), VPN (6.55% vs 1.50%),
  mismatch de país (16.50% vs 1.43%), mismatch de facturación/envío
  (12.99% vs 1.52%), intento de estafa por IA (15.22% vs 1.63%),
  velocidad alta, conteo de txns alto, horas desde la última txn bajo,
  distancia alta (PLANA: 1.72% vs 1.66%, no discrimina), disputas
  previas (1.55% con 0, 3.64% con 3).
- **Intermediario:** no medible con estos datos — gap registrado, no
  inventado.

## 2. Conteo de señales de riesgo distintas

Por transacción, se cuentan los tipos DISTINTOS de señal de riesgo
entre 10 definidas (extranjero, comercio nuevo, VPN, mismatch de país,
mismatch de facturación/envío, intento de estafa por IA, sin
autenticación, reintento de CVV ≥ 1, velocidad sobre p90
—`velocity_score` ≥ 33.10—, distancia sobre p90
—`distance_from_home_km` ≥ 31.43 km—). Umbral: ≥ 5 tipos distintos.

| Señales distintas | n | Tasa de fraude |
|---|---|---|
| 0 | 373,315 | 0.19% |
| 1 | 241,861 | 1.06% |
| 2 | 69,307 | 5.40% |
| 3 | 12,692 | 22.38% |
| 4 | 2,236 | 59.66% |
| 5 | 459 | 85.40% |
| 6 | 81 | 98.77% |
| 7 | 8 | 100% |
| 8 | 1 | 100% |

**Umbral ≥ 5 señales:** 549 txns (0.08%), tasa de fraude 87.61% frente
a 1.60% sin llegar al umbral; captura 481 de 11,689 fraudes (4.1%);
lift 52.46x sobre la base. La tasa crece monótona con el conteo.

**Variante sin distancia:** la señal de distancia es PLANA (fraude
1.72% sobre p90 vs 1.66% bajo p90): no discrimina en estos datos. Sin
ella, el umbral ≥ 5 son 388 txns con 94.07% de fraude y 3.1% de los
fraudes capturados (vs 549 / 87.61% / 4.1% con distancia). La distancia
añade 161 txns por encima del umbral con menor tasa: es ruido en estos
datos, no señal. Se mantiene en el conteo por definición y se reporta
la variante.

**El conteo de señales solo ya tiene AUC 0.8702** (Mann-Whitney sobre
train): un solo entero, sin modelo, separa fraude de no-fraude mejor
que `merchant_risk_score` solo (0.7489) o `cvv_retry_count` solo
(0.7081). El umbral ≥ 5 solo tiene AUC 0.5205: es preciso (87.61%)
pero raro (0.08% de las txns), así que su valor está en el conteo, no
en el binario.

![Tasa de fraude por conteo de señales de riesgo](salida-fraude/fig-conteo-senales.png)

## 3. Cuota de fraude por merchant_category

Cuota de fraude = fraude / (total − fraude). Transversal, sin tiempo.

| Categoría | Cuota | Cuota fraude (n) | Cuota fraude (USD) | Nivel |
|---|---|---|---|---|
| Gift Cards | 0.0787 | 5.90% | 1.96% | Nivel 1 |
| Crypto Exchange | 0.0698 | 2.95% | 10.91% | Nivel 2 |
| Gaming | 0.0583 | 11.22% | 3.36% | Nivel 0 |
| Electronics | 0.0318 | 12.84% | 46.90% | Nivel 0 |
| Streaming | 0.0230 | 9.08% | 1.09% | Nivel 1 |
| Travel | 0.0189 | 4.57% | 21.42% | Nivel 1 |
| Utilities | 0.0140 | 4.97% | 1.25% | Nivel 1 |
| Healthcare | 0.0130 | 3.95% | 1.09% | Nivel 1 |
| Fuel | 0.0125 | 6.34% | 1.49% | Nivel 1 |
| Restaurants | 0.0116 | 9.92% | 2.75% | Nivel 1 |
| Groceries | 0.0112 | 13.56% | 3.75% | Nivel 0 |
| Online Retail | 0.0112 | 14.71% | 4.05% | Nivel 0 |

Por channel: Online 0.0177, POS 0.0169, In-App 0.0166, ATM 0.0161,
Contactless 0.0158.

**Lectura:** por conteo de fraude las categorías con más fraude son
Online Retail, Groceries, Electronics y Gaming (las cuatro de Nivel 0).
Por monto la concentración es mucho mayor: Electronics 46.90% y Travel
21.42% del dinero defraudado (Gift Cards solo 1.96% del monto pese a su
cuota más alta: mucho fraude chico). La cuota mide la tasa; la cuota de
fraude por conteo/monto mide dónde está el fraude: son dos ejes.

![Cuota de fraude por merchant_category](salida-fraude/fig-cuota-fraude-categoria.png)

## 4. Serie temporal: DATOS INSUFICIENTES (honesto)

No hay serie temporal real por entidad para ajustar una tendencia
temporal. Evidencia medida: correlación de `id` con `time_of_day_hour`
−0.002, con `day_of_week` 0.0005, con `is_fraud` −0.0006; tasa de fraude
por decil de `id` entre 1.59% y 1.73% (plana); media de
`time_of_day_hour` por decil de `id` entre 11.72 y 11.79 (plana). El `id`
es solo orden de fila, no tiempo. No hay fecha, ni id de tarjeta, ni id
de comercio; `time_of_day_hour` y `day_of_week` son cíclicos. Sin serie
por entidad no se ajusta ninguna tendencia temporal: se declara DATOS
INSUFICIENTES y se usan el conteo de señales + autenticación +
concentración por categoría como features. No se inventa ninguna serie.

## 5. Modelo: ¿las señales de riesgo añaden señal?

`HistGradientBoostingClassifier` (200 árboles, lr 0.08, 31 hojas), CV
estratificada de 5 pliegues, métrica ROC AUC. Las tablas de referencia
(cuota de fraude por categoría/canal) se calculan SOLO con el pliegue
de entrenamiento de cada fold: sin fuga. Baseline nulo (constante
0.01669953): AUC 0.5.

| Features | AUC media | std | AUC por pliegue |
|---|---|---|---|
| Solo crudas | 0.9449 | 0.0035 | [0.9428, 0.9471, 0.9498, 0.9451, 0.9396] |
| Solo señales de riesgo | 0.9075 | 0.0054 | [0.9018, 0.9097, 0.9145, 0.9109, 0.9006] |
| Combinadas (crudas + señales) | 0.9448 | 0.0031 | [0.9424, 0.9462, 0.9495, 0.9455, 0.9406] |

**Lectura honesta: las señales de riesgo NO añaden señal sobre las
crudas en AUC.** El modelo combinado (0.9448) no supera al de solo
crudas (0.9449): la diferencia (−0.0001) está dentro del ruido (std
0.0031–0.0035). Las features de señales son transformaciones de
columnas crudas que el árbol ya puede usar. El valor de las señales
aquí es la lectura (conteo, autenticación, concentración por
categoría), no el AUC. Con solo 18 features de señales se llega a
0.9075, pero el árbol con las crudas ya captura esa información.

El envío usa el modelo combinado entrenado con TODO train (el modelo
que el notebook entrena al final); con solo crudas el envío sería
equivalente en CV. La decisión de qué modelo enviar es de Elán.

## 6. Reglas de la competencia (vinculantes) y qué implican

- **Envíos:** máximo 10 por día por equipo (Rules Sec. 2.2; el Overview
  decía 5, el texto vinculante son 10). Hasta 2 envíos finales
  seleccionables. Equipos de máximo 3 personas. Implicación: hay
  margen para 10 envíos/día, pero no se debe usar para sobreajustar al
  leaderboard público.
- **Código:** no es competencia de código para puntuar, pero los
  ganadores deben entregar código de entrenamiento + código de
  inferencia + documentación (Sec. 8.3). Implicación: este módulo y el
  notebook son ese código; se conservan limpios y reproducibles.
- **Datos externos:** permitidos si son públicos, gratuitos y
  accesibles para todos (Sec. 3.1). Ver §7.
- **Leaderboard:** el público es solo una muestra del test; el PRIVADO
  decide el resultado final. Implicación: la selección de modelo se
  hace por CV local (5 pliegues), no por envíos repetidos ni por
  sobreajuste al leaderboard público.
- **Prohibido:** AutoML (salvo licencia que lo permita como
  componente); etiquetar a mano o con ML el test/validación;
  información privada o exploits; redistribuir los datos de la
  competencia; compartir código en privado fuera del equipo. Este
  adaptador NUNCA entrena con el test ni infiere sus etiquetas.
- **Premio:** mercancía de Kaggle, valor $0; licencia del ganador
  "None" (no se cede la solución).
- **Cierre:** 30 nov 2026, 11:59 PM UTC. Sin envíos hechos todavía.
- **Datos:** todo sintético, inspirado en el original; licencia de la
  competencia CC BY 4.0.

## 7. Datos externos (original CC0): PROBADO y DESCARTADO

El dataset original "Credit Card Fraud Detection 2026" (CC0, dominio
público, 20,000 filas, fraude 1.695%, 339 fraudes, cero nulos, mismas
columnas salvo `transaction_id` en vez de `id`) se descargó con sesión
de Kaggle a `datos/credit-card-fraud-detection-2026.csv` y se corrió la
variante comparada (`evaluar_con_externo()`, celda 10 del notebook):
comparación de distribución + CV de 5 pliegues en los 3 modos,
validando SOLO sobre pliegues de la competencia (el externo solo entra
como entrenamiento extra, nunca como validación).

**Cambio de distribución (medido):** monto medio 132.42 vs 79.89 USD;
distancia media 22.14 vs 13.59 km; saldo medio 3,316.66 vs 2,065.00 USD;
sin autenticación 4.44% vs 6.44%; intento de IA 9.07% vs 15.22%; Gift
Cards 4.38% vs 7.30%; Crypto 4.99% vs 6.53%.

| Modo | AUC con externo | std | AUC sin externo | std | Δ vs mejor sin externo |
|---|---|---|---|---|---|
| Crudas | 0.9449 | 0.0036 | 0.9449 | 0.0035 | +0.0000 |
| Señales | 0.9074 | 0.0053 | 0.9075 | 0.0054 | −0.0375 |
| Combinadas | 0.9448 | 0.0034 | 0.9448 | 0.0031 | −0.0001 |

**Veredicto: NO AYUDA.** Ninguna variante con externo supera al mejor
sin externo (crudas 0.9449) por más de la std entre pliegues: el
externo (2.9% del entrenamiento) no mueve el AUC en ningún modo. El
externo se descarta como datos extra y el `submission.csv` queda
intacto (md5 `c07bfd803b2704a13ec7f7d2afff7bd8`, el mismo que antes de la
comparación). Detalle en `verificacion/comparacion_externo.json` y 11
checks nuevos en `verificacion/verificacion-fraude.json` (50/50 OK).

## 8. Verificación

- **Datos:** train 699,960 × 26, test 299,984 × 25, sample 299,984 × 2;
  tasa de fraude 1.67% (11,689); cero nulos en train y test.
- **Señales re-medidas** (no dadas por buenas): Sin Autenticación
  6.44%, Biométrico 0.50%, intento IA 15.22%, Gift Cards 7.30%,
  Crypto 6.53%, Gaming 5.51%, Groceries 1.11% — todas exactas.
- **Conteo de señales:** tasa monótona creciente con el conteo; umbral
  ≥ 5: 549 txns, 87.61% de fraude, 4.1% de fraudes capturados.
- **Cuota:** Gift Cards 0.0787, Crypto 0.0698, Groceries 0.0112;
  Nivel 0 = 4 categorías; Electronics 46.9% del monto de fraude.
- **Serie temporal:** DATOS INSUFICIENTES, con la evidencia medida (§4).
- **Distancia plana:** p90 1.72% vs resto 1.66%.
- **Notebook:** 12 celdas (6 código + 6 markdown), corre de arriba a
  abajo; su celda final imprime 12/12 verificaciones PASS.
- **Envío:** 7/7 checks OK contra `sample_submission.csv` (columnas
  exactas, 299,984 filas, mismos ids, mismo orden, sin NaN,
  probabilidades en [0, 1], sin ids duplicados).
- **Verificación independiente:** `verificacion/verificar_fraude.py`
  re-calcula todo desde los CSV y escribe
  `verificacion/verificacion-fraude.json`.
- **Comparación con externo:** 11 checks nuevos en
  `verificacion/verificacion-fraude.json` (50/50 OK total);
  `verificacion/comparacion_externo.json` con la comparación completa.

## 9. Gaps (registrados, nada inventado)

- **Serie temporal:** sin serie temporal por entidad (§4).
- **Intermediario:** no medible (no hay intermediario identificado).
- **Distancia:** plana en estos datos (1.72% vs 1.66%); se mantiene en
  el conteo por definición, con la variante medida.
- **channel / card_type / device_type:** planos (fraude 1.46%–1.74% en
  todas las clases): no discriminan.
- **day_of_week / time_of_day_hour:** planos (1.58%–1.83%): no hay
  patrón horario ni semanal de fraude en estos datos.
- **Datos externos:** el original CC0 se probó en §7 (variante
  comparada; veredicto: NO AYUDA, descartado).

## 10. Cómo enviar (cuando Elán lo decida)

1. Subir `submission.csv` a la competencia (máximo 10 envíos/día;
   hasta 2 finales seleccionables).
2. El notebook `fraude-tarjetas.ipynb` y `fraude_tarjetas.py` son el
   código de entrenamiento + inferencia + documentación que se entrega
   si se gana (Sec. 8.3).
3. No enviar el mismo modelo dos veces esperando que el leaderboard
   público suba: la selección se hizo por CV local (§5).

## 11. ASI por tipo de tarjeta (descriptivo, no es feature del modelo)

Definición del repo: **ASI = δH · α / F**. Se calcula por tipo de
tarjeta sobre train (699,960 filas) con estos proxies documentados:

- **δH:** entropía de Shannon normalizada [0,1] de la distribución de
  `merchant_category` DENTRO de cada tipo de tarjeta (diversidad de
  contextos que enfrenta la tarjeta). H = −Σ p·log2(p), normalizada por
  log2(12) (12 categorías).
- **α:** proporción de transacciones con autenticación iniciada por el
  usuario (Biometric / PIN / OTP) sobre el total de ese tipo. 3D Secure
  NO se cuenta (es un reto del protocolo / del comercio) y "No
  Authentication" tampoco. Valores reales de `auth_method` en train:
  3D Secure 210,036; OTP 175,178; PIN 154,020; Biometric 93,666;
  No Authentication 67,060.
- **α2 (sensibilidad):** proporción de transacciones en comercios
  conocidos (`is_new_merchant=False`) sobre el total de ese tipo.
- **F:** coeficiente de variación (std/media, std muestral ddof=1) de
  `amount_usd` dentro de cada tipo de tarjeta.

| Tipo | n | δH | α | α2 | F | ASI | ASI2 | Fraude |
|---|---|---|---|---|---|---|---|---|
| Amex | 45,797 | 0.878194 | 0.608053 | 0.846278 | 2.532080 | 0.210889 | 0.293512 | 1.90% |
| Discover | 17,235 | 0.878911 | 0.610270 | 0.848100 | 2.510580 | 0.213645 | 0.296905 | 1.46% |
| Mastercard | 254,857 | 0.871802 | 0.601263 | 0.849151 | 2.583018 | 0.202934 | 0.286599 | 1.60% |
| RuPay | 37,070 | 0.878422 | 0.605746 | 0.848071 | 2.529043 | 0.210396 | 0.294564 | 1.68% |
| Visa | 345,001 | 0.878736 | 0.605239 | 0.846421 | 2.568393 | 0.207073 | 0.289590 | 1.70% |
| TODAS | 699,960 | 0.876329 | 0.604126 | 0.847534 | 2.568102 | 0.206150 | 0.289209 | 1.67% |

**Lectura:** los cinco tipos son casi indistinguibles en los tres ejes:
δH 0.872–0.879 (la mezcla de comercios es prácticamente la misma en
todos los tipos), α 0.601–0.610, F 2.51–2.58. El ASI queda en
0.203–0.214 (ASI2 0.287–0.297), con Discover arriba y Mastercard abajo;
la dispersión entre tipos es de ~0.011, ~5% relativa. No hay relación
monótona con la tasa de fraude (Discover tiene el ASI más alto y la
tasa de fraude más baja, 1.46%; Amex tiene el segundo ASI más alto y la
tasa más alta, 1.90%): no se lee nada de fraude a partir del ASI.
Coherente con §9: `card_type` es plano en estos datos.

**Advertencias honestas (vinculantes):**

1. El umbral ASI > 1 fue calibrado en el cohorte Elite de HackerEarth
   2026; NO se transfiere mecánicamente a fraude. Aquí el ASI es
   descriptivo/comparativo, no un veredicto de soberanía. Además, con
   F ≈ 2.5 en estos datos, ASI > 1 es aritméticamente imposible por
   construcción (δH ≤ 1 y α ≤ 1, así que ASI ≤ 1/F ≈ 0.4): el umbral
   no puede cruzarse aquí aunque la tarjeta fuera perfectamente
   diversa y autónoma.
2. Como feature del modelo, el ASI por tipo de tarjeta son solo 5
   valores constantes (uno por tipo) y no aporta nada más allá de lo
   que `card_type` ya da al modelo; su valor es interpretativo, no de
   AUC. No entra al submission salvo que la CV demuestre lo contrario
   bajo la regla de decisión (superar 0.9449 por más de la std ~0.0035).
3. El proxy de α es interpretativo: en el repo α es eventos iniciados
   por el usuario vs impuestos por el sistema; aquí no hay log de
   eventos, así que se usa el método de autenticación como proxy, y
   la exclusión de 3D Secure de "iniciada por el usuario" es una
   decisión documentada, no un dato del dataset.

Verificación: los 6 renglones se re-calcularon por vía independiente
(`scipy.stats.entropy`, código separado del script) con coincidencia
exacta (< 1e-5 en δH, α, α2, F, ASI y ASI2). Detalle completo
(conteos por categoría y por método de autenticación, medias y std de
monto) en `verificacion/asi_por_tarjeta.json`.

## 12. Ajuste ligero (2026-11-08): NO GANA NADA

Orden de Elán (~17:10): "calibrar el instrumento, solo un ajuste
ligero". Corrida completa en 7,090.1 s (~1 h 58 min), 5-fold
estratificado, semilla 42. **Regla de decisión (vinculante):**
regenerar `submission.csv` solo si una variante supera 0.9449 por más
de la std entre pliegues (~0.0035); si no, el submission queda intacto
y todo lo probado se documenta como descartado.

**Diagnóstico de calibración (solo diagnóstico):** pendiente 0.9954,
intercepto −0.0038, log loss 0.0464, Brier 0.0112, media predicha
0.0166 vs observada 0.0167 — el modelo ya estaba bien calibrado.
Calibrar probabilidades NO mueve el ROC AUC (invariante a
transformaciones monótonas del puntaje).

**Rejilla HGB (10 configs, modelo combinado salvo nota):** base
0.9448 ± 0.0031; g1 0.9449 ± 0.0030; g2 0.9442 ± 0.0032; g3 0.9419;
g4 0.9389; **g5 0.9454 ± 0.0031 (mejor, +0.0005 = ruido)**; g6
0.9433 ± 0.0032; g7 0.9452 ± 0.0029; g8 0.9445 ± 0.0030; g9
0.9426 ± 0.0029. g5 en crudas: 0.9452 ± 0.0033. **LightGBM:**
0.9453 ± 0.0031 [0.9431, 0.9470, 0.9498, 0.9459, 0.9408] — también
ruido.

**Features extra (cada una por separado, mismo CV):**
ratio_monto_saldo 0.9447 ± 0.0031; monto_rel_categoria 0.9448 ± 0.0028;
log_monto_velocidad 0.9448 ± 0.0031; inter_velocidad_rafaga 0.9448;
inter_distancia_extranjero 0.9448; inter_monto_riesgo 0.9449 ± 0.0028;
inter_cvv_disputas 0.9449 ± 0.0030; inter_horas_velocidad 0.9447.

**Desviaciones vs patrón del grupo (idea de Elán: los usuarios tienen
patrones y las anomalías se notan; sin ID de usuario, desviación vs
grupo merchant_category × card_type × channel, 300 grupos):**
z_monto_categoria 0.9448; dev_monto_cat_tarjeta 0.9448;
dev_velocidad_canal 0.9447; dev_distancia_tarjeta 0.9448;
dev_horas_categoria 0.9450 ± 0.0031 (+0.0001, la mejor desviación,
sigue siendo ruido); dev_hora_circular 0.9447.

**Veredicto: NO GANA NADA.** Ninguna variante (rejilla, LightGBM,
features, interacciones, desviaciones) supera 0.9449 por más de la
std. El submission NO se regeneró y queda intacto (md5
`c07bfd803b2704a13ec7f7d2afff7bd8`, el mismo que se envió a Kaggle:
AUC público 0.94375). Detalle completo en
`verificacion/ajuste_ligero.json` y
`verificacion/ajuste-ligero-corrida2.log`.

Nota de procedencia: la corrida terminó a las ~19:49 pero el agente
que la lanzó quedó en silencio sin entregar su reporte; el veredicto
se cerró desde los logs y `ajuste_ligero.json` verificados (todos los
números de esta sección salen de ahí).
