# Fraude con tarjeta (Playground Series S6E11)

**Fractal Core Research** — Adaptador v1.0.0 (fraude_tarjetas.py). Generado por el adaptador, no a mano.

**Competencia:** Kaggle Playground Series S6E11, "Predicting Credit Card Fraud". Métrica: ROC AUC. Envío: CSV id,is_fraud (probabilidades 0–1). Cierre: 30 nov 2026, 11:59 PM UTC. Sin envíos hechos todavía.

**Datos:** train.csv 699,960 filas, test.csv 299,984 filas, sample_submission.csv 299,984 filas. Tasa de fraude en train: 1.67% (11,689 fraudes). Cero nulos en train y test. Todo sintético, inspirado en el original; licencia de la competencia CC BY 4.0.

## 0. Reglas de la competencia (vinculantes)

- **Envíos:** máximo 10 por día por equipo (Rules Sec. 2.2; el Overview decía 5, el texto vinculante son 10). Hasta 2 envíos finales seleccionables. Equipos de máximo 3 personas.
- **Código:** no es competencia de código para puntuar, pero los ganadores deben entregar código de entrenamiento + código de inferencia + documentación (Sec. 8.3). Este módulo y el notebook son ese código: se conservan limpios y reproducibles.
- **Datos externos:** permitidos si son públicos, gratuitos y accesibles para todos (Sec. 3.1). Candidato: el dataset original "Credit Card Fraud Detection 2026" (CC0, 20,000 filas). Sus distribuciones son cercanas, no idénticas: se valida el cambio de distribución antes de usarlo (ver §6).
- **Leaderboard:** el público es solo una muestra del test; el PRIVADO decide el resultado final. La selección de modelo se hace por CV local (5 pliegues), no por envíos repetidos ni por sobreajuste al leaderboard público.
- **Prohibido:** AutoML (salvo licencia que lo permita como componente); etiquetar a mano o con ML el test/validación; información privada o exploits; redistribuir los datos de la competencia; compartir código en privado fuera del equipo. Este adaptador NUNCA entrena con el test ni infiere sus etiquetas.
- **Premio:** mercancía de Kaggle, valor $0; licencia del ganador "None" (no se cede la solución).

## 1. Las features (medidas, no declaradas)

| Grupo | Definición (medida) |
|---|---|
| **Monto** | amount_usd (el dinero que se mueve; 1-6,870 USD, media 79.9) y account_balance_usd (el saldo de la cuenta). |
| **Autenticación** | auth_method. Con autenticación: Biometric 0.50%, PIN 0.87%, OTP 1.41%, 3D Secure 1.47% de fraude. Sin autenticación: No Authentication 6.44% (37.0% de todo el fraude con 9.58% de las txns). cvv_retry_count es forzar la verificación con reintentos (0 -> 0.93%, 1 -> 8.75%, 2 -> 41.73%, 3 -> 36.0%). |
| **Concentración por categoría** | merchant_category por concentración de fraude (cuota por conteo y por monto) + merchant_risk_score. Niveles por cuota de fraude: Nivel 0 >= 10%, Nivel 1 3-10%, Nivel 2 1-3%, Nivel 3 < 1%. |
| **Señales de riesgo** | La transacción que mueve el mismo monto sin autenticación ni verificación: transacción extranjera, comercio nuevo, VPN, mismatch de país, mismatch de facturación/envío, intento de estafa por IA, reintentos de CVV, velocidad alta, distancia alta. |
| **Intermediario** | No medible con estos datos (no hay intermediario identificado) — gap registrado, no inventado. |

**Ficha por categoría (merchant_category, por observables):** origen = categoría del comercio; posición = nivel de concentración por cuota de fraude; naturaleza = de alto riesgo si su cuota de fraude es alta y su autenticación es baja, de bajo riesgo si su cuota es baja; patrón = concentra fraude (cuota alta) o lo distribuye (cuota baja). Las fichas completas están en la tabla de cuota de fraude (§3).

## 2. Conteo de señales de riesgo distintas

Por transacción, se cuentan los tipos DISTINTOS de señal de riesgo entre 10 definidas (extranjero, comercio nuevo, VPN, mismatch de país, mismatch de facturación/envío, intento de estafa por IA, sin autenticación, reintento de CVV, velocidad sobre p90, distancia sobre p90). Umbral: >= 5 tipos distintos.

| Señales distintas | n | Tasa de fraude |
|---|---|---|
| 0 | 373,315 | 0.19% |
| 1 | 241,861 | 1.06% |
| 2 | 69,307 | 5.40% |
| 3 | 12,692 | 22.38% |
| 4 | 2,236 | 59.66% |
| 5 | 459 | 85.40% |
| 6 | 81 | 98.77% |
| 7 | 8 | 100.00% |
| 8 | 1 | 100.00% |

**Umbral >= 5 señales:** 549 transacciones (0.08% del total), tasa de fraude 87.61% frente a 1.60% sin llegar al umbral; captura 481 de 11689 fraudes (4.1%). Lift: 52.46x sobre la base. La tasa crece monótona con el conteo (0 señales: 0.19%; 3: 22.38%; 4: 59.66%; 5: 85.40%; 6: 98.77%; 7-8: 100%).

**Variante sin distancia:** la señal de distancia es PLANA (fraude 1.72% sobre p90 vs 1.66% bajo p90): no discrimina en estos datos. Sin ella, el umbral >= 5 son 388 txns con 94.07% de fraude y 3.1% de los fraudes capturados (vs 549 / 87.61% / 4.1% con distancia). La distancia añade 161 txns por encima del umbral con menor tasa: es ruido en estos datos, no señal. Se mantiene en el conteo por definición y se reporta la variante.

**El conteo de señales solo ya tiene AUC 0.8702** (Mann-Whitney sobre train): un solo entero, sin modelo, separa fraude de no-fraude mejor que merchant_risk_score solo (0.7489) o cvv_retry_count solo (0.7081). El umbral >= 5 solo tiene AUC 0.5205: es preciso (87.61%) pero raro (0.08% de las txns), así que su valor está en el conteo, no en el binario.

## 3. Cuota de fraude por merchant_category (transversal, sin tiempo)

Cuota de fraude = fraude / (total - fraude). Es una cuota por categoría, sin tiempo: no hay serie temporal aquí.

| Categoría | n | Fraude | Cuota | Cuota fraude (n) | Cuota fraude (USD) | Nivel |
|---|---|---|---|---|---|---|
| Gift Cards | 9,456 | 690 | 0.0787 | 5.90% | 1.96% | Nivel 1 |
| Crypto Exchange | 5,285 | 345 | 0.0698 | 2.95% | 10.91% | Nivel 2 |
| Gaming | 23,784 | 1,311 | 0.0583 | 11.22% | 3.36% | Nivel 0 |
| Electronics | 48,693 | 1,501 | 0.0318 | 12.84% | 46.90% | Nivel 0 |
| Streaming | 47,196 | 1,061 | 0.0230 | 9.08% | 1.09% | Nivel 1 |
| Travel | 28,715 | 534 | 0.0189 | 4.57% | 21.42% | Nivel 1 |
| Utilities | 42,027 | 581 | 0.0140 | 4.97% | 1.25% | Nivel 1 |
| Healthcare | 35,889 | 462 | 0.0130 | 3.95% | 1.09% | Nivel 1 |
| Fuel | 59,880 | 741 | 0.0125 | 6.34% | 1.49% | Nivel 1 |
| Restaurants | 101,360 | 1,159 | 0.0116 | 9.92% | 2.75% | Nivel 1 |
| Groceries | 142,578 | 1,585 | 0.0112 | 13.56% | 3.75% | Nivel 0 |
| Online Retail | 155,097 | 1,719 | 0.0112 | 14.71% | 4.05% | Nivel 0 |

**Lectura:** por conteo de fraude las categorías con más fraude son Online Retail (14.71%), Groceries (13.56%), Electronics (12.84%) y Gaming (11.22%) — las cuatro de Nivel 0. Por monto de fraude la concentración es mucho mayor: Electronics 46.90% y Travel 21.42% del dinero defraudado (Gift Cards solo 1.96% del monto pese a su cuota más alta: mucho fraude chico). La cuota mide la tasa; la cuota de fraude por conteo/monto mide dónde está el fraude: son dos ejes.

## 4. Serie temporal: DATOS INSUFICIENTES (honesto)

No hay serie temporal real por entidad para ajustar una tendencia temporal. Evidencia medida: correlación de id con time_of_day_hour -0.002, con day_of_week 0.0005, con is_fraud -0.0006; tasa de fraude por decil de id entre 1.59% y 1.73% (plana); media de time_of_day_hour por decil de id entre 11.72 y 11.79 (plana). El id es solo orden de fila, no tiempo.

**Lo que falta (exacto):**

- columna de fecha u orden temporal real (time_of_day_hour y day_of_week son cíclicos, no una línea de tiempo)

- id de tarjeta (no hay tarjeta que seguir en el tiempo)

- id de comercio (no hay comercio que seguir en el tiempo)

- serie temporal por entidad con n>=8 para ajustar una tendencia temporal (log-log)

**Lo que sí existe:**

- Cuota de fraude transversal por merchant_category y por channel (sin tiempo)

- Conteo de señales de riesgo distintas por transacción (10 tipos de señal)

- autenticación por auth_method (con vs sin autenticación)

- niveles de concentración por cuota de fraude por categoría

## 5. Modelo: ¿las señales de riesgo añaden señal?

HistGradientBoostingClassifier (200 árboles, lr 0.08, 31 hojas), CV estratificada de 5 pliegues, AUC media ± std. Las tablas de referencia (cuota de fraude por categoría/canal) se calculan SOLO con el pliegue de entrenamiento de cada fold: sin fuga. Baseline nulo (constante 0.01669953): AUC 0.5.

| Features | AUC media | std | AUC por pliegue |
|---|---|---|---|
| Solo crudas | 0.9449 | 0.0035 | [0.9428, 0.9471, 0.9498, 0.9451, 0.9396] |
| Solo señales de riesgo | 0.9075 | 0.0054 | [0.9018, 0.9097, 0.9145, 0.9109, 0.9006] |
| Combinadas (crudas + señales) | 0.9448 | 0.0031 | [0.9424, 0.9462, 0.9495, 0.9455, 0.9406] |

**Lectura:** combinadas 0.9448 vs solo crudas 0.9449 vs solo señales 0.9075. Las señales de riesgo NO añaden señal sobre las crudas en AUC: el modelo combinado no supera al de solo crudas (las features de señales son transformaciones de columnas crudas que el árbol ya puede usar). El valor de las señales aquí es la lectura (conteo, autenticación, concentración por categoría), no el AUC.

## 6. Datos externos (original CC0): variante comparada

Dataset original "Credit Card Fraud Detection 2026" (CC0, 20,000 filas). Tasa de fraude: externo 1.69% vs competencia 1.67%. El cambio de distribución es real: monto medio 132.4 vs 79.9 USD; distancia media 22.1 vs 13.6 km; sin autenticación 4.44% vs 6.44%; intento de IA 9.07% vs 15.22%; Gift Cards 4.38% vs 7.30%.

CV de 5 pliegues validando SOLO sobre pliegues de la competencia (el externo solo entra como entrenamiento extra):

| Modo | AUC con externo | std | AUC sin externo | std | Δ vs mejor sin externo |
|---|---|---|---|---|---|
| crudas | 0.9449 | 0.0036 | 0.9449 | 0.0035 | +0.0000 |
| senales | 0.9074 | 0.0053 | 0.9075 | 0.0054 | -0.0375 |
| combinadas | 0.9448 | 0.0034 | 0.9448 | 0.0031 | -0.0001 |

Mejor sin externo: crudas 0.9449. **Veredicto: NO AYUDA: ninguna variante con externo supera al mejor sin externo (0.9449) por más de la std entre pliegues; el externo se descarta como datos extra (se documenta, no se usa).**

## 7. Envío (submission.csv)

Modelo final (combinadas) entrenado con TODO train. Archivo: `salida-fraude/submission.csv` — 299,984 filas, columnas id,is_fraud. Verificación contra sample_submission.csv: 7/7 checks OK (sin fallos). Media de probabilidad predicha: 0.0166 (base 0.0167). **NO enviado a Kaggle: el envío lo decide Elán después.**

## 8. Gaps (registrados, nada inventado)

- **Serie temporal:** sin serie temporal por entidad (ver §4).
- **Intermediario:** no medible (no hay intermediario identificado).
- **Distancia:** plana en estos datos (1.72% vs 1.66%); se mantiene en el conteo por definición, con la variante medida.
- **channel / card_type / device_type:** planos (fraude 1.46%-1.74% en todas las clases): no discriminan.
- **day_of_week / time_of_day_hour:** planos (1.58%-1.83%): no hay patrón horario ni semanal de fraude en estos datos.
- **Datos externos:** el original CC0 se probó en §6 (variante comparada con CV; veredicto ahí).
