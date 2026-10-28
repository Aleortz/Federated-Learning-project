# Apoyo para la exposición (10 min)

Cifras: simulación local, seed 42, una ejecución (ver `results/comparison/summary.csv`). Rehacerlas con los resultados de la demo física si se dispone de ellos.

## Guion
1. **Problema y dataset (1 min)**: predecir si el ingreso es >50K (Adult, 32 561 filas, 92 características, clases 76/24 %). Modelo: regresión logística.
2. **Arquitectura (1 min)**: 1 servidor + 4 clientes, canal seguro (DH-2048, ECDSA mutua, AES-256-GCM, anti-replay). `master_edge/` y `worker_node/`.
3. **Ronda federada (1.5 min)**: el servidor envía W, b → cada cliente entrena 1 época local con SGD en sus datos → devuelve W, b, n → FedAvg → evaluación en un test independiente del servidor. 10 rondas.
4. **Qué queda local / qué viaja (1 min)**: filas y etiquetas nunca salen; viajan índices de partición (al inicio), pesos del modelo, n y métricas agregadas.
5. **FedAvg (0.5 min)**: `w = Σ (n_k/Σn) w_k` sobre los clientes con actualización válida (`common/aggregation.py`, test con valores conocidos).
6. **IID vs non-IID (1.5 min)**: IID = permutación (≈5 698 muestras y 24 % de positivos por cliente; JS≈0). non-IID = Dirichlet α=0.3 (clientes de 14 539 a 601 muestras; 1 % a 99 % de positivos; JS=0.126). Gráfico `results/comparison/plots/partition_comparison.png`.
7. **Resultados (1.5 min)**: accuracy 0.843 (IID) vs 0.796 (non-IID); F1 macro 0.768 vs 0.581; la variabilidad entre clientes pasa de 0.005 a 0.318.
8. **Demo de desconexión (1.5 min)**: desactivar la red de un cliente; el servidor lo excluye por timeout, agrega con los demás, y al volver lo sincroniza y lo reincorpora en la siguiente ronda (`events.csv`).
9. **Limitaciones y conclusiones (0.5 min)**: una semilla; sin privacidad diferencial ni agregación segura; el master ve los pesos; regresión logística sencilla.

## Preguntas probables y respuestas (basadas en el código)
- **¿Centralizado vs federado?** En el centralizado los datos se juntan en un sitio; aquí cada cliente entrena con los suyos y solo envía parámetros. Con IID el resultado (0.843) es casi el del centralizado (≈0.845).
- **¿Por qué los datos no van al servidor?** El servidor solo recibe los mensajes de la tabla de README §6; el servidor descarta las filas de entrenamiento y solo conserva val/test. Un test verifica los campos permitidos.
- **¿Qué transmite cada cliente?** `register`, `partition_ack` y `update` (W, b, n, métricas agregadas, tiempo).
- **¿Cómo se calcula FedAvg?** Media de los pesos ponderada por n_k de los clientes válidos; los pesos suman 1 (`fedavg()`); si falta un cliente, se renormaliza.
- **¿Cómo se construyó IID?** Permutación aleatoria del conjunto de entrenamiento y `array_split` en 4; se valida que no haya solapes ni pérdida y se muestra la tabla de clases real.
- **¿Cómo se demuestra que non-IID es heterogéneo?** Tabla de clases por cliente y divergencia de Jensen-Shannon respecto de la distribución global: 0.126 vs 0.000; el código exige JS ≥ 0.05 o falla.
- **¿Por qué cambia el rendimiento?** Cada cliente optimiza su propia distribución y los modelos locales divergen (*client drift*); con clientes casi de una sola clase, el promedio sesga hacia la mayoritaria (recall macro 0.58). No es una ley general: con otra semilla o α puede variar.
- **¿Qué pasa si un cliente se desconecta?** Vence `round_timeout`, se registra `client_timeout`, y si hay ≥ `min_clients` válidas se agrega con ellas; si no, la ronda falla y se reintenta o aborta.
- **¿Cómo se evita esperar indefinidamente?** Cada ronda tiene un plazo; los reintentos están acotados (`round_retries`); y el keepalive TCP detecta pares caídos.
- **¿Y cuando vuelve?** Se reconecta con handshake, se registra, recibe `sync` (ronda y versión) y entra en la siguiente ronda; su actualización tardía se rechaza por ronda obsoleta.
- **¿Qué garantiza el canal seguro?** Autenticación mutua, confidencialidad e integridad en tránsito, anti-replay. No protege de un master malicioso ni da privacidad diferencial.
- **¿El federado garantiza privacidad?** No: los pesos filtran información y el servidor los ve en claro.
- **¿Limitaciones?** Una sola ejecución por escenario, sin significancia estadística; simulación local no equivale a 5 PCs; preprocesamiento con estadísticas globales; `min_clients=2` es configurable.
