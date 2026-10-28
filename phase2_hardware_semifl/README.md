# Aprendizaje federado centralizado: 1 servidor + 4 clientes (canal seguro)

Regresión logística federada sobre **Adult (UCI Census Income)**, agregación **FedAvg**, experimentos **IID** y **non-IID**, tolerancia a desconexiones y reconexión. Todo viaja por el canal seguro ya existente (`COMUNICACION.md`).

> **Estado honesto.** Todo lo de este documento se ejecutó y verificó **en una sola computadora** (master y workers como hilos o procesos locales sobre `127.0.0.1`). **No se ha probado todavía entre cinco computadoras físicas ni con la tarjeta de red desactivada de verdad**: eso lo debe hacer el equipo siguiendo las secciones 5 y 7.

## 1. Qué hay y cómo encaja

```
                 MASTER (master_edge/)
     modelo global · FedAvg · evaluación en test independiente
                          │  canal seguro (DH + ECDSA + AES-GCM)
        ┌────────────┬────┴───────┬────────────┐
     WORKER_1     WORKER_2     WORKER_3     WORKER_4     (worker_node/)
   datos locales  datos locales datos locales datos locales  → entrenan localmente
```

| Módulo | Función |
|---|---|
| `common/protocol.py`, `keys.py`, `config.py` | Canal seguro (existente; solo se añadió *keepalive* TCP, ver §8) |
| `common/data.py` | Adult: carga, preprocesamiento, split train/val/test estratificado y huella del dataset |
| `common/particion.py` | IID, Dirichlet non-IID (con umbral de heterogeneidad JS), validación, guardado, estadísticas |
| `common/model.py`, `aggregation.py`, `metrics.py` | Regresión logística (numpy), FedAvg + validaciones, métricas (macro) y Jensen-Shannon |
| `common/fl_config.py` | Configuración (defaults < JSON < variables `SFL_*` < CLI) |
| `master_edge/coordinador.py`, `aggregator.py` | Rondas, quórum, timeouts, reconexión, eventos, checkpoints |
| `master_edge/run_server.py` | CLI del servidor real |
| `worker_node/fl_worker.py`, `run_worker.py`, `logistic_model.py` | Worker: entrenamiento local y CLI |
| `common/simulacion.py`, `scripts/run_local.py` | Simulación local (hilos, claves temporales) |
| `scripts/*` | Preparación de datos, resiliencia, comparación, lanzadores `fl.ps1` / `fl.sh` |
| `tests/` | pytest: unitarios, integración y extremo a extremo |

`CHANGE_REPO` (Fashion-MNIST + CNN PyTorch por HTTP) se estudió pero **no se reutilizó su código**: es incompatible con el canal seguro (mensajes ≤ 64 KB, un CNN de 207 k parámetros no cabe) y con las dependencias de esta fase. De allí se tomó la idea de FedAvg ponderado, quórum mínimo, timeout de ronda y reincorporación en la siguiente ronda.

### Decisiones técnicas
- **Dataset/modelo**: Adult (32 561 filas, 92 características tras preprocesar, 2 clases), regresión logística (92 pesos + sesgo ≈ 2.5 KB en JSON). Ya estaba en el repo (`data/adult.csv`) y la phase 1 usaba el mismo modelo. Sin GPU ni PyTorch. Mismo dataset, modelo, preprocesamiento e hiperparámetros en IID y non-IID.
- **Splits** (seed 42, estratificados): train 22 793 · val 3 256 · test 6 512. Solo las filas de *train* se reparten a los clientes. El test global lo evalúa el master.
- **IID**: permutación + `array_split`. **non-IID**: Dirichlet por clase (α=0.3 por defecto; menor α = más sesgo), con tamaño mínimo por cliente y **JS media ≥ `min_js` (0.05)** respecto de la distribución global; si tras 200 intentos no se logra, falla con error explícito.
- **Métricas**: Precision/Recall/F1 con **promedio macro**. Las métricas globales se calculan evaluando el modelo global en el test, no promediando métricas locales.
- **Política de participación por defecto: `min_clients = 2`** (como la referencia `CHANGE_REPO`, `min_fit_clients=2`). **Es un valor configurable que el equipo debe confirmar con el enunciado del curso**; con `--min-clients 4` se exige participación completa.

## 2. Instalación

Requisitos: Python ≥ 3.10 (verificado con 3.11.9), Windows o Linux, red local, TCP **8080** libre en el servidor. CPU basta.

```powershell
cd phase2_hardware_semifl
.\scripts\fl.ps1 setup            # crea .venv e instala requirements.txt
```
Linux: `scripts/fl.sh setup`. O manualmente: `pip install -r requirements.txt`.
Cada máquina necesita `../data/adult.csv` (o `SFL_DATA_CSV`). **Todos los comandos se ejecutan desde `phase2_hardware_semifl/`** (los lanzadores `fl.ps1`/`fl.sh` lo hacen solos).

Preparar datos y particiones (opcional, se generan solas): `.\scripts\fl.ps1 data` · `.\scripts\fl.ps1 partitions`. Guardan JSON/CSV/PNG en `results/partitions/`.

## 3. Prueba rápida en una sola computadora (simulación local)

```powershell
.\scripts\fl.ps1 test         # tests
.\scripts\fl.ps1 iid          # → results/iid/
.\scripts\fl.ps1 noniid       # → results/non_iid/
.\scripts\fl.ps1 disconnect   # baseline + caída simulada → results/baseline/, results/disconnect/
.\scripts\fl.ps1 compare      # → results/comparison/
```
Equivalentes sin lanzador: `python -m pytest -q`, `python -m scripts.run_local --scheme iid --run-name iid`, `python -m scripts.run_local --scheme noniid --run-name non_iid`, `python -m scripts.run_resiliencia`, `python -m scripts.comparar`. Una ejecución previa de un mismo nombre no se borra: se mueve a `results/_archive/`.

## 4. Claves (una vez, por máquina)

Cada nodo genera solo la suya; la privada nunca sale de la máquina (`keys/` está en `.gitignore`).

```powershell
python -m common.keys --genkey MASTER       # en el servidor
python -m common.keys --genkey WORKER_1     # en cada cliente: WORKER_1 … WORKER_4
```
Cada comando imprime una línea `--addpub ID=04…`. El **master** ejecuta las 4 líneas de los workers; **cada worker** ejecuta la línea del master. Verifica con `python -m common.keys --list` que las huellas coincidan. Los IDs deben ser exactamente `WORKER_1…WORKER_4` en esta configuración (la partición *i* va al *i*-ésimo ID en orden alfabético).

## 5. Ejecución distribuida real (5 computadoras)

| PC | Rol | Qué necesita |
|---|---|---|
| 1 | Servidor | repo, `data/adult.csv`, `keys/MASTER.key` + públicas de los 4 workers |
| 2–5 | `WORKER_1…4` | repo, `data/adult.csv`, `keys/<ID>.key` + pública del MASTER |

1. **IP del servidor**: `ipconfig` (Windows) / `ip addr` (Linux). Úsala en los clientes; **no `localhost`**.
2. **Firewall del servidor** (solo ese puerto, sin desactivar el firewall), PowerShell como administrador:
   `New-NetFirewallRule -DisplayName "SFL 8080" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow`
3. **Conectividad** desde cada cliente: `Test-NetConnection <IP_SERVIDOR> -Port 8080` (debe dar `TcpTestSucceeded : True` con el servidor ya arrancado).
4. **Orden de arranque**:
   1. Servidor (PC 1): `.\scripts\fl.ps1 server iid` (o `noniid`). Equivale a `python -m master_edge.run_server --scheme iid --run-name iid`.
   2. Clientes 1 a 4 (PC 2–5): `.\scripts\fl.ps1 client WORKER_1 <IP_SERVIDOR>` (equivale a `python -m worker_node.run_worker --id WORKER_1 --master <IP>`). También sirve `SFL_MASTER_HOST`.
   3. El servidor espera hasta `connect_timeout` (120 s) a que **los 4** se registren (`registered` en `events.csv`/consola), reparte las particiones, espera su confirmación y **arranca solo**: no hay paso manual de “iniciar experimento”. El orden de arranque de los workers no importa: reintentan cada 4 s.
   4. Al terminar el servidor envía `fin` y los workers salen solos. Resultados en `results/<run-name>/` **del servidor**; los logs de cada worker, en `results/worker_logs/<ID>.log` de su máquina.
5. **IID y non-IID**: solo cambia `--scheme` (`iid`/`noniid`) y el `--run-name` (resultados separados); el resto de parámetros son los mismos. Después, `python -m scripts.comparar` en el servidor.

Ajustes útiles (servidor): `--rounds 10 --local-epochs 1 --batch-size 64 --lr 0.1 --min-clients 2 --round-timeout 30 --round-retries 1 --alpha 0.3 --seed 42`. Todos están en `common/fl_config.py`; también como `SFL_<NOMBRE>` o `--config archivo.json`. **`--seed`, `--val-frac` y `--test-frac` deben coincidir en master y workers** (si no, la huella del dataset difiere y el master rechaza al worker: `register_rejected`).

## 6. Qué se transmite (y qué no)

| Mensaje | Dirección | Contenido |
|---|---|---|
| `register` | W→M | huella del dataset, nº de características, si tiene partición, rondas hechas |
| `partition` / `partition_chunk` | M→W | **índices** de las filas de ese worker (no datos), histograma de clases |
| `partition_ack` | W→M | n e histograma recalculado con sus etiquetas locales |
| `round_start` | M→W | nº de ronda, versión, **W y b globales**, hiperparámetros, semilla |
| `update` | W→M | ronda, versión, n muestras, **W y b locales**, métricas locales agregadas, tiempo de entrenamiento |
| `sync` / `fin` | M→W | estado tras (re)conectar / cierre |

**No viajan**: filas del dataset, etiquetas individuales ni archivos. El ID del cliente es el autenticado en el handshake (no el que declare el mensaje). Un test (`test_extremo_a_extremo_cuatro_clientes`) comprueba que los mensajes de los workers solo contienen esos campos.

**Alcance de la seguridad**: el canal protege los parámetros *en tránsito* (autenticación mutua, cifrado, anti-replay). **No** implementa privacidad diferencial ni agregación segura criptográfica, y **el master ve en claro los pesos de cada cliente**, de los que se puede inferir información; un master malicioso no está cubierto. Federar los datos no garantiza privacidad por sí solo. Además, el master conoce los índices y por tanto las etiquetas del reparto (el reparto se genera como paso de preparación; el bucle federado nunca usa filas ni etiquetas de entrenamiento) y las estadísticas de estandarización salen de todo el train (simplificación documentada en `common/data.py`).

## 7. Política de participación, timeouts y desconexión

**Ronda** (`Coordinador.ejecutar_ronda`): cada intento tiene un `round` único y creciente.
1. Se espera hasta `round_timeout` a tener ≥ `min_clients` workers listos (conectados + registrados + con partición).
2. Se envía `round_start` a los listos (*esperados*). Se cierra la ronda cuando responden todos o vence `round_timeout`.
3. Una actualización se acepta solo si coinciden `round` y `version`, el worker era esperado, no es duplicada, `n>0`, y W/b tienen la forma correcta y son finitos. Si no, se rechaza y se registra (`stale_update_rejected`, `invalid_update_rejected`). Tras cerrar, la ronda rechaza cualquier actualización tardía.
4. Con ≥ `min_clients` válidas: FedAvg **solo con esos clientes** (los pesos `n_k/Σn` se renormalizan), versión +1, checkpoint. Quien no respondió queda `timeout`/`disconnected` en `client_metrics.csv`.
5. Con menos: **ronda fallida**; el modelo no cambia, se espera `retry_interval` y se reintenta. Tras más de `round_retries` fallos seguidos, **abortar de forma controlada** (`experiment_aborted`, código de salida 1). Nunca se espera indefinidamente.
6. Un worker que vuelve: se registra de nuevo → `reconnected`, el master lo **sincroniza** (`sync`: ronda y versión actuales, reenvía la partición si la perdió) y **entra en la siguiente ronda que se abra** (`participation_recovered`). Nunca se incorpora a una ronda en curso.

Con `--min-clients 4` la ronda se agrega solo con los cuatro; con 2, continúa con 2 o 3.

**Detección de un cliente caído sin cierre limpio** (tarjeta de red desactivada): el *timeout de ronda* lo excluye de inmediato; además el *keepalive* TCP (ajustado, `SFL_KEEPALIVE_*`; ~20–40 s según SO) cierra la sesión muerta y genera `disconnected`.

### Prueba física de desconexión (paso a paso)
1. Servidor y 4 clientes arrancados (§5) con **`--round-delay 3 --rounds 15`** (pausa entre rondas para que dé tiempo a actuar) y `--round-timeout 15`. Confirmar `registered` ×4.
2. Observar rondas con `expected=4 responded=4`.
3. En una ronda intermedia (p. ej. 4), **con autorización de quien controla esa PC**, desactivar manualmente su tarjeta de red (Configuración › Red › Deshabilitar). *El software no lo hace.*
4. Esperado en el servidor: `client_timeout` (ronda sin ese cliente, en ≤ `round_timeout`) y/o `disconnected` (tras el keepalive); la ronda se agrega con 3 (`round_aggregated clientes=[…3…]`) si `min_clients ≤ 3`; si `min_clients = 4`, `round_failed` + `round_retry` y, si no vuelve a tiempo, `experiment_aborted`.
5. Reactivar la tarjeta de red. El worker imprime `[..] Master … no disponible; reintento` y luego `Sesión segura`; el master registra `reconnected` + `synced` y, en la siguiente ronda abierta, `participation_recovered` (`expected=4`).
6. Revisar `results/<run>/events.csv`, `global_metrics.csv` (columnas `expected/responded/aggregated/duration_s`), `client_metrics.csv` (estado por cliente y ronda) y `python -m scripts.comparar` (gráfico de resiliencia si el run se llama `disconnect`; añadir un run `baseline` sin caída para compararlo).

**Fallo esperado vs. error real**: son esperados `client_timeout`, `disconnected`, `stale_update_rejected`, `round_failed` mientras el cliente está caído. Son errores reales: `register_rejected` (dataset/seed distintos), `partition_mismatch`, `invalid_update_rejected` de un cliente sano, trazas de Python, o que un cliente no vuelva tras reactivar la red (revisar IP/firewall).

## 8. Cambios al canal seguro existente
Solo uno, aditivo: `common/protocol.py::_keepalive_rapido` (+ constantes en `common/config.py`) acorta la detección de un par caído (por defecto el SO tarda ~2 h). El formato de trama y el handshake **no cambian**. `worker_node/client.py` ganó `reconnect_s`, `on_connect` y `desconectar()` (solo para simulación); `worker_node/receptor.py` ganó `EnsambladorParticion` (la función `recibir_particion` conserva su API). `generar_particion` aceptó `pool` y `min_js` (compatible hacia atrás).

## 9. Resultados (simulación local, una ejecución, seed 42)

Generados con `fl.ps1 iid|noniid|disconnect|compare`; leer `results/comparison/summary.csv` para los valores exactos.

| Indicador | IID | non-IID (α=0.3) |
|---|---:|---:|
| Accuracy final (test) | 0.8426 | 0.7961 |
| Precision macro | 0.7944 | 0.8608 |
| Recall macro | 0.7509 | 0.5807 |
| F1 macro | 0.7683 | 0.5809 |
| Pérdida final | 0.3352 | 0.4384 |
| Rondas completadas | 10 | 10 |
| Std entre clientes de acc. local del modelo global (media de rondas) | 0.005 | 0.318 |
| JS media de la partición vs. global | 0.000 | 0.126 |

Una regresión logística centralizada con los mismos datos alcanza ≈0.845 de accuracy, de modo que IID federado ≈ centralizado. Con non-IID el recall macro y el F1 caen (el modelo global se inclina hacia la clase mayoritaria de los clientes grandes) y el ajuste del modelo global a cada cliente es muy desigual. **Limitaciones**: una sola semilla y una sola partición; no hay barras de error ni se puede afirmar que la diferencia sea estadísticamente significativa; los tiempos de la simulación local (≈0.03 s/ronda) no reflejan una red real.

**Resiliencia (simulación local)**: `results/baseline/` frente a `results/disconnect/` (WORKER_3 cae `offline` 12 s desde la ronda 4, con `min_clients=2`): el cliente estuvo ausente de las rondas 4–9, el modelo se agregó con 3 clientes, volvió en la ronda 10, y la accuracy final fue 0.8420 frente a 0.8426. Es una sola ejecución con un modelo simple: no se atribuye causalidad.

## 10. Tests

`python -m pytest -q` → **55 pasados, 1 omitido** (`test_cinco_computadoras_fisicas`, marcado `multimachine`: requiere las 5 PCs). Cubren: particiones (conservación, solapamientos, reproducibilidad, heterogeneidad, error si no se alcanza), FedAvg con valores conocidos, validación de rondas/versiones/duplicados/tardías/no finitos, métricas, entrenamiento local, serialización y errores de comunicación, configuración inválida, 4 clientes de extremo a extremo (IID y non-IID), timeout con actualización tardía, caída/reconexión/sincronización/reincorporación, ronda sin quórum (recuperación y aborto), worker no autorizado. Los de integración usan **sockets reales pero en una sola máquina**.

## 11. Solución de problemas

| Síntoma | Causa / acción |
|---|---|
| Worker: `Master … no disponible; reintento` | IP incorrecta, servidor apagado o firewall; `Test-NetConnection <IP> -Port 8080` |
| `dispositivo NO AUTORIZADO` / `falta la clave pública de 'MASTER'` / firma inválida | faltan o no coinciden los `--addpub`; comparar `--list` |
| Servidor: `solo X de 4 workers se registraron` | faltan workers/ID incorrecto; subir `--connect-timeout` |
| `register_rejected … NO coincide` | distinto `adult.csv`, `--seed`, `--val-frac` o `--test-frac` entre master y worker |
| `Dataset ausente` (`DatasetError`) | colocar `data/adult.csv` o `SFL_DATA_CSV` |
| `ValueError … solapamiento/pierde/solo N muestras`, `RuntimeError … intentos` | partición inválida o JS no alcanzada: subir α, bajar `--min-js`/`--min-partition-size`; borrar el JSON de `results/partitions/` si es de otro dataset |
| `invalid_update_rejected` / `dimension incompatible` | versiones de código distintas entre máquinas |
| `client_timeout` constante | `--round-timeout` corto para la velocidad real; subirlo |
| Cliente no se reconecta | red aún caída, o el master ya terminó; ver log del worker |
| Faltan gráficos | `pip install matplotlib`; se regeneran con `python -m scripts.comparar` o `reporting.graficos_experimento(dir)` |
| El modelo no mejora | revisar `lr`, `local_epochs`; en non-IID extremo es esperable un rendimiento peor |

## 12. Reparto sugerido (4 integrantes) y contratos
1: servidor/rondas/FedAvg (`master_edge/`, `common/aggregation.py`). 2: clientes y comunicación segura (`worker_node/`, `common/protocol.py`). 3: datos y particiones (`common/data.py`, `particion.py`). 4: tests, resultados, gráficos, resiliencia y documentación (`tests/`, `common/reporting.py`, `scripts/`). Contratos compartidos: los mensajes de la §6 (docstring de `master_edge/coordinador.py`), el manifiesto de partición (`particion.construir_manifiesto`) y las columnas de los CSV (`coordinador.CAMPOS_*`). Material de exposición: `PRESENTACION.md`.
