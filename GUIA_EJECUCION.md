# Guía de ejecución: master y workers

Reparte a cada worker su partición de datos (IID o non-IID) por el canal seguro. Más detalle en `phase2_hardware_semifl/COMUNICACION.md`.

> Todos los comandos se ejecutan **desde `phase2_hardware_semifl/`**, porque los módulos importan `common`.

```
cd phase2_hardware_semifl
```

## 0. Requisitos (en todas las máquinas)

```
pip install numpy matplotlib cryptography
```

- Cada máquina necesita `data/adult.csv` en la raíz del proyecto. El worker lo usa para verificar su partición y el master para generarla.
- El **master** debe tener abierto el puerto TCP **8080** en el firewall.
- Las máquinas deben alcanzarse por red. Anota la IP del master (`ipconfig`).

## 1. Claves (una sola vez)

Cada nodo genera **solo su propia clave**. La privada (`keys/<ID>.key`) nunca sale de su máquina ni se sube a git.

**En el master:**

```
python -m common.keys --genkey MASTER
```

Imprime una línea `python -m common.keys --addpub MASTER=04...`. Compártela con los workers.

**En cada worker** (ID de 1 a 19 caracteres):

```
python -m common.keys --genkey WORKER_1
```

Imprime su línea `--addpub WORKER_1=04...`. Envíasela al master.

**Intercambio de públicas:**

- Cada worker ejecuta el `--addpub MASTER=...` que recibió.
- El master ejecuta el `--addpub` de cada worker.
- Comprueba con `python -m common.keys --list` que las huellas coincidan en ambos lados.

Opcional, para no repetir la IP del master en cada comando (en PowerShell: `$env:SFL_MASTER_HOST="192.168.1.50"`):

```
set SFL_MASTER_HOST=192.168.1.50
```

## 2. Ejecutar

### Master

```
python -m master_edge.distribuidor --scheme noniid --workers 3
```

| Opción | Significado |
|---|---|
| `--scheme iid\|noniid` | Tipo de reparto |
| `--workers N` | Workers a esperar; el reparto se hace en N particiones |
| `--alpha 0.05` | Sesgo non-IID (más bajo, más sesgo) |
| `--seed 42` | Semilla del reparto |
| `--timeout 120` | Segundos que espera a los workers |

Espera a los N workers, reparte, imprime la tabla de muestras por clase y guarda `results/particion_<scheme>.png`.

### Cada worker

```
python -m worker_node.receptor --id WORKER_1 --master 192.168.1.50
```

Se conecta con handshake cifrado, recibe su partición, verifica el histograma con su `adult.csv` local y responde al master. Imprime su propia tabla.

**Orden de arranque:** no importa. El worker reintenta cada 4 s y el master espera hasta 120 s.

## 3. Interpretar la salida

- **Master:** `[OK] todos los workers confirmaron su particion` indica que cada worker recibió exactamente lo enviado. `[X] <worker>: ack ausente o distinto` indica un fallo.
- **Worker:** muestra su número de muestras y su distribución de clases. En IID cada uno tiene cerca de 76 % / 24 %. En non-IID verás clases muy desbalanceadas.

## 4. Errores frecuentes

| Mensaje | Causa |
|---|---|
| `[CONFIG] ... MASTER.key` | Falta generar la clave del master |
| `falta la clave publica de 'MASTER'` | El worker no hizo `--addpub MASTER=...` |
| `dispositivo NO AUTORIZADO` | El master no tiene la pública de ese worker |
| `no disponible; reintento cada 4 s` | IP incorrecta, master apagado o firewall bloqueando el 8080 |
| `solo X de N workers conectados` | Faltan workers o `--workers` es mayor que los que hay |
| `el manifiesto tiene N particiones y hay M workers` | Hay más workers conectados que `--workers`; el master toma los N primeros por orden alfabético |

## 5. Prueba rápida en una sola máquina

Sin claves ni red, solo para ver la distribución:

```
python -m scripts.demo_particiones --workers 4 --alpha 0.05
```

Levanta un master y 4 workers locales con claves temporales, reparte IID y non-IID, imprime ambas tablas y guarda `results/particiones_iid_vs_noniid.png`.

## Notas

- La partición `i` va al `i`-ésimo worker en **orden alfabético de ID**. Con los mismos IDs, semilla y N, cada worker recibe lo mismo.
- Esta guía cubre solo el reparto de datos. El entrenamiento federado, FedAvg, IID/non-IID, desconexiones y resultados están en `phase2_hardware_semifl/README.md`.
