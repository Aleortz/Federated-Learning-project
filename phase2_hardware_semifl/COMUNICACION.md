# Capa de comunicación segura (master ↔ workers)

Autor: Bryan Amaya

Canal autenticado y cifrado por el que viajan los parámetros del modelo, la telemetría y el modelo global entre el master edge y los workers. La red Wi-Fi se trata como **no confiable**: toda la seguridad viene del protocolo.

Esta capa **no** implementa lógica de FL. `aggregator.py`, `scorer.py`, `cloud_sync.py`, `logistic_model.py` y `telemetry.py` la usan a través de dos clases: `MasterServer` y `WorkerClient`.

## Archivos

| Archivo | Contenido |
|---|---|
| `common/protocol.py` | Handshake, cifrado, key chain y codificación de mensajes JSON |
| `common/config.py` | ID e IP del master, puerto, carpeta de claves, tamaño máximo de mensaje |
| `common/keys.py` | Generación e intercambio de claves |
| `master_edge/server.py` | `MasterServer` y modo de prueba por consola |
| `worker_node/client.py` | `WorkerClient` y modo de prueba por consola |
| `scripts/atacante.py` | Pruebas de seguridad (ataques) |

Requiere `pip install cryptography` (ya está en `requirements.txt`). **Todos los comandos se ejecutan desde `phase2_hardware_semifl/`**, porque los módulos importan `common`.

## Uso desde el código de FL

### En el master

```python
from master_edge.server import MasterServer

srv = MasterServer()                    # usa keys/MASTER.key y keys/authorized.txt
srv.start()
srv.wait_workers(3, timeout=60)         # espera a 3 workers autenticados

srv.broadcast({"type": "global", "round": r, "W": W, "b": b})   # acepta arrays de numpy

worker_id, msg = srv.recv(timeout=30)   # (ID autenticado, dict) o None si vence el timeout
```

También está `srv.send(worker_id, msg)` para un worker concreto y `srv.workers()` para ver quién está conectado. Si se pasa `on_message=func` al constructor, `func(worker_id, msg)` se llama con cada mensaje en lugar de encolarlo.

### En el worker

```python
from worker_node.client import WorkerClient

cli = WorkerClient("WORKER_1", master_host="192.168.1.50")
cli.start()                             # se reconecta solo si se cae
cli.wait_connected(timeout=30)

msg = cli.recv(timeout=60)              # dict del master
cli.send({"type": "params", "round": msg["round"], "W": W_k, "b": b_k, "n": n_k})
```

### Mensajes

Cada mensaje es un `dict` con un campo `"type"`; el resto de los campos es libre. Los arrays de numpy se convierten solos a listas, y al recibirlos se reconstruyen con `np.array(msg["W"])`. Los floats viajan sin pérdida de precisión.

El grupo define los tipos de mensaje. Una propuesta:

| `type` | Dirección | Campos |
|---|---|---|
| `telemetry` | worker → master | cpu, ram, batería, etc. (para `scorer.py`) |
| `global` | master → worker | round, W, b |
| `params` | worker → master | round, W, b, n |
| `fin` | master → worker | — |

El ID que entregan `recv()` y `on_message` es el **autenticado en el handshake**: un worker no puede hacerse pasar por otro poniendo otro nombre dentro del mensaje.

Cada mensaje puede ocupar hasta 65535 bytes. Como referencia, el modelo de la phase 1 (103 coeficientes más el intercepto) ocupa unos 1.9 KB.

## Configuración de claves (una vez)

Cada nodo genera **solo su propia clave**. La privada (`keys/<ID>.key`) nunca sale de su PC y `keys/` está en el `.gitignore`.

```bash
python -m common.keys --genkey MASTER        # en la PC del master
python -m common.keys --genkey WORKER_1      # en cada PC worker, con su propio ID
```

Cada `--genkey` imprime una línea `python -m common.keys --addpub ID=04...` para compartir:

- **El master** ejecuta la línea de cada worker.
- **Cada worker** ejecuta la línea del master.

Para verificar, comparen las huellas de `python -m common.keys --list`: tienen que coincidir en todas las máquinas.

## Prueba manual

```bash
python -m master_edge.server                                  # PC master
python -m worker_node.client --id WORKER_1 --master <IP>      # cada worker
```

En el worker, cualquier texto o JSON que se escriba se envía al master. En el master, `@WORKER_1 texto` envía un mensaje a ese worker. `p` muestra el estado, `k` las cadenas de claves y `q` sale; `-v` activa los volcados hex.

En Windows hay que permitir el puerto TCP 8080 en el firewall de la PC del master. La IP del master se obtiene con `ipconfig`, y se puede fijar con `--master` o con la variable de entorno `SFL_MASTER_HOST`.

## Diseño criptográfico

| Función | Primitiva |
|---|---|
| Clave de sesión | Diffie-Hellman MODP 2048 (RFC 3526), efímero en cada sesión |
| Autenticación mutua | ECDSA P-256 sobre el transcript del handshake |
| Derivación | HKDF-SHA256 |
| Cifrado e integridad | AES-256-GCM, con la cabecera completa como AAD |
| Evolución de claves | Key chain: una clave nueva por mensaje |

```
Worker (initiator)                                   Master (responder)
  │── Hello: MAGIC, ID_W, N_W, Y_W ──────────────────────►│ ¿ID autorizado? si no → status 0
  │◄── status, ID_M, N_M, Y_M, SIG_M ─────────────────────│
  │  verifica SIG_M con la pública GRABADA del master      │
  │── SIG_W ──────────────────────────────────────────────►│ verifica con la pública de ID_W
  │◄── resultado ──────────────────────────────────────────│
     z = g^(x_W·x_M) mod p → HKDF → CK_W→M, CK_M→W, SID
```

`MK_n = HKDF(CK_n, "P1-MSG")` cifra el mensaje n y `CK_(n+1) = HKDF(CK_n, "P1-CHAIN")` avanza el estado. El receptor solo avanza su cadena después de autenticar el mensaje.

## Pruebas de seguridad

| Prueba | Cómo | Resultado esperado |
|---|---|---|
| Ciphertext alterado | `2` en la consola | RECHAZADO (GCM) |
| TAG alterado | `3` | RECHAZADO (GCM) |
| Replay | `4` | RECHAZADO (SEQ) |
| Remitente falsificado | `5` | RECHAZADO (ID_S) |
| Paquete forjado | `6` | RECHAZADO (GCM) |
| ID no autorizado | `python scripts/atacante.py <IP> no-autorizado` | status 0 |
| Suplantación sin clave | `python scripts/atacante.py <IP> suplantar WORKER_2` | firma inválida |
| Atacante interno | `python scripts/atacante.py <IP> miembro WORKER_2 WORKER_3` | firma inválida |
| Inyección sin handshake | `python scripts/atacante.py <IP> inyectar` | descartado |
| Master impostor (MITM) | `python scripts/atacante.py 0.0.0.0 servidor-falso` y apuntar un worker a esa IP | el worker corta |

Después de cada prueba, el siguiente mensaje normal tiene que ser aceptado: la sesión sigue sincronizada.

## Límites conocidos

- El canal protege los parámetros **en tránsito**. El master los ve en claro, porque los necesita para agregar. Esta capa no protege contra un master malicioso que intente inferir datos a partir de W_k; para eso está la perturbación de parámetros de la phase 1.
- No hay post-compromise security: quien obtenga `CK_n` puede derivar las claves siguientes hasta el próximo handshake.
- El master calcula DH y firma antes de comprobar que el worker tiene su clave privada (solo verifica que el ID esté autorizado). No hay rate limiting.
