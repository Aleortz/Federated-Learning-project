"""
Worker federado: se conecta al master por el canal seguro, recibe su particion de INDICES,
entrena localmente en cada ronda y devuelve solo W, b, n y metricas agregadas.

Los datos (filas, etiquetas) se quedan en esta maquina: el master solo ve lo que viaja en
los mensajes "register", "partition_ack" y "update" (ver README, seccion Protocolo).

Mensajes que atiende (master -> worker): partition, partition_chunk, round_start, sync, fin.
Tras cada (re)conexion envia "register" y el master lo sincroniza; participa desde la
siguiente ronda que se abra (nunca se incorpora a una ronda ya empezada).

Uso (desde phase2_hardware_semifl/):
    python -m worker_node.run_worker --id WORKER_1 --master 192.168.1.50
"""
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from common import config, data
from worker_node.client import WorkerClient
from worker_node.logistic_model import EntrenadorLocal
from worker_node.receptor import EnsambladorParticion


@dataclass
class SimulacionCaida:
    """
    SOLO PARA PRUEBAS Y SIMULACION LOCAL. Al recibir la ronda `en_ronda` (o la primera
    posterior), una unica vez:
      modo "silent":  no responde durante `segundos` y despues manda su actualizacion (que
                      llega tarde y el master debe rechazar).
      modo "offline": cierra la sesion y no se reconecta durante `segundos`.
    La desconexion REAL de la demo se hace a mano, desactivando la tarjeta de red.
    """
    en_ronda: int
    modo: str = "offline"
    segundos: float = 10.0
    disparada: bool = False


class FLWorker:
    def __init__(self, worker_id, master_host, port=config.PORT, keydir=config.KEYDIR,
                 dataset=None, csv=data.ADULT_CSV, val_frac=0.10, test_frac=0.20, seed=42,
                 reconnect_s=config.RECONNECT_S, simulacion=None):
        self.id = worker_id
        self.log = logging.getLogger(f"worker.{worker_id}")
        self.ds = dataset or data.cargar(csv, val_frac, test_frac, seed)
        self.huella = self.ds.huella()
        self.sim = simulacion
        self.entrenador = None
        self.particion = None
        self.ens = EnsambladorParticion(labels=self.ds.y)
        self.rondas_hechas = 0
        self._fin = threading.Event()
        self.cli = WorkerClient(worker_id, master_host, port, keydir=keydir,
                                reconnect_s=reconnect_s, on_connect=self._registrar)

    # ---------------- ciclo de vida ----------------
    def start(self):
        self.cli.start()

    def stop(self):
        self._fin.set()
        self.cli.stop()

    def _registrar(self):
        try:
            self.cli.send({"type": "register", "data_fingerprint": self.huella,
                           "n_features": self.ds.n_features,
                           "has_partition": self.entrenador is not None,
                           "rounds_done": self.rondas_hechas})
            self.log.info("registrado ante el master (particion local: %s)",
                     "si" if self.entrenador else "no")
        except (OSError, ConnectionError) as e:
            self.log.warning("no se pudo registrar: %s", e)

    def run(self, max_idle=None):
        """Bucle principal; vuelve cuando el master envia 'fin' o se llama a stop()."""
        ultimo = time.time()
        while not self._fin.is_set():
            msg = self.cli.recv(timeout=0.5)
            if msg is None:
                if max_idle and time.time() - ultimo > max_idle:
                    self.log.error("sin mensajes del master en %.0f s: abandono", max_idle)
                    return False
                continue
            ultimo = time.time()
            tipo = msg.get("type")
            try:
                if tipo in ("partition", "partition_chunk"):
                    self._on_particion(msg)
                elif tipo == "round_start":
                    self._on_ronda(msg)
                elif tipo == "sync":
                    self.log.info("sync del master: ronda %s, version %s, estado %s",
                             msg.get("round"), msg.get("version"), msg.get("phase"))
                elif tipo == "fin":
                    self.log.info("el master finalizo el experimento (%s)", msg.get("reason"))
                    return True
                else:
                    self.log.debug("mensaje ignorado: %s", tipo)
            except (ValueError, KeyError) as e:
                self.log.error("mensaje '%s' invalido: %s", tipo, e)
        return False

    # ---------------- particion ----------------
    def _on_particion(self, msg):
        p = self.ens.agregar(msg)
        if p is None:
            return
        idx = np.asarray(p["indices"], dtype=np.int64)
        if not np.all(np.isin(idx, self.ds.train_idx)):
            raise ValueError("la particion incluye filas que no son de entrenamiento: rechazada")
        X, y = self.ds.subset(idx)
        self.entrenador = EntrenadorLocal(X, y)
        self.particion = p
        self.cli.send({"type": "partition_ack", "n": p["n"], "hist": p["hist"]})
        self.log.info("particion %s recibida: %d muestras, hist %s", p["scheme"], p["n"], p["hist"])

    # ---------------- ronda ----------------
    def _ultima_ronda(self, msg):
        """Si hay rondas mas nuevas ya en la cola, salta a la mas reciente."""
        while True:
            nuevo = self.cli.recv(timeout=0)
            if nuevo is None:
                return msg
            if nuevo.get("type") == "round_start" and nuevo["round"] > msg["round"]:
                self.log.info("salto de la ronda %d a la %d (ya iba atrasado)", msg["round"], nuevo["round"])
                msg = nuevo
            elif nuevo.get("type") == "fin":
                self._fin.set()
                return msg
            elif nuevo.get("type") in ("partition", "partition_chunk"):
                self._on_particion(nuevo)

    def _on_ronda(self, msg):
        if self.entrenador is None:
            self.log.warning("ronda %s recibida sin particion local: ignorada", msg.get("round"))
            return
        sim = self.sim
        if sim and not sim.disparada and msg["round"] >= sim.en_ronda:
            sim.disparada = True
            self.log.warning("SIMULACION de caida (%s, %.1f s) en la ronda %d", sim.modo, sim.segundos, msg["round"])
            if sim.modo == "offline":
                self.cli.desconectar(sim.segundos)
                return
            time.sleep(sim.segundos)          # modo silent: despues responde tarde A PROPOSITO
        else:
            msg = self._ultima_ronda(msg)

        rid = msg["round"]
        res = self.entrenador.ronda(msg["W"], msg["b"], msg["local_epochs"], msg["batch_size"],
                                    msg["lr"], msg["l2"], seed=msg["seed"])
        self.log.info("ronda %d (modelo v%d): %d muestras, perdida local %.4f, %.2f s",
                 rid, msg["version"], res["n"], res["metrics"]["train_loss"], res["train_time_s"])
        try:
            self.cli.send({"type": "update", "round": rid, "version": msg["version"], **res})
            self.rondas_hechas += 1
        except (OSError, ConnectionError) as e:
            self.log.warning("ronda %d: no se pudo enviar la actualizacion (%s); espero a reconectar", rid, e)
